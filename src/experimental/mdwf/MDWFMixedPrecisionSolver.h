/*
 * Mixed-precision even/odd MDWF solves: single-precision CG iterations with
 * double-precision reliable updates (the coupled-5D counterpart of
 * ConjugateGradient::invert_mixed in src/modules/inverter/inverter.cpp,
 * after Sleijpen and van der Vorst; Clark et al., Comput. Phys. Commun. 181
 * (2010) 1517).
 *
 * MDWFCoupledMixedCG solves A x = b for the Hermitian positive operator of a
 * double adapter, iterating with the same operator in single precision:
 *
 *   per iteration (float):  s = A_f p_f, alpha = |r_f|^2 / <p_f, s>, r_f -= alpha s,
 *                           accum += alpha p (double), p_f = r_f + beta p_f, p = p_f;
 *   reliable update when |r_f| < delta |r| at the last update (or |r_f| reaches the target):
 *                           x += accum, accum = 0, r = b - A x (double), r_f = r,
 *                           p = r + beta (p - <p, r>/|r|^2 r), p_f = p.
 *
 * Convergence is decided on the double residual of a reliable update only, with
 * the stopping rule of MDWFCoupledCG (|r|^2 <= precision^2 max(|b|^2, 1)), so a
 * mixed and a double solve stop at the same true residual. Iterations counts the
 * single-precision operator applications; each reliable update adds one double
 * application.
 *
 * MDWFMobiusEvenOddMixedSolver has the interface of MDWFMobiusEvenOddSolver
 * (solve / solveDagger / solveNormal, lastIterations, lastResidue) with the
 * even/odd pre- and post-processing in double, a single-precision copy of the
 * gauge field and of the even/odd operator (clover blocks and block inverses
 * converted from the refreshed double operator on every refresh), and the mixed CG for the
 * Schur normal equation. Targets using it need SINGLEPREC=1 as well as
 * DOUBLEPREC=1.
 */

#pragma once

#include "MDWFMobiusEvenOdd.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

struct MDWFCoupledMixedCGResult {
    int iterations;          // single-precision operator applications
    int reliableUpdates;     // double-precision residual recomputations
    double residue;          // true relative residual sqrt(|b - A x|^2 / max(|b|^2, 1)) of the last update
    bool converged;
};

template<class AdapterD, class AdapterF, size_t BlockSize = 64>
class MDWFCoupledMixedCG {
public:
    using SpinorD = typename AdapterD::Spinor;
    using SpinorF = typename AdapterF::Spinor;

    MDWFCoupledMixedCGResult invert(AdapterD &adapterD, AdapterF &adapterF, SpinorD &x, const SpinorD &b,
                                    int maxIter, double precision, double delta, bool update = true) const {
        if (!(delta > 0.0 && delta < 1.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF mixed CG needs 0 < delta < 1, got ", delta));
        }
        CommunicationBase &comm = b.getComm();
        SpinorD r(comm, "MDWF_mixed_cg_resd");
        SpinorD p(comm, "MDWF_mixed_cg_srchd");
        SpinorD accum(comm, "MDWF_mixed_cg_accum");
        SpinorD ax(comm, "MDWF_mixed_cg_opx");
        SpinorF rF(comm, "MDWF_mixed_cg_resf");
        SpinorF pF(comm, "MDWF_mixed_cg_srchf");
        SpinorF sF(comm, "MDWF_mixed_cg_opsf");

        x = static_cast<double>(0.0) * b;
        accum = x;
        r = b;
        p = r;
        rF.convert_precision(r);
        pF.convert_precision(p);

        const double sourceNorm = adapterD.norm2(r);
        const double targetNorm = precision * precision * std::max(sourceNorm, 1.0);
        double rNorm = sourceNorm;           // |r|^2 as tracked by the iteration
        double rNormUpdate = sourceNorm;     // true |r|^2 at the last reliable update
        int reliableUpdates = 0;
        auto relative = [&](double norm) { return std::sqrt(norm / std::max(sourceNorm, 1.0)); };

        if (rNorm <= targetNorm) {
            if (update) {
                x.updateAll();
            }
            return {0, 0, relative(rNorm), true};
        }

        for (int iteration = 0; iteration < maxIter; iteration++) {
            pF.updateAll();
            adapterF.apply(sF, pF, false);
            const double pAp = real<double>(adapterF.dotProduct5D(pF, sF));
            if (!std::isfinite(pAp) || pAp <= 0.0) {
                throw std::runtime_error(stdLogger.fatal("MDWF mixed CG: non-positive <p, A p> = ", pAp));
            }
            const double alpha = rNorm / pAp;
            rF.template axpyThisB<BlockSize>(static_cast<float>(-alpha), sF);
            accum.template axpyThisB<BlockSize>(alpha, p);
            const double rNormNext = adapterF.norm2(rF);
            const double beta = rNormNext / rNorm;

            if (rNormNext < delta * delta * rNormUpdate || rNormNext <= targetNorm) {
                // Reliable update: fold the accumulated correction into x and recompute the residual in double.
                x += accum;
                accum = static_cast<double>(0.0) * accum;
                x.updateAll();
                adapterD.apply(ax, x, false);
                r = b;
                r.template axpyThisB<BlockSize>(-1.0, ax);
                const double trueNorm = adapterD.norm2(r);
                reliableUpdates++;
                if (trueNorm <= targetNorm) {
                    if (update) {
                        x.updateAll();
                    }
                    return {iteration + 1, reliableUpdates, relative(trueNorm), true};
                }
                // Keep the search direction conjugate to the new residual (p = r + beta (p - <p,r>/|r|^2 r)).
                const double pr = real<double>(adapterD.dotProduct5D(p, r));
                p.template axpyThisB<BlockSize>(-pr / trueNorm, r);
                p *= COMPLEX(double)(beta, 0.0);
                p += r;
                pF.convert_precision(p);
                rF.convert_precision(r);
                rNorm = trueNorm;
                rNormUpdate = trueNorm;
            } else {
                pF *= COMPLEX(float)(static_cast<float>(beta), 0.0f);
                pF += rF;
                p.convert_precision(pF);
                rNorm = rNormNext;
            }
        }

        // Not converged: return the solution so far with its true residual.
        x += accum;
        x.updateAll();
        adapterD.apply(ax, x, false);
        r = b;
        r.template axpyThisB<BlockSize>(-1.0, ax);
        return {maxIter, reliableUpdates, relative(adapterD.norm2(r)), false};
    }
};

template<size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusEvenOddMixedSolver {
public:
    using EvenOdd = MDWFMobiusCloverEvenOdd<double, HaloDepthGauge, HaloDepthSpin, Ls>;
    using EvenOddF = MDWFMobiusCloverEvenOdd<float, HaloDepthGauge, HaloDepthSpin, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using NormalOpF = MDWFMobiusSchurNormalOperator<EvenOddF>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepthGauge, HaloDepthSpin, Ls, NormalOp>;
    using AdapterF = MDWFCoupledSolverAdapter<float, HaloDepthGauge, HaloDepthSpin, Ls, NormalOpF>;
    using CG = MDWFCoupledMixedCG<Adapter, AdapterF>;
    using SpinorAll = typename EvenOdd::SpinorAll;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;

private:
    Gaugefield<double, true, HaloDepthGauge, R18> &_gauge;
    Gaugefield<float, true, HaloDepthGauge, R18> _gaugeF;
    EvenOdd _eo;
    EvenOddF _eoF;
    NormalOp _normal;
    NormalOpF _normalF;
    Adapter _adapter;
    AdapterF _adapterF;
    SpinorE _be, _xe, _te, _ue;
    SpinorO _bo, _xo, _to, _uo;
    double _delta;
    int _lastIterations;
    int _lastReliableUpdates;
    double _lastResidue;

    void refresh() {
        _eo.refresh();
        _gaugeF.convert_precision(_gauge);
        _gaugeF.updateAll();
        _eoF.refreshFrom(_eo);
    }

    void runCG(SpinorE &x, const SpinorE &b, int maxIter, double precision, const char *what) {
        CG cg;
        const MDWFCoupledMixedCGResult result = cg.invert(_adapter, _adapterF, x, b, maxIter, precision, _delta);
        _lastIterations += result.iterations;
        _lastReliableUpdates += result.reliableUpdates;
        _lastResidue = result.residue;
        rootLogger.info("MDWF mixed-precision CG (", what, "): ", result.iterations, " single-precision iterations, ",
                        result.reliableUpdates, " reliable updates, residue ", result.residue);
        if (!result.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd mixed-precision ", what,
                                                     " solve did not converge: iterations = ", result.iterations,
                                                     ", reliable updates = ", result.reliableUpdates,
                                                     ", residue = ", result.residue));
        }
    }

public:
    MDWFMobiusEvenOddMixedSolver(Gaugefield<double, true, HaloDepthGauge, R18> &gauge, double M5, double mf,
                                 double b5, double csw, const std::string &name, double delta = 0.1)
        : _gauge(gauge),
          _gaugeF(gauge.getComm(), name + "_gaugef"),
          _eo(gauge, M5, mf, b5, csw, name + "_eo"),
          _eoF(_gaugeF, static_cast<float>(M5), static_cast<float>(mf), static_cast<float>(b5),
               static_cast<float>(csw), name + "_feo"),
          _normal(_eo, gauge.getComm(), name + "_normal"),
          _normalF(_eoF, gauge.getComm(), name + "_fnormal"),
          _adapter(_normal),
          _adapterF(_normalF),
          _be(gauge.getComm(), name + "_be"), _xe(gauge.getComm(), name + "_xe"),
          _te(gauge.getComm(), name + "_te"), _ue(gauge.getComm(), name + "_ue"),
          _bo(gauge.getComm(), name + "_bo"), _xo(gauge.getComm(), name + "_xo"),
          _to(gauge.getComm(), name + "_to"), _uo(gauge.getComm(), name + "_uo"),
          _delta(delta), _lastIterations(0), _lastReliableUpdates(0), _lastResidue(0.0) {}

    void setDelta(double delta) {
        _delta = delta;
    }

    EvenOdd &evenOdd() {
        return _eo;
    }

    // M x = b.
    void solve(SpinorAll &x, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _lastReliableUpdates = 0;
        refresh();
        solveNoRefresh(x, b, maxIter, precision);
    }

    // M^dagger y = b.
    void solveDagger(SpinorAll &y, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _lastReliableUpdates = 0;
        refresh();
        solveDaggerNoRefresh(y, b, maxIter, precision);
    }

    // M^dagger M x = b, as M^-1 (M^dagger)^-1 b.
    void solveNormal(SpinorAll &x, SpinorAll &work, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _lastReliableUpdates = 0;
        refresh();
        solveDaggerNoRefresh(work, b, maxIter, precision);
        solveNoRefresh(x, work, maxIter, precision);
    }

    int lastIterations() const {
        return _lastIterations;
    }

    int lastReliableUpdates() const {
        return _lastReliableUpdates;
    }

    double lastResidue() const {
        return _lastResidue;
    }

private:
    // Same pre- and post-processing as MDWFMobiusEvenOddSolver, in double.
    void solveNoRefresh(SpinorAll &x, const SpinorAll &b, int maxIter, double precision) {
        EvenOdd::split(_be, _bo, b);
        _eo.MooInv(_to, _bo);
        _eo.Meo(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(-1.0, _te);
        _eo.schur(_te, _ue, true);
        runCG(_xe, _te, maxIter, precision, "M");
        _eo.Moe(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(-1.0, _to);
        _eo.MooInv(_xo, _uo);
        EvenOdd::merge(x, _xe, _xo);
    }

    void solveDaggerNoRefresh(SpinorAll &y, const SpinorAll &b, int maxIter, double precision) {
        EvenOdd::split(_be, _bo, b);
        _eo.MooInv(_to, _bo, true);
        _eo.MoeDagger(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(-1.0, _te);
        runCG(_te, _ue, maxIter, precision, "M^dagger");
        _eo.schur(_xe, _te, false);
        _eo.MeoDagger(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(-1.0, _to);
        _eo.MooInv(_xo, _uo, true);
        EvenOdd::merge(y, _xe, _xo);
    }
};
