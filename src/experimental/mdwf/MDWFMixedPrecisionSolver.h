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

#include "MDWFSpinor.h"

#include "MDWFMobiusEvenOdd.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct MDWFCoupledMixedCGResult {
    int iterations;          // single-precision operator applications
    int reliableUpdates;     // double-precision residual recomputations
    double residue;          // true relative residual sqrt(|b - A x|^2 / max(|b|^2, 1)) of the last update
    bool converged;
};

// Field names of the mixed CG work vectors: 4 double (r, p, accum, A x) and 4 single (r, p, A p, x). The defaults are
// private per call; callers that know which shared buffers are idle during the solve can pass SHARED_ names.
struct MDWFMixedCGWorkNames {
    std::string rD = "MDWF_mixed_cg_resd", pD = "MDWF_mixed_cg_srchd", accumD = "MDWF_mixed_cg_accum",
                axD = "MDWF_mixed_cg_opx", rF = "MDWF_mixed_cg_resf", pF = "MDWF_mixed_cg_srchf",
                sF = "MDWF_mixed_cg_opsf", xF = "MDWF_mixed_cg_itf";
};

// Single-precision accumulation of x between reliable updates (Clark et al.) and fused vector updates in the mixed
// CG: on unless MDWF_FAST_MIXED=0 (then the original loop with a double accumulator).
inline bool &mdwfFastMixedCG() {
    static bool on = [] {
        const char *env = std::getenv("MDWF_FAST_MIXED");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

// x += a p, r -= a s in one pass (single-precision CG iteration).
template<class Spinor>
struct MDWFMixedCGUpdate {
    using T = MDWFSpinorTraits<Spinor>;
    using floatT = typename T::floatT;
    VectArrayAcc<floatT, T::elems> _x, _p, _r, _s;
    floatT _a;

    MDWFMixedCGUpdate(Spinor &x, const Spinor &p, Spinor &r, const Spinor &s, floatT a)
        : _x(x.getAccessor()), _p(p.getAccessor()), _r(r.getAccessor()), _s(s.getAccessor()), _a(a) {}

    __host__ __device__ void operator()(gSiteStack site) {
        Vect<floatT, T::elems> x = _x.getElement(site);
        x += _a * _p.getElement(site);
        _x.setElement(site, x);
        Vect<floatT, T::elems> r = _r.getElement(site);
        r -= _a * _s.getElement(site);
        _r.setElement(site, r);
    }

    void launch() {
        typedef GIndexer<T::layout, T::halo> GInd;
        const size_t sites = (T::layout == All) ? GInd::getLatData().vol4 : GInd::getLatData().sizeh;
        CalcGSiteStack<T::layout, T::halo> calc;
        iterateFunctorNoReturn<true, (DEFAULT_NBLOCKS / T::stacks) * T::stacks>(*this, calc, sites, T::stacks);
    }
};

// p = r + b p.
template<class Spinor>
struct MDWFMixedCGDirection {
    using T = MDWFSpinorTraits<Spinor>;
    using floatT = typename T::floatT;
    VectArrayAcc<floatT, T::elems> _r, _p;
    floatT _b;

    MDWFMixedCGDirection(const Spinor &r, const Spinor &p, floatT b) : _r(r.getAccessor()), _p(p.getAccessor()), _b(b) {}

    __host__ __device__ Vect<floatT, T::elems> operator()(gSiteStack site) {
        Vect<floatT, T::elems> p = _b * _p.getElement(site);
        p += _r.getElement(site);
        return p;
    }
};

template<class AdapterD, class AdapterF, size_t BlockSize = 64>
class MDWFCoupledMixedCG {
public:
    using SpinorD = typename AdapterD::Spinor;
    using SpinorF = typename AdapterF::Spinor;

    MDWFCoupledMixedCG() = default;
    explicit MDWFCoupledMixedCG(MDWFMixedCGWorkNames names) : _names(std::move(names)) {}

    // useGuess: x holds the starting vector on entry (otherwise x = 0); the stopping rule is unchanged.
    MDWFCoupledMixedCGResult invert(AdapterD &adapterD, AdapterF &adapterF, SpinorD &x, const SpinorD &b,
                                    int maxIter, double precision, double delta, bool update = true,
                                    bool useGuess = false) const {
        if (!(delta > 0.0 && delta < 1.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF mixed CG needs 0 < delta < 1, got ", delta));
        }
        CommunicationBase &comm = b.getComm();
        SpinorD r(comm, _names.rD);
        SpinorD p(comm, _names.pD);
        SpinorD accum(comm, _names.accumD);
        SpinorD ax(comm, _names.axD);
        SpinorF rF(comm, _names.rF);
        SpinorF pF(comm, _names.pF);
        SpinorF sF(comm, _names.sF);

        if (mdwfFastMixedCG()) {
            return invertFast(adapterD, adapterF, x, b, maxIter, precision, delta, update, useGuess);
        }
        const double sourceNorm = adapterD.norm2(const_cast<SpinorD &>(b));
        accum = static_cast<double>(0.0) * b;
        r = b;
        if (useGuess) {
            mdwfPrepareInput(adapterD, x);
            adapterD.apply(ax, x, false);
            r.template axpyThisB<BlockSize>(-1.0, ax);
        } else {
            x = static_cast<double>(0.0) * b;
        }
        p = r;
        rF.convert_precision(r);
        pF.convert_precision(p);

        const double targetNorm = precision * precision * std::max(sourceNorm, 1.0);
        double rNorm = adapterD.norm2(r);    // |r|^2 as tracked by the iteration
        double rNormUpdate = rNorm;          // true |r|^2 at the last reliable update
        int reliableUpdates = 0;
        auto relative = [&](double norm) { return std::sqrt(norm / std::max(sourceNorm, 1.0)); };

        if (rNorm <= targetNorm) {
            if (update) {
                mdwfUpdateHalo(x);
            }
            return {0, 0, relative(rNorm), true};
        }

        for (int iteration = 0; iteration < maxIter; iteration++) {
            mdwfPrepareInput(adapterF, pF);
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
                mdwfPrepareInput(adapterD, x);
                adapterD.apply(ax, x, false);
                r = b;
                r.template axpyThisB<BlockSize>(-1.0, ax);
                const double trueNorm = adapterD.norm2(r);
                reliableUpdates++;
                if (trueNorm <= targetNorm) {
                    if (update) {
                        mdwfUpdateHalo(x);
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
        mdwfUpdateHalo(x);
        adapterD.apply(ax, x, false);
        r = b;
        r.template axpyThisB<BlockSize>(-1.0, ax);
        return {maxIter, reliableUpdates, relative(adapterD.norm2(r)), false};
    }

    /*
     * The same reliable-update CG with the iterate accumulated in single precision between reliable updates
     * (xF += alpha pF, folded into the double x at each update) and no double search vector in the loop: per
     * iteration one operator application, <p, A p>, one fused update (xF, rF), |rF|^2, p = r + beta p, all single.
     * At a reliable update, as above: x += xF, r = b - A x (double), p = r + beta (p - <p,r>/|r|^2 r) in double.
     */
    MDWFCoupledMixedCGResult invertFast(AdapterD &adapterD, AdapterF &adapterF, SpinorD &x, const SpinorD &b,
                                        int maxIter, double precision, double delta, bool update,
                                        bool useGuess) const {
        CommunicationBase &comm = b.getComm();
        SpinorD r(comm, _names.rD);
        SpinorD p(comm, _names.pD);
        SpinorD xs(comm, _names.accumD);   // double scratch (x correction, p at updates)
        SpinorD ax(comm, _names.axD);
        SpinorF rF(comm, _names.rF);
        SpinorF pF(comm, _names.pF);
        SpinorF sF(comm, _names.sF);
        SpinorF xF(comm, _names.xF);

        const double sourceNorm = adapterD.norm2(const_cast<SpinorD &>(b));
        r = b;
        if (useGuess) {
            mdwfPrepareInput(adapterD, x);
            adapterD.apply(ax, x, false);
            r.template axpyThisB<BlockSize>(-1.0, ax);
        } else {
            x = static_cast<double>(0.0) * b;
        }
        rF.convert_precision(r);
        pF = rF;
        xF = static_cast<float>(0.0) * rF;

        const double targetNorm = precision * precision * std::max(sourceNorm, 1.0);
        double rNorm = adapterD.norm2(r);
        double rNormUpdate = rNorm;
        int reliableUpdates = 0;
        auto relative = [&](double norm) { return std::sqrt(norm / std::max(sourceNorm, 1.0)); };
        if (rNorm <= targetNorm) {
            if (update) {
                mdwfUpdateHalo(x);
            }
            return {0, 0, relative(rNorm), true};
        }

        for (int iteration = 0; iteration < maxIter; iteration++) {
            mdwfPrepareInput(adapterF, pF);
            adapterF.apply(sF, pF, false);
            const double pAp = real<double>(adapterF.dotProduct5D(pF, sF));
            if (!std::isfinite(pAp) || pAp <= 0.0) {
                rootLogger.error("MDWF fast mixed CG diagnostics: iteration ", iteration, ", reliable updates ",
                                 reliableUpdates, ", |b|^2 ", sourceNorm, ", |r|^2 (tracked) ", rNorm, ", |rF|^2 ",
                                 adapterF.norm2(rF), ", |pF|^2 ", adapterF.norm2(pF), ", |sF|^2 ", adapterF.norm2(sF),
                                 ", |xF|^2 ", adapterF.norm2(xF), ", |x|^2 ", adapterD.norm2(x));
                throw std::runtime_error(stdLogger.fatal("MDWF mixed CG: non-positive <p, A p> = ", pAp));
            }
            const double alpha = rNorm / pAp;
            MDWFMixedCGUpdate<SpinorF>(xF, pF, rF, sF, static_cast<float>(alpha)).launch();
            const double rNormNext = adapterF.norm2(rF);
            const double beta = rNormNext / rNorm;

            if (rNormNext < delta * delta * rNormUpdate || rNormNext <= targetNorm) {
                xs.convert_precision(xF);
                x += xs;
                xF = static_cast<float>(0.0) * rF;
                mdwfPrepareInput(adapterD, x);
                adapterD.apply(ax, x, false);
                r = b;
                r.template axpyThisB<BlockSize>(-1.0, ax);
                const double trueNorm = adapterD.norm2(r);
                reliableUpdates++;
                if (trueNorm <= targetNorm) {
                    if (update) {
                        mdwfUpdateHalo(x);
                    }
                    return {iteration + 1, reliableUpdates, relative(trueNorm), true};
                }
                p.convert_precision(pF);
                const double pr = real<double>(adapterD.dotProduct5D(p, r));
                p.template axpyThisB<BlockSize>(-pr / trueNorm, r);
                p *= COMPLEX(double)(beta, 0.0);
                p += r;
                pF.convert_precision(p);
                rF.convert_precision(r);
                rNorm = trueNorm;
                rNormUpdate = trueNorm;
            } else {
                pF.template iterateOverBulk<BlockSize>(MDWFMixedCGDirection<SpinorF>(rF, pF, static_cast<float>(beta)));
                rNorm = rNormNext;
            }
        }

        xs.convert_precision(xF);
        x += xs;
        mdwfUpdateHalo(x);
        adapterD.apply(ax, x, false);
        r = b;
        r.template axpyThisB<BlockSize>(-1.0, ax);
        return {maxIter, reliableUpdates, relative(adapterD.norm2(r)), false};
    }

private:
    MDWFMixedCGWorkNames _names;
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
        mdwfUpdateHalo(_gaugeF);
        _eoF.refreshFrom(_eo);
    }

    void runCG(SpinorE &x, const SpinorE &b, int maxIter, double precision, const char *what) {
        // SHARED work buffers, as MDWFEvenOddMixedSchurSolver: private default names alias in MemoryManagement's
        // getSmartName when they share a prefix, which broke the fast mixed CG (EOFA NaN, E6).
        MDWFMixedCGWorkNames names;
        names.rD = EvenOdd::sharedEvenName("fsol_cg_rd");
        names.pD = EvenOdd::sharedEvenName("fsol_cg_pd");
        names.accumD = EvenOdd::sharedEvenName("fsol_cg_ad");
        names.axD = EvenOdd::sharedEvenName("fsol_cg_xd");
        names.rF = EvenOddF::sharedEvenName("fsol_cg_rf");
        names.pF = EvenOddF::sharedEvenName("fsol_cg_pf");
        names.sF = EvenOddF::sharedEvenName("fsol_cg_sf");
        names.xF = EvenOddF::sharedEvenName("fsol_cg_xf");
        CG cg(names);
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

/*
 * Mixed-precision solves of the Schur normal equation Mhat^+ Mhat x = b for a family of double even/odd operators
 * (the masses of a Hasenbusch ladder): one single-precision gauge copy and ONE single-precision operator, re-pointed
 * at the double operator of each solve (refreshFrom takes over the clover blocks and the mass parameters), and the
 * reliable-update CG above.
 *
 * Optional chronological initial guess (Brower, Ivanenko, Levi, Orginos, Nucl. Phys. B 484 (1997) 353): a History
 * keeps the last k solutions of one solve family (e.g. the MD force solves of one Hasenbusch factor) in single
 * precision; the guess is the Galerkin minimiser of the A-norm error in their span, x0 = V c with
 * (V^+ A V) c = V^+ b, formed with the single-precision operator. The guess only changes the starting point: the
 * stopping rule on the double residual is the same, so the solution is as accurate as a solve from zero. (As with
 * any history-dependent guess, MD reversibility then holds to the solver tolerance.)
 *
 * Work vectors use SHARED_ buffers that are idle during light-quark solves: the strange-RHMC pole slots pool6 ...
 * pool10 (double) and single-precision scratch.
 */
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddMixedSchurSolver {
public:
    using EvenOdd = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using EvenOddF = MDWFMobiusCloverEvenOdd<float, HaloDepth, HaloDepth, Ls>;
    using NormalOpF = MDWFMobiusSchurNormalOperator<EvenOddF>;
    using AdapterF = MDWFCoupledSolverAdapter<float, HaloDepth, HaloDepth, Ls, NormalOpF>;
    using SpinorD = typename EvenOdd::SpinorE;
    using SpinorF = typename EvenOddF::SpinorE;
    using GaugeF = Gaugefield<float, true, HaloDepth, R18>;

    class History {
        std::vector<std::unique_ptr<SpinorF>> _v;
        size_t _next = 0;
        size_t _filled = 0;
        friend class MDWFEvenOddMixedSchurSolver;

    public:
        History(CommunicationBase &comm, size_t size, const std::string &name) {
            for (size_t i = 0; i < size; i++) {
                _v.emplace_back(new SpinorF(comm, name + "_chrono" + std::to_string(i)));
            }
        }

        void push(SpinorD &x) {
            if (_v.empty()) {
                return;
            }
            _v[_next]->convert_precision(x);
            _next = (_next + 1) % _v.size();
            _filled = std::min(_filled + 1, _v.size());
        }

        size_t filled() const {
            return _filled;
        }
    };

private:
    GaugeF _gaugeF;
    EvenOddF _eoF;
    NormalOpF _normalF;
    AdapterF _adapterF;
    double _delta;
    MDWFCoupledMixedCGResult _last;

    static std::string dname(const std::string &role) {
        return EvenOdd::sharedEvenName(role);
    }

    static std::string fname(const std::string &role) {
        return EvenOddF::sharedEvenName(role);
    }

    // Galerkin guess from the history; false if the history is empty or its Gram matrix is singular.
    bool guess(SpinorD &x, SpinorD &b, History &h) {
        const size_t k = h.filled();
        if (k == 0) {
            return false;
        }
        CommunicationBase &comm = b.getComm();
        SpinorF sF(comm, fname("chrono_av"));
        SpinorF bF(comm, fname("chrono_b"));
        bF.convert_precision(b);
        std::vector<std::complex<double>> G(k * k), c(k);
        for (size_t j = 0; j < k; j++) {
            mdwfUpdateHalo(*h._v[j]);
            _adapterF.apply(sF, *h._v[j], false);
            for (size_t i = 0; i < k; i++) {
                const COMPLEX(double) gij = _adapterF.dotProduct5D(*h._v[i], sF);
                G[i * k + j] = {gij.cREAL, gij.cIMAG};
            }
            const COMPLEX(double) hj = _adapterF.dotProduct5D(*h._v[j], bF);
            c[j] = {hj.cREAL, hj.cIMAG};
        }
        // Gaussian elimination with partial pivoting (k is a handful).
        for (size_t col = 0; col < k; col++) {
            size_t piv = col;
            for (size_t row = col + 1; row < k; row++) {
                if (std::abs(G[row * k + col]) > std::abs(G[piv * k + col])) {
                    piv = row;
                }
            }
            if (!(std::abs(G[piv * k + col]) > 1e-12 * std::abs(G[0]))) {
                return false;
            }
            if (piv != col) {
                for (size_t j = 0; j < k; j++) {
                    std::swap(G[col * k + j], G[piv * k + j]);
                }
                std::swap(c[col], c[piv]);
            }
            for (size_t row = col + 1; row < k; row++) {
                const std::complex<double> f = G[row * k + col] / G[col * k + col];
                for (size_t j = col; j < k; j++) {
                    G[row * k + j] -= f * G[col * k + j];
                }
                c[row] -= f * c[col];
            }
        }
        for (size_t col = k; col-- > 0;) {
            for (size_t j = col + 1; j < k; j++) {
                c[col] -= G[col * k + j] * c[j];
            }
            c[col] /= G[col * k + col];
        }
        SpinorD tmp(comm, dname("pool10"));
        x = static_cast<double>(0.0) * b;
        for (size_t i = 0; i < k; i++) {
            tmp.convert_precision(*h._v[i]);
            tmp *= COMPLEX(double)(c[i].real(), c[i].imag());
            x += tmp;
        }
        return true;
    }

public:
    MDWFEvenOddMixedSchurSolver(Gaugefield<double, true, HaloDepth, R18> &gauge, double M5, double b5, double csw,
                                double delta, const std::string &name)
        : _gaugeF(gauge.getComm(), name + "_gaugef"),
          _eoF(_gaugeF, static_cast<float>(M5), 0.1f, static_cast<float>(b5), static_cast<float>(csw), name + "_eof"),
          _normalF(_eoF, gauge.getComm(), name + "_nrmf"),
          _adapterF(_normalF),
          _delta(delta),
          _last{0, 0, 0.0, true} {}

    // Solves (Mhat^+ Mhat) x = b, or (Mhat Mhat^+) x = b with mmdag, with Mhat = eo (refreshed), adapterD its double
    // Schur-normal adapter (in the same operator order). The single-precision operator is shared: its order is set here.
    template<class AdapterD>
    MDWFCoupledMixedCGResult solve(EvenOdd &eo, AdapterD &adapterD, SpinorD &x, SpinorD &b, int maxIter,
                                   double precision, History *history, bool mmdag = false) {
        _gaugeF.convert_precision(eo.gauge());
        mdwfUpdateHalo(_gaugeF);
        _eoF.refreshFrom(eo);
        _normalF.setMMdag(mmdag);
        const bool useGuess = history != nullptr && guess(x, b, *history);
        MDWFMixedCGWorkNames names;
        names.rD = dname("pool6");
        names.pD = dname("pool7");
        names.accumD = dname("pool8");
        names.axD = dname("pool9");
        names.rF = fname("mixed_cg_r");
        names.pF = fname("mixed_cg_p");
        names.sF = fname("mixed_cg_s");
        names.xF = fname("mixed_cg_x");
        MDWFCoupledMixedCG<AdapterD, AdapterF> cg(names);
        _last = cg.invert(adapterD, _adapterF, x, b, maxIter, precision, _delta, true, useGuess);
        if (history != nullptr && _last.converged) {
            history->push(x);
        }
        return _last;
    }

    const MDWFCoupledMixedCGResult &last() const {
        return _last;
    }
};
