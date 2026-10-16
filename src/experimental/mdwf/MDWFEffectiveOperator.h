/*
 * The 4D effective Dirac operator of Mobius (clover) domain-wall fermions,
 *
 *   D(m) = [ P^-1 M(1)^-1 M(m) P ]_{s=0},
 *
 * with M(m) the 5D operator of MDWFMobiusCloverLinearOperator (boundary mass m, c5 = b5 - 1) and P the chiral
 * permutation that puts a 4D field onto the walls,
 *
 *   lift      (P eta)_{s=0} = P_- eta,  (P eta)_{s=Ls-1} = P_+ eta,  zero elsewhere,
 *   restrict  [P^-1 z]_{s=0} = P_- z(0) + P_+ z(Ls-1)   (the same projection as the m_res quark field),
 *
 * P_+ the upper six components (the part coupled to s-1, MDWFFifthDim.h), so the light mode of the free theory sits
 * as P_- at s = 0 and P_+ at s = Ls-1. The left factor D_- = 1 - c5 D_W of M (M = D_- M_std, mass independent)
 * cancels in M(1)^-1 M(m). One application of D(m) costs one Pauli-Villars solve M(1) z = M(m) P eta.
 *
 * Exact relations used below (all hold at finite Ls, since M(m) is affine in m and D(1) = 1):
 *   D(m) = m + (1 - m) D(0),   D(0) = (1 + gamma5 eps) / 2,   eps = gamma5 (2 D(0) - 1),
 * with eps = eps_Ls(H_M) the Mobius approximation of sign(H_M), Hermitian. Hence
 *   gamma5 D + D gamma5 - 2 D gamma5 D = gamma5 (1 - eps^2) / 2   at m = 0:
 * the Ginsparg-Wilson violation is the sign-function defect 1 - eps^2. Per normalized vector v,
 *   gwDefect(v) = <v|1 - eps^2|v> = 1 - |eps v|^2.
 * D(m)^dagger = gamma5 D(m) gamma5; the normal operator D^dagger D = gamma5 D gamma5 D costs two solves.
 *
 * Single rank.
 */

#pragma once

#include "MDWFAdjointOperator.h"
#include "MDWFFifthDim.h"
#include "MDWFMobiusEvenOdd.h"
#include "MDWFMobiusMapping.h"

#include <string>

// 5D field from a 4D one: P_- eta on s = 0, P_+ eta on s = Ls-1.
template<size_t HaloDepth, size_t Ls>
struct MDWFLiftToWalls {
    Vect12ArrayAcc<double> _in;

    explicit MDWFLiftToWalls(const MDWFSpinor<double, true, All, HaloDepth, 1> &in) : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        if (site.stack == 0) {
            return mdwfProjectMinus(_in.getElement(GInd::getSiteStack(site, 0)));
        }
        if (site.stack == Ls - 1) {
            return mdwfProjectPlus(_in.getElement(GInd::getSiteStack(site, 0)));
        }
        return Vect12<double>(0.0);
    }
};

// 4D field from a 5D one: P_- z(0) + P_+ z(Ls-1).
template<size_t HaloDepth, size_t Ls>
struct MDWFRestrictFromWalls {
    Vect12ArrayAcc<double> _in;

    explicit MDWFRestrictFromWalls(const MDWFSpinor<double, true, All, HaloDepth, Ls> &in) : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        return mdwfProjectMinus(_in.getElement(GInd::getSiteStack(site, 0)))
               + mdwfProjectPlus(_in.getElement(GInd::getSiteStack(site, Ls - 1)));
    }
};

// SolverT: MDWFMobiusEvenOddSolver<double, ...> or MDWFMobiusEvenOddMixedSolver<...> (solve, lastIterations).
template<size_t HaloDepth, size_t Ls, class SolverT = MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>>
class MDWFEffectiveDiracOperator {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using Spinor5 = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls> _op;   // M(m)
    SolverT _pv;                                                             // M(1)^-1 (Pauli-Villars)
    Spinor5 _lift, _rhs, _sol;
    Spinor _t1, _t2;
    double _m;
    int _maxIter;
    double _precision;
    long _solves = 0;
    long _iterations = 0;

public:
    MDWFEffectiveDiracOperator(Gauge &fermionGauge, double M5, double m, double b5, double csw, int maxIter,
                               double precision, const std::string &name)
        : _op(fermionGauge, M5, m, b5, csw, name + "_m"),
          _pv(fermionGauge, M5, 1.0, b5, csw, name + "_pv"),
          _lift(fermionGauge.getComm(), name + "_lift"),
          _rhs(fermionGauge.getComm(), name + "_rhs"),
          _sol(fermionGauge.getComm(), name + "_sol"),
          _t1(fermionGauge.getComm(), name + "_t1"),
          _t2(fermionGauge.getComm(), name + "_t2"),
          _m(m), _maxIter(maxIter), _precision(precision) {
        if (!(m < 1.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF effective operator needs m < 1, got ", m));
        }
    }

    SolverT &solver() {
        return _pv;
    }

    double mass() const {
        return _m;
    }

    long solves() const {
        return _solves;
    }

    long iterations() const {
        return _iterations;
    }

    // out = D(m) in.
    void applyD(Spinor &out, const Spinor &in, bool update = false) {
        _lift.template iterateOverBulk<BLOCKSIZE>(MDWFLiftToWalls<HaloDepth, Ls>(in));
        _lift.updateAll();
        _op.apply(_rhs, _lift, true);
        _pv.solve(_sol, _rhs, _maxIter, _precision);
        _solves++;
        _iterations += _pv.lastIterations();
        out.template iterateOverBulk<BLOCKSIZE>(MDWFRestrictFromWalls<HaloDepth, Ls>(_sol));
        if (update) {
            out.updateAll();
        }
    }

    // out = D(m)^dagger in = gamma5 D(m) gamma5 in.
    void applyDagger(Spinor &out, const Spinor &in, bool update = false) {
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(_t2, in, true);
        applyD(_t1, _t2);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _t1, update);
    }

    // out = D(m)^dagger D(m) in = gamma5 D gamma5 D in (Hermitian, positive).
    void applyNormal(Spinor &out, const Spinor &in, bool update = false) {
        applyD(_t1, in);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(_t2, _t1, true);
        applyD(_t1, _t2);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _t1, update);
    }

    // out = eps in = gamma5 (2 D(0) - 1) in, with D(0) = (D(m) - m) / (1 - m):
    // eps = gamma5 [ (2 / (1 - m)) D(m) - ((1 + m) / (1 - m)) ] in.
    void applyEpsilon(Spinor &out, const Spinor &in, bool update = false) {
        applyD(_t1, in);
        _t1 *= COMPLEX(double)(2.0 / (1.0 - _m), 0.0);
        _t1.template axpyThisB<64>(-(1.0 + _m) / (1.0 - _m), in);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _t1, update);
    }
};

// The kernel: H_W^2 = (gamma5 D_W(-M5))^2 = D_W^dagger D_W on 4D fields, D_W with clover (mass = 4 - M5), no solves.
template<size_t HaloDepth>
class MDWFHermitianWilsonSquaredOperator {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Clover = Spinorfield<double, true, All, HaloDepth, 18, 1>;

private:
    Gauge &_gauge;
    double _mass, _csw;
    Spinor _t1, _t2, _tmp;
    Clover _fu, _fl, _fiu, _fil;

public:
    MDWFHermitianWilsonSquaredOperator(Gauge &fermionGauge, double M5, double csw, const std::string &name)
        : _gauge(fermionGauge), _mass(mdwfShamirKernelMass<double>(M5)), _csw(csw),
          _t1(fermionGauge.getComm(), name + "_t1"), _t2(fermionGauge.getComm(), name + "_t2"),
          _tmp(fermionGauge.getComm(), name + "_tmp"),
          _fu(fermionGauge.getComm(), name + "_fu"), _fl(fermionGauge.getComm(), name + "_fl"),
          _fiu(fermionGauge.getComm(), name + "_fiu"), _fil(fermionGauge.getComm(), name + "_fil") {}

    // in must have its halo up to date (the eigensolvers update it before every application).
    void apply(Spinor &out, const Spinor &in, bool update = false) {
        applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(_t1, _gauge, _tmp, _fu, _fl, _fiu, _fil, in, _mass,
                                                                    _csw, false);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(_t2, _t1, true);
        applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(_t1, _gauge, _tmp, _fu, _fl, _fiu, _fil, _t2,
                                                                    _mass, _csw, false);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _t1, update);
    }
};

// D(m)^dagger D(m) as an operator object for the eigensolvers (apply(out, in)).
template<class Effective>
class MDWFEffectiveNormalOperator {
    Effective &_d;

public:
    using Spinor = typename Effective::Spinor;

    explicit MDWFEffectiveNormalOperator(Effective &d) : _d(d) {}

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        _d.applyNormal(out, in, update);
    }
};
