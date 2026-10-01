/*
 * Residual mass m_res of Mobius (clover) domain-wall fermions, with Grid's conventions so that numbers compare directly
 * with Grid/GPT measurements (CayleyFermion5D: ImportPhysicalFermionSource,
 * ExportPhysicalFermionSolution, ContractJ5q):
 *
 *   import  (4D source eta -> 5D source):  b(s=0) = P_+ eta,  b(s=Ls-1) = P_- eta,
 *                                          then b <- D_- b,  D_- = 1 - c5 D_W (each slice)
 *   export  (5D solution psi -> 4D quark): q = P_- psi(0) + P_+ psi(Ls-1)
 *   J5q     (midpoint):                    p = P_+ psi(Ls/2-1) + P_- psi(Ls/2)
 *
 * with P_+ the upper six components (coupled to s-1 by the operator), as in
 * MDWFFifthDim.h. For a point source at x0 and its 12 spin-colour components
 * a, with psi_a = M^-1 import(e_a delta_x0),
 *
 *   C_PP(n)  = sum_a sum_{x, x_mu = n} |q_a(x)|^2      (pseudoscalar correlator, gamma5 hermiticity)
 *   C_J5q(n) = sum_a sum_{x, x_mu = n} |p_a(x)|^2      (midpoint pseudoscalar density with P(0))
 *   m_res(n) = C_J5q(n) / C_PP(n),  averaged over a plateau.
 *
 * Each solve is contracted along all four directions mu (the solves do not depend on mu): on
 * finite-temperature lattices m_res comes from the spatial (screening) correlators, which are long,
 * while the temporal ones on N_t = 8 reach only distance 4.
 *
 * D_W includes the clover term for c_sw != 0 (the kernel of
 * MDWFMobiusCloverLinearOperator). Antiperiodic temporal boundary conditions
 * for the fermions are applied by solving on a copy of the gauge field whose
 * temporal links on the last time slice carry a factor -1.
 *
 * Single rank.
 */

#pragma once

#include "MDWFMobiusEvenOdd.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>

// Copy of the gauge field with U_t(x) -> -U_t(x) on the last time slice (antiperiodic fermion BCs in time).
template<size_t HaloDepth>
struct MDWFAntiperiodicTimeLinks {
    SU3Accessor<double, R18> _gauge;
    int _ltLast;

    MDWFAntiperiodicTimeLinks(Gaugefield<double, true, HaloDepth, R18> &gauge, int ltLast)
        : _gauge(gauge.getAccessor()), _ltLast(ltLast) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        const SU3<double> link = _gauge.getLink(siteMu);
        if (siteMu.mu == 3 && static_cast<int>(siteMu.coord.t) == _ltLast) {
            return static_cast<double>(-1.0) * link;
        }
        return link;
    }
};

template<size_t HaloDepth>
void mdwfFermionGaugeField(Gaugefield<double, true, HaloDepth, R18> &fermionGauge,
                           Gaugefield<double, true, HaloDepth, R18> &gauge, bool antiperiodicTime) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF fermion boundary phases are single-rank only"));
    }
    if (antiperiodicTime) {
        fermionGauge.iterateOverBulkAllMu(MDWFAntiperiodicTimeLinks<HaloDepth>(gauge, static_cast<int>(lat.lt) - 1));
    } else {
        fermionGauge = gauge;
    }
    fermionGauge.updateAll();
}

// Unprojected 5D point source: e_a at x0 on slices 0 (P_+ part) and Ls-1 (P_- part), zero elsewhere.
template<size_t HaloDepth, size_t Ls>
struct MDWFPhysicalPointSource {
    int _x, _y, _z, _t, _component;

    MDWFPhysicalPointSource(int x, int y, int z, int t, int component)
        : _x(x), _y(y), _z(z), _t(t), _component(component) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        Vect12<double> out(0.0);
        const bool atSource = static_cast<int>(site.coord.x) == _x && static_cast<int>(site.coord.y) == _y
                              && static_cast<int>(site.coord.z) == _z && static_cast<int>(site.coord.t) == _t;
        if (!atSource) {
            return out;
        }
        const bool upper = _component < 6;
        if ((site.stack == 0 && upper) || (site.stack == Ls - 1 && !upper)) {
            out.data[_component] = COMPLEX(double)(1.0, 0.0);
        }
        return out;
    }
};

// Write index grouping the sites by their coordinate along direction dir (0 = x, ..., 3 = t): slice-major,
// so that reduceStackedLocal(values, L_dir, vol4 / L_dir) returns the slice sums.
template<size_t HaloDepth>
struct MDWFWriteAtSlices {
    int _dir;

    explicit MDWFWriteAtSlices(int dir) : _dir(dir) {}

    __host__ __device__ size_t operator()(const gSite &site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const sitexyzt c = site.coord;
        const size_t lx = GInd::getLatData().lx, ly = GInd::getLatData().ly, lz = GInd::getLatData().lz;
        const size_t lt = GInd::getLatData().lt, vol4 = GInd::getLatData().vol4;
        switch (_dir) {
            case 0:
                return c.x * (vol4 / lx) + c.y + ly * (c.z + lz * c.t);
            case 1:
                return c.y * (vol4 / ly) + c.x + lx * (c.z + lz * c.t);
            case 2:
                return c.z * (vol4 / lz) + c.x + lx * (c.y + ly * c.t);
            default:
                return c.t * (vol4 / lt) + c.x + lx * (c.y + ly * c.z);
        }
    }
};

// Per-site |P_- psi(0) + P_+ psi(Ls-1)|^2 (pseudoscalar) or |P_+ psi(Ls/2-1) + P_- psi(Ls/2)|^2 (J5q).
template<size_t HaloDepth, size_t Ls>
struct MDWFResidualMassDensity {
    Vect12ArrayAcc<double> _psi;
    bool _midpoint;

    MDWFResidualMassDensity(const MDWFSpinor<double, true, All, HaloDepth, Ls> &psi, bool midpoint)
        : _psi(psi.getAccessor()), _midpoint(midpoint) {}

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const size_t sPlus = _midpoint ? Ls / 2 - 1 : Ls - 1;
        const size_t sMinus = _midpoint ? Ls / 2 : 0;
        const Vect12<double> v = mdwfProjectPlus(_psi.getElement(GInd::getSiteStack(site, sPlus)))
                                 + mdwfProjectMinus(_psi.getElement(GInd::getSiteStack(site, sMinus)));
        return real(v * v);
    }
};

// Contribution of one source spin-colour component a (one solve), along each direction mu = 0 (x), ..., 3 (t).
struct MDWFResidualMassComponent {
    std::array<std::vector<double>, 4> pp;     // pp[mu][n] = sum_{x, x_mu = n} |q_a(x)|^2, n = global coordinate
    std::array<std::vector<double>, 4> j5q;    // j5q[mu][n] = sum_{x, x_mu = n} |p_a(x)|^2
    int iterations;
    double residue;
    double seconds;
};

struct MDWFResidualMassCorrelators {
    std::array<std::vector<double>, 4> pp;     // C_PP along mu, global coordinate (not shifted by the source)
    std::array<std::vector<double>, 4> j5q;    // C_J5q along mu
    int iterations;                            // total even/odd CG iterations of the 12 solves
    double maxResidue;
};

inline std::array<size_t, 4> mdwfLatticeExtents(const LatticeData &lat) {
    return {lat.lx, lat.ly, lat.lz, lat.lt};
}

inline void mdwfResizeDirectionalCorrelators(std::array<std::vector<double>, 4> &c, const LatticeData &lat) {
    const std::array<size_t, 4> extents = mdwfLatticeExtents(lat);
    for (int mu = 0; mu < 4; mu++) {
        c[mu].assign(extents[mu], 0.0);
    }
}

/*
 * C_PP and C_J5q for a point source at (x, y, z, t) with the Mobius clover operator on fermionGauge
 * (already carrying the fermion boundary phases), via 12 even/odd preconditioned solves.
 */
template<size_t HaloDepth, size_t Ls>
class MDWFResidualMassMeasurement {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using CloverField = Spinorfield<double, true, All, HaloDepth, 18, 1>;
    using Solver = MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>;

private:
    Gauge &_gauge;
    MDWFMobiusOperatorParameters<double> _params;
    double _csw;
    Solver _solver;
    Spinor _source, _imported, _solution, _wilson, _wilsonTmp;
    CloverField _fUpper, _fLower, _fInvUpper, _fInvLower;

public:
    MDWFResidualMassMeasurement(Gauge &fermionGauge, double M5, double mf, double b5, double csw,
                                const std::string &name)
        : _gauge(fermionGauge),
          _params(M5, mf, b5),
          _csw(csw),
          _solver(fermionGauge, M5, mf, b5, csw, name + "_solver"),
          _source(fermionGauge.getComm(), name + "_src5"),
          _imported(fermionGauge.getComm(), name + "_imp5"),
          _solution(fermionGauge.getComm(), name + "_sol5"),
          _wilson(fermionGauge.getComm(), name + "_dw5"),
          _wilsonTmp(fermionGauge.getComm(), name + "_dwtmp5"),
          _fUpper(fermionGauge.getComm(), name + "_fup"),
          _fLower(fermionGauge.getComm(), name + "_flo"),
          _fInvUpper(fermionGauge.getComm(), name + "_fiup"),
          _fInvLower(fermionGauge.getComm(), name + "_filo") {}

    // imported = (1 - c5 D_W) source, slice by slice (Grid's Dminus).
    void importSource(Spinor &imported, const Spinor &source) {
        applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, Ls>(_wilson, _gauge, _wilsonTmp, _fUpper, _fLower,
                                                                    _fInvUpper, _fInvLower, source, _params.mass,
                                                                    _csw, false);
        imported = source;
        imported.template axpyThisB<64>(-_params.c5, _wilson);
        imported.updateAll();
    }

    // One solve: source component a (0..11) of the point source at (x, y, z, t), contracted along all directions.
    MDWFResidualMassComponent measureComponent(int a, int x, int y, int z, int t, int maxIter, double precision) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        MDWFResidualMassComponent result{{}, {}, 0, 0.0, 0.0};
        _source.template iterateOverBulk<BLOCKSIZE>(MDWFPhysicalPointSource<HaloDepth, Ls>(x, y, z, t, a));
        _source.updateAll();
        importSource(_imported, _source);
        const auto solveStart = std::chrono::steady_clock::now();
        _solver.solve(_solution, _imported, maxIter, precision);
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - solveStart).count();
        result.iterations = _solver.lastIterations();
        result.residue = _solver.lastResidue();
        rootLogger.info("MDWF m_res: source component ", a, " of 12: ", result.iterations,
                        " even/odd CG iterations, residue ", result.residue, ", ", result.seconds, " s");
        LatticeContainer<true, double> reduction(_gauge.getComm());
        reduction.adjustSize(lat.vol4);
        const std::array<size_t, 4> extents = mdwfLatticeExtents(lat);
        CalcGSite<All, HaloDepth> calcGSite;
        std::vector<double> slices;
        for (int kind = 0; kind < 2; kind++) {
            for (int mu = 0; mu < 4; mu++) {
                reduction.template iterateFunctor<BLOCKSIZE>(MDWFResidualMassDensity<HaloDepth, Ls>(_solution, kind == 1),
                                                             calcGSite, MDWFWriteAtSlices<HaloDepth>(mu), lat.vol4);
                reduction.reduceStackedLocal(slices, extents[mu], lat.vol4 / extents[mu]);
                ((kind == 0) ? result.pp : result.j5q)[mu].assign(slices.begin(), slices.begin() + extents[mu]);
            }
        }
        return result;
    }

    MDWFResidualMassCorrelators measure(int x, int y, int z, int t, int maxIter, double precision) {
        typedef GIndexer<All, HaloDepth> GInd;
        MDWFResidualMassCorrelators result{{}, {}, 0, 0.0};
        mdwfResizeDirectionalCorrelators(result.pp, GInd::getLatData());
        mdwfResizeDirectionalCorrelators(result.j5q, GInd::getLatData());
        for (int a = 0; a < 12; a++) {
            const MDWFResidualMassComponent c = measureComponent(a, x, y, z, t, maxIter, precision);
            result.iterations += c.iterations;
            result.maxResidue = std::max(result.maxResidue, c.residue);
            for (int mu = 0; mu < 4; mu++) {
                for (size_t n = 0; n < c.pp[mu].size(); n++) {
                    result.pp[mu][n] += c.pp[mu][n];
                    result.j5q[mu][n] += c.j5q[mu][n];
                }
            }
        }
        return result;
    }

    Solver &solver() {
        return _solver;
    }

    const MDWFMobiusOperatorParameters<double> &params() const {
        return _params;
    }
};
