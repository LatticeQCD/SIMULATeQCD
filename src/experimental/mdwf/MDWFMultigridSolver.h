/*
 * Multigrid (two-grid) solver for the MDWF measurement solves (Stage 1 of DSDRMDWFruns/SQCDruns/multigrid/DESIGN.md).
 *
 * MDWFMobiusEvenOddMGSolver is MDWFMobiusEvenOddMixedSolver (same interface, same even/odd pre- and post-processing in
 * double) whose Schur-normal solve A x = b, A = Mhat^+ Mhat, is replaced by the flexible CG of MDWFTwoGrid.h,
 * preconditioned by the symmetric two-grid cycle, for the masses |m_f| <= mg_mass_max. Every other mass, and every
 * mass when the multigrid is off (mdwfMultigridConfig().enabled = false, the default), runs the unchanged mixed CG:
 * with the multigrid off the output is byte-identical to MDWFMobiusEvenOddMixedSolver.
 *
 * Coarse space (per gauge field and mass, built at the first solve and shared by all solvers of that mass, e.g. the
 * m_res point-source columns and the light noise vectors; mdwfMultigridRelease frees it):
 *   ns near-null vectors = setup_iterations plain double-CG iterations on A x = hash noise (seed + 7919 (k + 1)),
 *   blocks of block[0..3] even sites x full Ls x chirality, A_c = R A P by probing, dense Cholesky of A_c (cuSOLVER:
 *   needs the build option MDWF_MG_CUSOLVER, which defines MDWF_TWOGRID_CUSOLVER), Chebyshev smoother of degree
 *   `degree` on [lambda_max / smoother_ratio, lambda_max]. Defaults = the best Stage 0 point (pilot3, 2026-11-16:
 *   b1633_ml40 point source, inviter 1000, ns 32, d 1, 4^3x8 blocks: 12 solves incl. setup 0.27 of plain double CG).
 * Outer solver: FCG in double with the production stopping rule |r|^2 <= precision^2 max(|b|^2, 1) and a true-residual
 * check, so the solution meets the same tolerance as the mixed CG; the preconditioner only changes the iteration count.
 * Single rank only (as mdwfMeasure).
 */

#pragma once

#include "MDWFMixedPrecisionSolver.h"
#include "MDWFTwoGrid.h"

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct MDWFMultigridConfig {
    bool enabled = false;
    double massMax = 0.01;              // |m_f| <= massMax uses the multigrid (light and m_res masses, not the strange)
    std::vector<int> block{4, 4, 4, 8};
    int ns = 32;
    int setupIterations = 1000;
    int degree = 1;
    double smootherRatio = 30.0;
    uint64_t seed = 20261115;           // the Stage 0 study seed: same near-null vectors as the study
    int maxCoarseDim = 33000;
};

inline MDWFMultigridConfig &mdwfMultigridConfig() {
    static MDWFMultigridConfig config;
    return config;
}

// Coarse space and operator for one (gauge field, M5, m_f, b5).
template<size_t HaloDepth, size_t Ls>
class MDWFMultigridContext {
public:
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using TG = MDWFTwoGrid<HaloDepth, Ls>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_comm;
    Op _op;
    TG _tg;
    std::unique_ptr<Vec> _b, _x;
    MDWFTwoGridStats _stats;
    double _setupSeconds = 0.0, _lambdaMax = 0.0;
    long _setupApplications = 0;
    long _solves = 0, _outer = 0, _applications = 0;
    double _solveSeconds = 0.0;

public:
    MDWFMultigridContext(Gauge &fermionGauge, double M5, double mass, double b5, const MDWFMultigridConfig &c)
        : _comm(fermionGauge.getComm()), _op(fermionGauge, M5, mass, b5, false, "MDWF_mg_op"),
          _tg(fermionGauge.getComm(), c.block) {
        const auto t0 = std::chrono::steady_clock::now();
        if (2 * _tg.nBlocks() * c.ns > c.maxCoarseDim) {
            throw std::runtime_error(stdLogger.fatal("MDWF multigrid: coarse dimension ", 2 * _tg.nBlocks() * c.ns,
                                                     " > ", c.maxCoarseDim, " (mg_ns or mg_block)"));
        }
        _op.refresh();
        {
            Vec start(_comm);
            start.noise(c.seed ^ 0x5eedULL);
            _lambdaMax = 1.1 * mdwfDeflLambdaMax(_op, start, 60);
        }
        const long a0 = _op.applications();
        Set raw;
        mdwfDeflAllocate(raw, c.ns, _comm);
        {
            Vec v(_comm);
            for (int k = 0; k < c.ns; k++) {
                v.noise(c.seed + 7919ULL * (k + 1));
                const int nIt = c.setupIterations;
                Vec &keep = *raw[k];
                mdwfDeflCGSnapshots<Op, HaloDepth, Ls>(_op, v, {nIt},
                    [&](int n, const Vec &w, double) { if (n == nIt) { keep = w; } }, _comm);
                keep *= COMPLEX(double)(1.0 / std::sqrt(keep.norm2()), 0.0);
            }
        }
        _setupApplications = _op.applications() - a0;
        _stats.setupApplications = _setupApplications;
        _stats.setupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        _tg.setBasis(raw, c.ns, _stats);
        _tg.buildCoarse(_op, _stats);
        _tg.setSmoother(_lambdaMax / c.smootherRatio, _lambdaMax, c.degree);
        _b.reset(new Vec(_comm));
        _x.reset(new Vec(_comm));
        _setupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rootLogger.info("MDWF multigrid setup: mass ", mass, ", blocks ", c.block[0], "x", c.block[1], "x", c.block[2],
                        "x", c.block[3], " x Ls ", Ls, ", ns ", c.ns, ", coarse dimension ", _tg.coarseDim(),
                        ", lambda_max ", _lambdaMax, ", smoother [", _lambdaMax / c.smootherRatio, ", ", _lambdaMax,
                        "] degree ", c.degree, "; near-null ", _setupApplications, " applications of A (",
                        _stats.setupSeconds, " s), orthonormalisation ", _stats.orthoSeconds, " s, probing ",
                        _stats.probeApplications, " applications (", _stats.probeSeconds, " s), Cholesky ",
                        _stats.factorSeconds, " s, Hermiticity defect ", _stats.hermiticity, "; total ",
                        _setupSeconds, " s");
    }

    // A x = b for the even-site fields of the measurement solver (same layout as MDWFCompactVec::Field).
    template<class Field>
    MDWFTwoGridSolveResult solve(Field &x, const Field &b, int maxIter, double precision) {
        _b->pack(b);
        const MDWFTwoGridSolveResult r = mdwfTwoGridFCG<HaloDepth, Ls>(_tg, _op, *_x, *_b, precision, maxIter, _comm);
        _x->unpack(x);
        _solves++;
        _outer += r.iterations;
        _applications += r.applications;
        _solveSeconds += r.seconds;
        return r;
    }

    void summary(double mass) const {
        rootLogger.info("MDWF multigrid summary: mass ", mass, ", setup ", _setupSeconds, " s, ", _solves, " solves, ",
                        _outer, " outer iterations, ", _applications, " applications of A, ", _solveSeconds,
                        " s in the solves (", _solves > 0 ? _solveSeconds / _solves : 0.0, " s per solve)");
    }
};

template<size_t HaloDepth, size_t Ls>
using MDWFMultigridRegistry =
    std::map<std::tuple<const void *, double, double, double>, std::shared_ptr<MDWFMultigridContext<HaloDepth, Ls>>>;

template<size_t HaloDepth, size_t Ls>
MDWFMultigridRegistry<HaloDepth, Ls> &mdwfMultigridRegistry() {
    static MDWFMultigridRegistry<HaloDepth, Ls> registry;
    return registry;
}

// Frees every coarse space (call before the end of the run: GPU memory must not outlive the communication setup).
template<size_t HaloDepth, size_t Ls>
void mdwfMultigridRelease() {
    auto &reg = mdwfMultigridRegistry<HaloDepth, Ls>();
    for (const auto &kv : reg) {
        kv.second->summary(std::get<2>(kv.first));
    }
    reg.clear();
}

template<size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusEvenOddMGSolver : public MDWFMobiusEvenOddMixedSolver<HaloDepthGauge, HaloDepthSpin, Ls> {
    static_assert(HaloDepthGauge == HaloDepthSpin, "MDWF multigrid: HaloDepthGauge must equal HaloDepthSpin");
    using Base = MDWFMobiusEvenOddMixedSolver<HaloDepthGauge, HaloDepthSpin, Ls>;
    using Ctx = MDWFMultigridContext<HaloDepthGauge, Ls>;
    using SpinorE = typename Base::SpinorE;
    using ExternalResult = typename Base::ExternalResult;

    Gaugefield<double, true, HaloDepthGauge, R18> &_gaugeRef;
    double _M5, _mf, _b5;
    std::shared_ptr<Ctx> _ctx;

    Ctx &context() {
        if (!_ctx) {
            auto &reg = mdwfMultigridRegistry<HaloDepthGauge, Ls>();
            const auto key = std::make_tuple(static_cast<const void *>(&_gaugeRef), _M5, _mf, _b5);
            auto it = reg.find(key);
            if (it == reg.end()) {
                it = reg.emplace(key, std::make_shared<Ctx>(_gaugeRef, _M5, _mf, _b5, mdwfMultigridConfig())).first;
            }
            _ctx = it->second;
        }
        return *_ctx;
    }

public:
    MDWFMobiusEvenOddMGSolver(Gaugefield<double, true, HaloDepthGauge, R18> &gauge, double M5, double mf, double b5,
                              double csw, const std::string &name, double delta = 0.1)
        : Base(gauge, M5, mf, b5, csw, name, delta), _gaugeRef(gauge), _M5(M5), _mf(mf), _b5(b5) {
        const MDWFMultigridConfig &c = mdwfMultigridConfig();
        if (!c.enabled || std::abs(mf) > c.massMax) {
            return;   // unchanged mixed CG
        }
        if (csw != 0.0) {
            throw std::runtime_error(stdLogger.fatal("MDWF multigrid: c_sw = 0 only (MDWFDeflOperator)"));
        }
        rootLogger.info("MDWF multigrid: solver ", name, " (mass ", mf, ") uses the two-grid FCG");
        this->setExternalSolve([this](SpinorE &x, const SpinorE &b, int maxIter, double precision,
                                      const char *what) {
            const MDWFTwoGridSolveResult r = context().solve(x, b, maxIter, precision);
            rootLogger.info("MDWF multigrid FCG (", what, "): ", r.iterations, " outer iterations, ", r.applications,
                            " applications of A, ", r.seconds, " s, true residual ", r.trueResidual);
            return ExternalResult{r.iterations, r.trueResidual, r.converged};
        });
    }
};
