/*
 * HMC multigrid rescue (2026-11-18, DSDRMDWFruns/SQCDruns/multigrid/HMC_RESCUE.md).
 *
 * The b1671_ml80 lightest Hasenbusch factor (f0, bare m_l < 0) has rare MD configurations with a near-zero mode of
 * Mhat(m_l) (lambda_min of the Schur-normal operator ~1e-9); mixed CG (and double CG) then stall at 200k iterations and
 * the stream dies. The two-grid FCG of MDWFTwoGrid.h converges on those fields (multigrid/logs/failfield_2970775).
 * With MDWF_HMC_MG_RESCUE=1 an UNCONVERGED mixed solve of MDWFEvenOddMixedSchurSolver::solve (any HMC even/odd ratio
 * solve) whose operator has |m_f| <= mass_max, m_+ = m_-, c_sw = 0 is finished by the two-grid FCG on the correction
 * A d = b - A x, with the stopping rule of the mixed CG checked on the HMC's own double operator. The rescued solution
 * therefore meets the same tolerance as a converged mixed solve; the HMC's accept/reject and the MD integrator are
 * unchanged. Switch off (default; or a binary built without MDWF_HMC_MG_RESCUE): nothing here runs and the arithmetic
 * is bit-identical.
 *
 * Environment (read once):
 *   MDWF_HMC_MG_RESCUE=1          on
 *   MDWF_HMC_MG_AFTER=N           cap of the mixed-CG iterations for eligible solves (0 = the solve's max_iter, i.e.
 *                                 rescue only after a full failure); a capped solve that did not converge is rescued
 *   MDWF_HMC_MG_MASS_MAX=0.01     eligible |m_f|
 *   MDWF_HMC_MG_BLOCK=4,4,4,8     block extents (must divide the LOCAL lattice: 32x16x16x8 on Nodes 1 2 2 1)
 *   MDWF_HMC_MG_NS=32, MDWF_HMC_MG_SETUP_ITER=1000, MDWF_HMC_MG_DEGREE=1, MDWF_HMC_MG_SMOOTHER_RATIO=30,
 *   MDWF_HMC_MG_SEED=20261118     the measurement defaults (Stage 1)
 *   MDWF_HMC_MG_MAX_OUTER=50000   outer FCG iterations per attempt
 *   MDWF_HMC_MG_REUSE=1           keep the near-null vectors between rescues (same operator parameters): only lambda_max,
 *                                 A_c (probing) and its factor are recomputed for the new gauge field; if the reused space
 *                                 does not converge within MAX_OUTER, a full setup follows and the solve is retried once
 */

#pragma once

#include "MDWFTwoGrid.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MDWFHmcMGRescueConfig {
    bool enabled = false;
    int mixedMaxIter = 0;
    double massMax = 0.01;
    std::vector<int> block{4, 4, 4, 8};
    int ns = 32;
    int setupIterations = 1000;
    int degree = 1;
    double smootherRatio = 30.0;
    uint64_t seed = 20261118;
    int maxOuter = 50000;
    bool reuse = true;
};

inline const MDWFHmcMGRescueConfig &mdwfHmcMGRescueConfig() {
    static const MDWFHmcMGRescueConfig config = [] {
        MDWFHmcMGRescueConfig c;
        auto env = [](const char *name) -> const char * { return std::getenv(name); };
        if (const char *e = env("MDWF_HMC_MG_RESCUE")) { c.enabled = std::string(e) == "1"; }
        if (const char *e = env("MDWF_HMC_MG_AFTER")) { c.mixedMaxIter = std::max(0, std::atoi(e)); }
        if (const char *e = env("MDWF_HMC_MG_MASS_MAX")) { c.massMax = std::atof(e); }
        if (const char *e = env("MDWF_HMC_MG_BLOCK")) {
            std::vector<int> b;
            std::stringstream ss(e);
            std::string item;
            while (std::getline(ss, item, ',')) {
                b.push_back(std::atoi(item.c_str()));
            }
            if (b.size() == 4) {
                c.block = b;
            }
        }
        if (const char *e = env("MDWF_HMC_MG_NS")) { c.ns = std::atoi(e); }
        if (const char *e = env("MDWF_HMC_MG_SETUP_ITER")) { c.setupIterations = std::atoi(e); }
        if (const char *e = env("MDWF_HMC_MG_DEGREE")) { c.degree = std::atoi(e); }
        if (const char *e = env("MDWF_HMC_MG_SMOOTHER_RATIO")) { c.smootherRatio = std::atof(e); }
        if (const char *e = env("MDWF_HMC_MG_SEED")) { c.seed = std::strtoull(e, nullptr, 10); }
        if (const char *e = env("MDWF_HMC_MG_MAX_OUTER")) { c.maxOuter = std::atoi(e); }
        if (const char *e = env("MDWF_HMC_MG_REUSE")) { c.reuse = std::string(e) == "1"; }
        if (c.enabled) {
            rootLogger.info("MDWF HMC MG rescue: ON, mixed cap ", c.mixedMaxIter, " (0 = max_iter), |m_f| <= ", c.massMax,
                            ", blocks ", c.block[0], "x", c.block[1], "x", c.block[2], "x", c.block[3], ", ns ", c.ns,
                            ", setup iterations ", c.setupIterations, ", degree ", c.degree, ", smoother ratio ",
                            c.smootherRatio, ", seed ", c.seed, ", max outer ", c.maxOuter, ", reuse ", c.reuse ? 1 : 0);
        }
        return c;
    }();
    return config;
}

// Two-grid space for one operator (gauge field object, M5, m_f, b5, operator order).
template<size_t HaloDepth, size_t Ls>
class MDWFHmcMGRescueContext {
public:
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using TG = MDWFTwoGrid<HaloDepth, Ls>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_comm;
    const void *_gauge;
    double _M5, _mass, _b5;
    bool _mmdag;
    MDWFHmcMGRescueConfig _c;
    Op _op;
    TG _tg;
    double _lambdaMax = 0.0;

    double estimateLambdaMax() {
        Vec start(_comm);
        start.noise(_c.seed ^ 0x5eedULL);
        return 1.1 * mdwfDeflLambdaMax(_op, start, 60);
    }

public:
    MDWFHmcMGRescueContext(Gauge &fermionGauge, double M5, double mass, double b5, bool mmdag,
                           const MDWFHmcMGRescueConfig &c)
        : _comm(fermionGauge.getComm()), _gauge(&fermionGauge), _M5(M5), _mass(mass), _b5(b5), _mmdag(mmdag), _c(c),
          _op(fermionGauge, M5, mass, b5, mmdag, "MDWF_hmc_mgr_op"), _tg(fermionGauge.getComm(), c.block) {
        const auto t0 = std::chrono::steady_clock::now();
        if (2 * _tg.nBlocks() * c.ns > 33000) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC MG rescue: coarse dimension ", 2 * _tg.nBlocks() * c.ns,
                                                     " > 33000 (MDWF_HMC_MG_NS or MDWF_HMC_MG_BLOCK)"));
        }
        _op.refresh();
        _lambdaMax = estimateLambdaMax();
        MDWFTwoGridStats stats;
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
        stats.setupApplications = _op.applications() - a0;
        stats.setupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        _tg.setBasis(raw, c.ns, stats);
        _tg.buildCoarse(_op, stats);
        _tg.setSmoother(_lambdaMax / c.smootherRatio, _lambdaMax, c.degree);
        rootLogger.info("MDWF HMC MG rescue: full setup, mass ", mass, ", mmdag ", mmdag ? 1 : 0, ", coarse dimension ",
                        _tg.coarseDim(), ", lambda_max ", _lambdaMax, "; near-null ", stats.setupApplications,
                        " applications (", stats.setupSeconds, " s), probing ", stats.probeApplications,
                        " applications (", stats.probeSeconds, " s), factor ", stats.factorSeconds,
                        " s, Hermiticity defect ", stats.hermiticity, "; total ",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), " s");
    }

    bool matches(const Gauge &fermionGauge, double M5, double mass, double b5, bool mmdag) const {
        return _gauge == &fermionGauge && _M5 == M5 && _mass == mass && _b5 == b5 && _mmdag == mmdag;
    }

    // New gauge field, same near-null vectors: lambda_max, A_c and its factor again.
    void refreshCoarse() {
        const auto t0 = std::chrono::steady_clock::now();
        _op.refresh();
        _lambdaMax = estimateLambdaMax();
        MDWFTwoGridStats stats;
        _tg.buildCoarse(_op, stats);
        _tg.setSmoother(_lambdaMax / _c.smootherRatio, _lambdaMax, _c.degree);
        rootLogger.info("MDWF HMC MG rescue: reused near-null vectors, mass ", _mass, ", lambda_max ", _lambdaMax,
                        ", probing ", stats.probeApplications, " applications (", stats.probeSeconds, " s), factor ",
                        stats.factorSeconds, " s; total ",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), " s");
    }

    MDWFTwoGridSolveResult solve(Vec &x, const Vec &b, double precision) {
        return mdwfTwoGridFCG<HaloDepth, Ls>(_tg, _op, x, b, precision, _c.maxOuter, _comm);
    }
};
