/*
 * main_cpuGaugeForceBench.cpp
 *
 * The Symanzik gauge force of the RHMC integrator (gauge_force() in gaugeActionDeriv.h) on a
 * random gauge field: the AVX-512 kernel (cpuGaugeForce.h) against the scalar functor, then
 * the time per call of both, for double and float. Prints one "CPU gauge force check:" and one
 * "CPU gauge force benchmark:" line per precision.
 *
 * Usage: cpuGaugeForceBench PARAM_FILE [Lattice='x y z t'] [Nodes='x y z t']
 */

#include "../simulateqcd.h"
#include "../gauge/gaugeActionDeriv.h"
#include "../gauge/cpuGaugeForce.h"
#include <cmath>

template<class floatT, size_t HaloDepth>
struct ScalarGaugeForce {
    SU3Accessor<floatT, R18> _gAcc;
    floatT _beta;
    ScalarGaugeForce(SU3Accessor<floatT, R18> gAcc, floatT beta) : _gAcc(gAcc), _beta(beta) {}
    __host__ __device__ SU3<floatT> operator()(gSiteMu site) {
        return gauge_force<floatT, HaloDepth, R18>(_gAcc, site, _beta);
    }
};

#ifdef CPU_HISQ_DSLASH_SIMD
//! Largest |a - b| over all bulk link entries, and largest |b|, over all ranks.
template<class floatT, size_t HaloDepth>
std::pair<double, double> difference(CommunicationBase &comm, Gaugefield<floatT, true, HaloDepth, R18> &a,
                                     Gaugefield<floatT, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    double maxDiff = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; ++i) {
        for (int mu = 0; mu < 4; ++mu) {
            const gSiteMu site = GInd::getSiteMu(i, mu);
            const SU3<floatT> x = a.getAccessor().getLink(site), y = b.getAccessor().getLink(site);
            const SU3<floatT> d = x - y;
            for (int e = 0; e < 9; ++e) {
                const int row = e / 3, col = e % 3;
                maxDiff = std::max(maxDiff, (double)abs(d(row, col)));
                maxRef = std::max(maxRef, (double)abs(y(row, col)));
            }
        }
    }
    return {comm.globalMaximum(maxDiff), comm.globalMaximum(maxRef)};
}
#endif

template<class floatT>
void benchGaugeForce(CommunicationBase &commBase, const double globalVolume, const int repetitions) {
    const size_t HaloDepth = 2;
    const floatT beta = 6.285;
    const char *prec = sizeof(floatT) == 8 ? "double" : "float";

    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase), forceScalar(commBase), forceSimd(commBase);
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge.random(d_rand.state);
    gauge.updateAll();
    const ScalarGaugeForce<floatT, HaloDepth> functor(gauge.getAccessor(), beta);

    forceScalar.iterateOverBulkAllMu(functor);
#ifdef CPU_HISQ_DSLASH_SIMD
    if (cpuGaugeForce<floatT, HaloDepth>(forceSimd, gauge, beta)) {
        const auto [diff, ref] = difference<floatT, HaloDepth>(commBase, forceSimd, forceScalar);
        rootLogger.info("CPU gauge force check: prec=", prec, " SIMD vs scalar max |difference| ",
                        sformatScientific(diff), " (relative ", sformatScientific(diff / ref), ")");
    } else {
        rootLogger.info("CPU gauge force check: prec=", prec, " SIMD kernel does not apply");
    }
#endif

    // gauge_force() does 70 SU(3) products per link (23 per plane and the final U * staple),
    // 198 flop each; additions are not counted.
    const double flops = 4.0 * globalVolume * 70 * 198;
    StopWatch<true> timer;
    timer.start();
    for (int i = 0; i < repetitions; ++i) forceScalar.iterateOverBulkAllMu(functor);
    timer.stop();
    const double scalarSeconds = timer.seconds() / repetitions;
    double simdSeconds = 0.0;
#ifdef CPU_HISQ_DSLASH_SIMD
    timer.reset();
    timer.start();
    for (int i = 0; i < repetitions; ++i) cpuGaugeForce<floatT, HaloDepth>(forceSimd, gauge, beta);
    timer.stop();
    simdSeconds = timer.seconds() / repetitions;
#endif
    rootLogger.info("CPU gauge force benchmark: prec=", prec, " reps=", repetitions,
                    " scalar_ms=", sformat("%.3f", scalarSeconds * 1e3),
                    " scalar_GFLOP/s=", sformat("%.1f", flops / scalarSeconds * 1e-9),
                    " simd_ms=", sformat("%.3f", simdSeconds * 1e3),
                    " simd_GFLOP/s=", sformat("%.1f", simdSeconds > 0 ? flops / simdSeconds * 1e-9 : 0.0));
}

int main(int argc, char **argv) {
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);
    LatticeParameters param;
    param.readfile(commBase, "../parameter/profiling/cpuDslashBench.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(4, param, commBase);

    const double globalVolume = static_cast<double>(param.latDim[0]) * param.latDim[1]
                                * param.latDim[2] * param.latDim[3];
    const int repetitions = 5;
    benchGaugeForce<double>(commBase, globalVolume, repetitions);
    benchGaugeForce<float>(commBase, globalVolume, repetitions);
    rootLogger.info("CPU gauge force benchmark: done");
    return 0;
}
