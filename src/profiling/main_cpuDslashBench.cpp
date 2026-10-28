/*
 * main_cpuDslashBench.cpp
 *
 * Throughput of the HISQ M^dagger M (two even-odd Dslash applications with their halo
 * exchanges, the operator applied in every CG iteration) on random fields, for double and
 * float with 1 and 4 right-hand sides. Prints one "CPU dslash benchmark:" line per case.
 *
 * Usage: cpuDslashBench PARAM_FILE [Lattice='x y z t'] [Nodes='x y z t']
 */

#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"
#include "../modules/dslash/cpuHisqDslash.h"
#include <cmath>

#ifdef CPU_HISQ_DSLASH_SIMD
template<class floatT>
COMPLEX(floatT) component(const Vect3<floatT> &v, int c) {
    return c == 0 ? v.getElement0() : (c == 1 ? v.getElement1() : v.getElement2());
}

//! Largest |a - b| over all components, relative to the largest |b|, over all ranks.
template<class floatT, Layout L, size_t HaloDepthSpin, size_t NStacks>
double relativeDifference(CommunicationBase &comm, Spinorfield<floatT, true, L, HaloDepthSpin, NStacks> &a,
                          Spinorfield<floatT, true, L, HaloDepthSpin, NStacks> &b) {
    Spinorfield<floatT, false, L, HaloDepthSpin, NStacks> hostA(comm), hostB(comm);
    hostA = a;
    hostB = b;
    typedef GIndexer<L, HaloDepthSpin> GInd;
    double maxDiff = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4 / (L == All ? 1 : 2); ++i) {
        for (size_t stack = 0; stack < NStacks; ++stack) {
            const gSiteStack site = GInd::getSiteStack(i, stack);
            const Vect3<floatT> x = hostA.getAccessor().getElement(site), y = hostB.getAccessor().getElement(site);
            for (int c = 0; c < 3; ++c) {
                maxDiff = std::max(maxDiff, (double)abs(component(x, c) - component(y, c)));
                maxRef = std::max(maxRef, (double)abs(component(y, c)));
            }
        }
    }
    return comm.globalMaximum(maxDiff) / comm.globalMaximum(maxRef);
}
#endif

template<class floatT, size_t NStacks>
void benchMdaggM(CommunicationBase &commBase, const double globalVolume, const int repetitions) {
    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;

    Gaugefield<floatT, true, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, true, HaloDepth, U3R14> gauge_Naik(commBase);
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge_smeared.random(d_rand.state);
    gauge_Naik.random(d_rand.state);
    gauge_smeared.updateAll();
    gauge_Naik.updateAll();

    Spinorfield<floatT, true, Even, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, true, Even, HaloDepthSpin, NStacks> spinorOut(commBase);
    spinorIn.gauss(d_rand.state);
    spinorIn.updateAll();
    HisqDSlash<floatT, true, Even, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.1);

    for (int i = 0; i < 3; ++i) dslash.applyMdaggM(spinorOut, spinorIn, true);

#ifdef CPU_HISQ_DSLASH_SIMD
    if (cpuSimd::enabled()) {
        // Same operators with the scalar functor: the vectorised kernel must agree to rounding.
        Spinorfield<floatT, true, Odd, HaloDepthSpin, NStacks> dSimd(commBase), dScalar(commBase);
        Spinorfield<floatT, true, Even, HaloDepthSpin, NStacks> mSimd(commBase), mScalar(commBase);
        dslash.Dslash(dSimd, spinorIn, true);
        dslash.applyMdaggM(mSimd, spinorIn, true);
        cpuSimd::enabledFlag() = false;
        dslash.Dslash(dScalar, spinorIn, true);
        dslash.applyMdaggM(mScalar, spinorIn, true);
        cpuSimd::enabledFlag() = true;
        rootLogger.info("CPU dslash benchmark check: prec=", sizeof(floatT) == 8 ? "double" : "float",
                        " nstacks=", NStacks, " SIMD vs scalar relative max difference: Dslash ",
                        sformatScientific(relativeDifference(commBase, dSimd, dScalar)), ", MdagM ",
                        sformatScientific(relativeDifference(commBase, mSimd, mScalar)));
    }
#endif

    StopWatch<true> timer;
    timer.start();
    for (int i = 0; i < repetitions; ++i) dslash.applyMdaggM(spinorOut, spinorIn, true);
    timer.stop();

    const double seconds = timer.seconds();
    // Each Dslash acts on one parity: 1146 flop per site and right-hand side.
    const double flops = 2.0 * repetitions * NStacks * 0.5 * globalVolume * 1146.0;
    // Memory model per output site and Dslash, no cache reuse between sites: 8 fat links
    // (R18, 18 reals), 8 Naik links (U3R14, 14 reals), 16 neighbour vectors and one output
    // vector (6 reals each) per right-hand side.
    const double bytes = 2.0 * repetitions * 0.5 * globalVolume * sizeof(floatT)
                         * (8 * 18 + 8 * 14 + NStacks * 17 * 6);
    rootLogger.info("CPU dslash benchmark: prec=", sizeof(floatT) == 8 ? "double" : "float",
                    " nstacks=", NStacks, " op=MdagM reps=", repetitions,
                    " time=", sformat("%.4f", seconds), "s",
                    " GFLOP/s=", sformat("%.2f", flops / seconds * 1e-9),
                    " model_GB/s=", sformat("%.1f", bytes / seconds * 1e-9));

    // The same work split into its parts: one Dslash without any halo exchange, and one
    // halo exchange of a spinor (an M^dagger M does two of each).
    Spinorfield<floatT, true, Odd, HaloDepthSpin, NStacks> spinorOdd(commBase);
    timer.reset();
    timer.start();
    for (int i = 0; i < repetitions; ++i) dslash.Dslash(spinorOdd, spinorIn, false);
    timer.stop();
    const double dslashSeconds = timer.seconds() / repetitions;
    timer.reset();
    timer.start();
    for (int i = 0; i < repetitions; ++i) spinorOdd.updateAll();
    timer.stop();
    const double haloSeconds = timer.seconds() / repetitions;
    // Halo update halves: COMM_START packs and posts the messages, COMM_FINISH waits for
    // them and unpacks.
    StopWatch<true> startTimer, finishTimer;
    for (int i = 0; i < repetitions; ++i) {
        startTimer.start();
        spinorOdd.updateAll(AllTypes | COMM_START);
        startTimer.stop();
        finishTimer.start();
        spinorOdd.updateAll(AllTypes | COMM_FINISH);
        finishTimer.stop();
    }
    rootLogger.info("CPU dslash benchmark parts: prec=", sizeof(floatT) == 8 ? "double" : "float",
                    " nstacks=", NStacks,
                    " dslash_ms=", sformat("%.3f", dslashSeconds * 1e3),
                    " dslash_GFLOP/s=", sformat("%.2f", NStacks * 0.5 * globalVolume * 1146.0 / dslashSeconds * 1e-9),
                    " halo_update_ms=", sformat("%.3f", haloSeconds * 1e3),
                    " halo_start_ms=", sformat("%.3f", startTimer.seconds() / repetitions * 1e3),
                    " halo_finish_ms=", sformat("%.3f", finishTimer.seconds() / repetitions * 1e3),
                    " MdagM_ms=", sformat("%.3f", seconds / repetitions * 1e3));
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
    // Keep each case at a few seconds for lattices around 32^3x8 on a full node.
    const int repetitions = 20;
    benchMdaggM<double, 1>(commBase, globalVolume, repetitions);
    benchMdaggM<double, 4>(commBase, globalVolume, repetitions);
    benchMdaggM<float, 1>(commBase, globalVolume, repetitions);
    benchMdaggM<float, 4>(commBase, globalVolume, repetitions);
    rootLogger.info("CPU dslash benchmark: done");
    return 0;
}
