/*
 * main_MixedPrecInverterTest.cpp
 *
 * Dennis Bollweg
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"


template<class floatT, class floatT_inner, Layout LatLayout, size_t NStacks, bool onDevice>
void run_func(CommunicationBase &commBase, RhmcParameters &param, RationalCoeff &rat) {

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_naik(commBase);
    Gaugefield<floatT_inner, onDevice, HaloDepth, R18> gauge_smeared_half(commBase);
    Gaugefield<floatT_inner, onDevice, HaloDepth, U3R14> gauge_naik_half(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_naik);

    StopWatch<true> timer;

    grnd_state<onDevice> d_rand;
    initialize_rng(243, d_rand);

    gauge.random(d_rand.state);

    smearing.SmearAll();
    gauge_smeared_half.convert_precision(gauge_smeared);
    gauge_naik_half.convert_precision(gauge_naik);

    ConjugateGradient<floatT, NStacks> cg;

    HisqDSlash<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_naik, param.m_ud());
    HisqDSlash<floatT_inner, onDevice, LayoutSwitcher<LatLayout>(), HaloDepth, HaloDepthSpin, NStacks> dslash_half(gauge_smeared_half, gauge_naik_half, param.m_ud());

    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepthSpin, NStacks> spinorOut(commBase);

    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepthSpin, NStacks> spinorOut2(commBase);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);

    timer.start();
    cg.invert_mixed(dslash, dslash_half, spinorOut, spinorIn, param.cgMax(), param.residue(), param.cgMixedPrec_delta());

    timer.stop();

    dslash.applyMdaggM(spinorOut2, spinorOut);

    spinorOut = spinorOut2 - spinorIn;

    SimpleArray<GCOMPLEX(double), NStacks> dot1(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);

    dot1 = spinorOut.dotProductStacked(spinorOut);
    dot2 = spinorIn.dotProductStacked(spinorIn);

    rootLogger.info("dot1 " ,  dot1[0] ,  " dot2 " ,  dot2[0]);
    SimpleArray<double, NStacks> err_arr(0.0);

    err_arr = real<double>(dot1)/real<double>(dot2);

    for (size_t i = 0; i < NStacks; ++i) {
        rootLogger.info(err_arr[i]);
    }

    rootLogger.info("relative error of (D^+D) * (D^+D)^-1 * phi - phi : " ,  max(err_arr));
    rootLogger.info("Time for inversion: " ,  timer);
    bool success = true;
    if (!(max(err_arr) < 10*param.residue()))
        success = false;

    if (success)
        rootLogger.info("Inverter test: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
    else
        throw std::runtime_error(stdLogger.fatal("Inverter test failed!"));
}


int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(DEBUG);

        CommunicationBase commBase(&argc, &argv);
        RhmcParameters param;

        param.readfile(commBase, "../parameter/tests/MixedPrecInverterTest.param", argc, argv);

        RationalCoeff rat;

        rat.readfile(commBase, param.rat_file());

        commBase.init(param.nodeDim(), param.gpuTopo());

        const int HaloDepthSpin = 4;
        initIndexer(HaloDepthSpin, param, commBase);

        rootLogger.info("Running mixed precision inverter test");

        rootLogger.info("testing float-half");
        run_func<float, __half, Even, 1, true>(commBase, param, rat);

        rootLogger.info("testing double-float");
        run_func<double, float, Even, 1, true>(commBase, param, rat);

        rootLogger.info("testing double-half");
        run_func<double, __half, Even, 1, true>(commBase, param, rat);

        return 0;
    }

    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        return -1;
    }
}
