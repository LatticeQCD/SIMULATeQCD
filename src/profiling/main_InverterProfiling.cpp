/*
 * main_InverterProfiling.cpp
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

//Run function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, size_t Multistack, bool onDevice>
void run_func(CommunicationBase &commBase, RhmcParameters &param, RationalCoeff &rat)
{
    StopWatch<true> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayoutRHS, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);

    rootLogger.info("Read configuration");

    rootLogger.info("Initialize random state");
    grnd_state<onDevice> d_rand;
    initialize_rng(13333, d_rand);

    gauge.one();
    gauge.updateAll();

    smearing.SmearAll();

    rootLogger.info("Testing multishift inverter:");
    // r_lf test
    rootLogger.info("Starting r_lf test");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_ud(gauge_smeared, gauge_Naik, 0.0);

    SimpleArray<floatT, Multistack> shifts(0.0);

    for (size_t i = 1; i <rat.r_inv_sf_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_sf_den[i]-rat.r_inv_sf_den[0];
    }
    shifts[0] = rat.r_inv_sf_den[0] + param.m_s()*param.m_s();

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorInMulti(commBase);
    spinorInMulti.gauss(d_rand.state);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, Multistack> spinorOutMulti(commBase);

    AdvancedMultiShiftCG<floatT, Multistack> cgM;


    // make phi:
    gpuProfilerStart();
    timer.start();
    cgM.invert(dslashMulti_ud, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());
    timer.stop();
    gpuProfilerStop();
    rootLogger.info("Time for inversion with multishift CG: " , timer);
}



int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        CommunicationBase commBase(&argc, &argv);
        RhmcParameters param;
        param.readfile(commBase, "../parameter/profiling/InverterProfile.param", argc, argv);
        param.cgMax.set(4000);

        RationalCoeff rat;
        rat.readfile(commBase, param.rat_file());

        commBase.init(param.nodeDim());

        const int HaloDepthSpin = 4;
        initIndexer(HaloDepthSpin, param, commBase);

        rootLogger.info("-------------------------------------");
        rootLogger.info("Running on Device");
        rootLogger.info("-------------------------------------");

        rootLogger.info("------------------");
        rootLogger.info("Testing Even - Odd");
        rootLogger.info("------------------");
        run_func<float, Even, Odd, 1, 14, true>(commBase, param, rat);
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        return -1;
    }
}
