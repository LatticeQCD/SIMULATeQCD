/*
 * main_HisqSmearingImagmuTest.cu
 *
 * D. Bollweg
 *
 * Quick single-GPU test comparing the output of the imaginary mu smearing against the old results from the ParallelGPUCode.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "testing.h"

#define PREC double
#define USE_GPU true

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    LatticeParameters param;
    StopWatch<true> timer;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/hisqSmearingTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;

    rootLogger.info("Initialize Lattice");
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);

    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC, true, HaloDepth> gauge_in(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_smeared_reference(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_Lv2(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_naik(commBase);
    GCOMPLEX(PREC) imgchmp;

    PREC chmp=0.4;

    timer.start();
    HisqSmearing<PREC, USE_GPU, HaloDepth,R18,R18,R18,R18> smearing(gauge_in, gauge_Lv2, gauge_naik);
    timer.stop();
    rootLogger.info("Time for initializing HisqSmearing: ",timer);
    timer.reset();

    rootLogger.info("Read configuration");
    gauge_in.readconf_nersc("../test_conf/gauge12750");

    gauge_in.updateAll();

    timer.start();
    smearing.SmearAll(chmp);
    timer.stop();
    rootLogger.info("Time for full smearing: ",timer);
    gauge_smeared_reference.readconf_nersc("../test_conf/smearing_imagmu_reference_conf");
    gauge_Lv2.writeconf_nersc("../test_conf/smearing_imagmu_testrun");

    gauge_Lv2.readconf_nersc("../test_conf/smearing_imagmu_testrun");

    bool pass = compare_fields<PREC,HaloDepth,true,R18>(gauge_Lv2,gauge_smeared_reference);

    if (pass) {
        rootLogger.info(CoutColors::green, "Test passed!", CoutColors::reset);
    } else {
        rootLogger.error("Test failed!");
        return 1;
    }

    return 0;
}
