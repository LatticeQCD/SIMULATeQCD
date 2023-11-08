/*
 * main_hisqForce.cpp
 *
 * This program tests the HISQ fermion force and has to yield the same result as the gaction_test_hisqforce.cpp
 * in the BielefeldGPUCode.
 *
 */

#include "../simulateqcd.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../modules/hisq/hisqForce.h"
#include "testing.h"

#define PREC double
#define USE_GPU true


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    StopWatch<true> timer;
    CommunicationBase commBase(&argc, &argv);
    RhmcParameters rhmc_param;
    rhmc_param.readfile(commBase,"../parameter/tests/hisqForce.param", argc, argv);

    commBase.init(rhmc_param.nodeDim());

    const size_t HaloDepth = 0;
    const size_t HaloDepthSpin = 4;

    rootLogger.info("Initialize Lattice");
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,rhmc_param,commBase);

    RationalCoeff rat;
    rat.readfile(commBase, rhmc_param.rat_file());

    Gaugefield<PREC, false,HaloDepth> force_host(commBase);
    rootLogger.info("Initialize Gaugefield & Spinorfield");

    Gaugefield<PREC, true, HaloDepth,R18> gauge(commBase);
    Gaugefield<PREC, true, HaloDepth> gaugeLvl2(commBase);
    Gaugefield<PREC, true, HaloDepth,U3R14> gaugeNaik(commBase);
    Gaugefield<PREC, true, HaloDepth> force(commBase);
    Spinorfield<PREC, true, Even, HaloDepthSpin> SpinorIn(commBase);

    grnd_state<true> d_rand;

    initialize_rng(rhmc_param.seed(),d_rand);
    rootLogger.info("seed=",rhmc_param.seed());
    gauge.random(d_rand.state);
    gauge.updateAll();

    HisqSmearing<PREC, true, HaloDepth,R18> smearing(gauge,gaugeLvl2,gaugeNaik);
    smearing.SmearAll();

    AdvancedMultiShiftCG<PREC, 14> CG;

    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin,1> dslash(gaugeLvl2,gaugeNaik,0.0);
    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 14> dslash_multi(gaugeLvl2,gaugeNaik,0.0);

    HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18, true> ip_dot_f2_hisq(gauge,force,CG,dslash,dslash_multi,rhmc_param,rat,smearing);

    timer.start();
    // gpuError_t gpuErr;
    // gpuErr = gpuProfilerStart();
    // if (gpuErr) GpuError("hisqForce: gpuProfilerStart", gpuErr);
    ip_dot_f2_hisq.TestForce(SpinorIn,force,d_rand);
    // gpuErr = gpuProfilerStop();
    // if (gpuErr) GpuError("hisqForce: gpuProfilerStop", gpuErr);
    timer.stop();

    force_host=force;

    SU3<PREC> test1 = force_host.getAccessor().getLink(GInd::getSiteMu(0,0,0,3,3));

    rootLogger.info("Time: " ,  timer);
    rootLogger.info("Force:");
    rootLogger.info(test1.getLink00(), test1.getLink01(), test1.getLink02());
    rootLogger.info(test1.getLink10(), test1.getLink11(), test1.getLink12());
    rootLogger.info(test1.getLink20(), test1.getLink21(), test1.getLink22());

    Gaugefield<PREC,true,HaloDepth> force_reference(commBase);

    force_reference.readconf_nersc(rhmc_param.GaugefileName());

    // readconf_nersc does some reunitarization. Writing and reading makes sure the comparison between this generated
    // force field and force_reference have both been reunitarized in the same way.
    force.writeconf_nersc("../test_conf/force_testrun",3,2);
    force.readconf_nersc("../test_conf/force_testrun");
    
    rootLogger.info("starting field comparison");
    bool pass = compare_fields<PREC,HaloDepth,true,R18>(force,force_reference,1e-8);
    if (pass) {
        rootLogger.info(CoutColors::green, "Force is correct", CoutColors::reset);
    } else {
        rootLogger.error("Force is wrong!");
        return 1;
    }

    return 0;
}

