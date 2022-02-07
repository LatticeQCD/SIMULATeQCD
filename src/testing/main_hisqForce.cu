/* 
 * main_hisqForce.cu                                                               
 * 
 * This program tests the HISQ fermion force and has to yield the same result as the gaction_test_hisqforce.cpp
 * in the BielefeldGPUCode.
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/HISQ/hisqForce.h"
#include "../gauge/gauge_kernels.cu"


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
    gpuProfilerStart();
    ip_dot_f2_hisq.TestForce(SpinorIn,force,d_rand);
    gpuProfilerStop();
    timer.stop();

//    force.writeconf_nersc("force_reference",3,2);

    force_host=force;

    GSU3<PREC> test1 = force_host.getAccessor().getLink(GInd::getSiteMu(0,0,0,3,3));

    rootLogger.info("Time: " ,  timer);
    rootLogger.info("Force:");
    rootLogger.info(test1.getLink00(), test1.getLink01(), test1.getLink02());
    rootLogger.info(test1.getLink10(), test1.getLink11(), test1.getLink12());
    rootLogger.info(test1.getLink20(), test1.getLink21(), test1.getLink22());
    
    Gaugefield<PREC,true,HaloDepth> force_reference(commBase);

//    rootLogger.warn("gauge_file=",rhmc_param.GaugefileName());
    force_reference.readconf_nersc(rhmc_param.GaugefileName());
    
    force.writeconf_nersc("../test_conf/force_testrun",3,2);

    force.readconf_nersc("../test_conf/force_testrun");

    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    LatticeContainer<true, int> dummy(commBase);
    dummy.adjustSize(elems);
    
    dummy.template iterateOverBulk<All,HaloDepth>(count_faulty_links<PREC,true,HaloDepth,R18>(force,force_reference,3e-9));

    int faults = 0;
    dummy.reduce(faults,elems);

    rootLogger.info(faults, " faulty links found!");

    if (faults == 0) {
        rootLogger.info(CoutColors::green, "Force is correct", CoutColors::reset);
    }
    else {
        rootLogger.error("Force is wrong!");
        return 1;
    }

       
    return 0;
}

