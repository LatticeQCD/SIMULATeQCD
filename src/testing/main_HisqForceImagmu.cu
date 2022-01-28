/* 
 * main_HisqForceImagmu.cu
 *
 * J. Goswami 
 *
 * Quick test of the HISQ force with imaginary chemical potential. Compares against results from ParallelGPUCode/BielefeldGPUcode. 
 *
 */
#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/HISQ/hisqForce.h"
#include "../gauge/gauge_kernels.cu" 

#define PREC float


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters rhmc_param;
    rhmc_param.readfile(commBase,"../parameter/tests/hisqForceImagMu.param", argc, argv);

    commBase.init(rhmc_param.nodeDim());


    const size_t HaloDepth = 0;
    const size_t HaloDepthSpin = 4;
    PREC chmp=rhmc_param.mu_f();

    rootLogger.info( "Initialize Lattice");
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,rhmc_param,commBase);


    RationalCoeff rat;
    rat.readfile(commBase, rhmc_param.rat_file());

   
    rootLogger.info("Initialize Gaugefield & Spinorfield");

    Gaugefield<PREC, true, HaloDepth,R18> gauge(commBase);
    Gaugefield<PREC, true, HaloDepth> gaugeLvl2(commBase);
    Gaugefield<PREC, true, HaloDepth,U3R14> gaugeNaik(commBase);
    Gaugefield<PREC, true, HaloDepth> force(commBase);
    Spinorfield<PREC, true, Even, HaloDepthSpin> SpinorIn(commBase);

    gauge.readconf_nersc("../test_conf/gauge12750");

    gauge.updateAll();

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    h_rand.make_rng_state(rhmc_param.seed());

    d_rand = h_rand;

    HisqSmearing<PREC, true, HaloDepth,R18> smearing(gauge,gaugeLvl2,gaugeNaik);
    smearing.SmearAll(chmp);
    AdvancedMultiShiftCG<PREC, 14> CG;


    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin,1> dslash(gaugeLvl2,gaugeNaik,0.0);
    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 14> dslash_multi(gaugeLvl2,gaugeNaik,0.0);

    HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18, true> ip_dot_f2_hisq(gauge,force,CG,dslash,dslash_multi,rhmc_param,rat,smearing);

    
    ip_dot_f2_hisq.TestForce(SpinorIn,force,d_rand);

    force.writeconf_nersc("../test_conf/force_imagmu_testrun");
    force.readconf_nersc("../test_conf/force_imagmu_testrun");

    Gaugefield<PREC,true,HaloDepth> force_reference(commBase);
    force_reference.readconf_nersc("../test_conf/force_imagmu_reference");

    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    LatticeContainer<true, int> dummy(commBase);
    dummy.adjustSize(elems);

    
    dummy.template iterateOverBulk<All,HaloDepth>(count_faulty_links<PREC,true,HaloDepth,R18>(force,force_reference));

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

