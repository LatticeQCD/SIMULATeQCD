/* 
 * main_hisqForce.cpp                                                               
 * 
 * This program tests the HISQ fermion force and has to yield the same result as the gaction_test_hisqforce.cpp
 * in the BielefeldGPUCode.
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/HISQ/hisqForce.h"
#include <fstream>
#include <iostream>

#define PREC float
#define MY_BLOCKSIZE 256
#define USE_GPU true

template <class floatT, size_t HaloDepth, CompressionType comp>
struct compare_smearing {
    gaugeAccessor<floatT,comp> gL;
    gaugeAccessor<floatT,comp> gR;
    compare_smearing(Gaugefield<floatT,false,HaloDepth, comp> &GaugeL, Gaugefield<floatT, false, HaloDepth, comp> &GaugeR)
        : gL(GaugeL.getAccessor()), gR(GaugeR.getAccessor()) {}

    void write_ascii(std::ofstream &Out) {
        for (unsigned int x = 0; x < GIndexer<All,HaloDepth>::getLatData().lx; x++)
        for (unsigned int y = 0; y < GIndexer<All,HaloDepth>::getLatData().ly; y++)
  	    for (unsigned int z = 0; z < GIndexer<All,HaloDepth>::getLatData().lz; z++)
  	    for (unsigned int t = 0; t < GIndexer<All,HaloDepth>::getLatData().lt; t++) {
  	        for (int mu = 0; mu < 4; mu++) {
          	    gSiteMu siteMu=GIndexer<All,HaloDepth>::getSiteMu(x,y,z,t,mu);
          	    GSU3<floatT> tmp = gL.getLink(siteMu);
          	    Out << tmp.getLink00()<<tmp.getLink01()<<tmp.getLink02() << std::endl;
          	    Out << tmp.getLink10()<<tmp.getLink11()<<tmp.getLink12() << std::endl;
          	    Out << tmp.getLink20()<<tmp.getLink21()<<tmp.getLink22() << std::endl;
          	    Out << std::endl;
  	        }
  	    }
    }

    __host__ __device__ int operator() (gSite site) {

        floatT sum = 0.0;
        for (int mu = 0; mu < 4; mu++) {

            gSiteMu siteMu=GIndexer<All,HaloDepth>::getSiteMu(site,mu);
            GSU3<floatT> diff = gL.getLink(siteMu) - gR.getLink(siteMu);
            floatT norm = 0.0;

            for (int i = 0; i < 3; i++) {
            	for (int j = 0; j < 3; j++) {
    	            norm += abs2(diff(i,j));
    	        }
            }
            sum += sqrt(norm);
            sum /= 16.0;
        }
        sum /= 4.0;

        #ifdef __GPU_ARCH__
        return (sum < 8e-5 ? 0 : 1);
        #else
        if (sum > 8e-5) {
            rootLogger.info("Found significant difference at " ,  site.getStr() ,  " the difference is " ,  sum ,  "\n");
        }
        return (sum < 8e-5 ? 0 : 1);
        #endif
    }
};


template <class floatT, size_t HaloDepth, CompressionType comp>
bool checkfields(Gaugefield<floatT,false,HaloDepth, comp> &GaugeL, Gaugefield<floatT, false, HaloDepth, comp> &GaugeR) {
    LatticeContainer<false,int> redBase(GaugeL.getComm());
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    
    redBase.adjustSize(elems);
    
    redBase.template iterateOverBulk<All,HaloDepth>(compare_smearing<floatT, HaloDepth, comp>(GaugeL,GaugeR));

    int faults = 0;
    redBase.reduce(faults,elems);

    rootLogger.info(faults ,  " faults detected");

    if (faults > 0) {
        return false;
    } else {
        return true;
    }
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    MicroTimer timer;
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
    rat.readfile(commBase, rhmc_param.rat_file(), argc, argv);

    Gaugefield<PREC, false,HaloDepth> force_host(commBase);
    rootLogger.info("Initialize Gaugefield & Spinorfield");

    Gaugefield<PREC, true, HaloDepth,R18> gauge(commBase);
    Gaugefield<PREC, false, HaloDepth> gauge_host(commBase);
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
    smearing.SmearAll();
    gauge_host=gaugeLvl2;
    gauge_host.updateAll();
    AdvancedMultiShiftCG<PREC, 14> CG;

    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin,1> dslash(gaugeLvl2,gaugeNaik,0.0);
    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 14> dslash_multi(gaugeLvl2,gaugeNaik,0.0);

    HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18, true> ip_dot_f2_hisq(gauge,force,CG,dslash,dslash_multi,rhmc_param,rat,smearing);

    timer.start();
    gpuProfilerStart();
    ip_dot_f2_hisq.TestForce(SpinorIn,force,d_rand);
    gpuProfilerStop();
    timer.stop();

    force_host=force;

    GSU3<PREC> test1 = force_host.getAccessor().getLink(GInd::getSiteMu(0,0,0,3,3));

    rootLogger.info("Time: " ,  timer);
    rootLogger.info("Force parallelGpu:");
    rootLogger.info(test1.getLink00(), test1.getLink01(), test1.getLink02());
    rootLogger.info(test1.getLink10(), test1.getLink11(), test1.getLink12());
    rootLogger.info(test1.getLink20(), test1.getLink21(), test1.getLink22());
    
    Gaugefield<PREC,false,HaloDepth> force_BIGPU(commBase);

    force_BIGPU.readconf_nersc("../test_conf/hisqF_BIGPU.nersc");
    compare_smearing<PREC, HaloDepth,R18> Comp(force_host,force_BIGPU);
    std::ofstream AsciiOutput;
    AsciiOutput.open("../test_conf/HisqForceCheck.txt",std::ofstream::out);
    Comp.write_ascii(AsciiOutput);

    force.writeconf_nersc("../test_conf/hisqF_PGPU.nersc");

    force.readconf_nersc("../test_conf/hisqF_PGPU.nersc");

    force_host=force;

    bool pass = checkfields<PREC,HaloDepth,R18>(force_host,force_BIGPU);
    if (pass) {
        rootLogger.info(CoutColors::green ,  "Force is correct" ,  CoutColors::reset);
    } else {
        rootLogger.info(CoutColors::red ,  "Force is wrong" ,  CoutColors::reset);
        rootLogger.info(CoutColors::red ,  "Please make sure that you are using the same seed, precision, and rational approx as gaction_test_hisqforce.cpp in the BielefeldGPUCode" ,  CoutColors::reset);
    }
    rootLogger.info("For a more precise check, compare Ascii Output test_conf/HisqForceCheck.txt with test_conf/BIGPU_HisqForce.txt");
       
    return 0;
}

