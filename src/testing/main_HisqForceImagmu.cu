//
// Created by Jishnu on 2020-03-30.
//
#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/HISQ/hisqForce.h"
#include <fstream>
#include <iostream>

#define PREC float
#define MY_BLOCKSIZE 256
#define USE_GPU true

// This programm tests the HISQ fermion force and has to yield the same result as the gaction_test_hisqforce.cpp in the BielefeldGPUCode


template <class floatT, size_t HaloDepth, CompressionType comp>
struct compare_smearing {
    gaugeAccessor<floatT,comp> gL;
    gaugeAccessor<floatT,comp> gR;
    compare_smearing(Gaugefield<floatT,false,HaloDepth, comp> &GaugeL, Gaugefield<floatT, false, HaloDepth, comp> &GaugeR) : gL(GaugeL.getAccessor()), gR(GaugeR.getAccessor()) {}

    void write_ascii(std::ofstream &Out) {
        for (unsigned int x = 0; x < GIndexer<All,HaloDepth>::getLatData().lx; x++) {
            for (unsigned int y = 0; y < GIndexer<All,HaloDepth>::getLatData().ly; y++) {
                for (unsigned int z = 0; z < GIndexer<All,HaloDepth>::getLatData().lz; z++) {
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
#ifdef __CUDA_ARCH__

        return (sum < 8e-5 ? 0 : 1);
#else
        if (sum > 8e-5) {
            rootLogger.info("Found significant difference at " , site , " the difference is " ,sum);
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

    rootLogger.info( faults , " faults detected");

    if (faults > 0) {
        return false;
    }
    else {
        return true;
    }
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters rhmc_param;
    rhmc_param.readfile(commBase,"../parameter/run_force_test_imagmu.param", argc, argv);

    commBase.init(rhmc_param.nodeDim());


    const size_t HaloDepth = 0;
    const size_t HaloDepthSpin = 4;
    double chmp=rhmc_param.mu_f();

    rootLogger.info( "Initialize Lattice");
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
    //    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");

    //gauge.one();
    gauge.readconf_nersc("../test_conf/gauge12750");

    gauge.updateAll();

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    h_rand.make_rng_state(rhmc_param.seed());

    d_rand = h_rand;




    HisqSmearing<PREC, true, HaloDepth,R18> smearing(gauge,gaugeLvl2,gaugeNaik);
    smearing.SmearAll(chmp);
    gauge_host=gaugeLvl2;
    gauge_host.updateAll();
    AdvancedMultiShiftCG<PREC, 14> CG;


    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin,1> dslash(gaugeLvl2,gaugeNaik,0.0);
    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 14> dslash_multi(gaugeLvl2,gaugeNaik,0.0);

    HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18, true> ip_dot_f2_hisq(gauge,force,CG,dslash,dslash_multi,rhmc_param,rat,smearing);

    
    ip_dot_f2_hisq.TestForce(SpinorIn,force,d_rand);


    force_host=force;

    GSU3<double> test1 = force_host.getAccessor().getLink(GInd::getSiteMu(0,0,0,3,3));


//  Test force for mu_f=0.4
    rootLogger.info("Force parallelGpu with imaginary chemical potential:");
    rootLogger.info(test1.getLink00(), test1.getLink01(), test1.getLink02());
    rootLogger.info(test1.getLink10(), test1.getLink11(), test1.getLink12());
    rootLogger.info(test1.getLink20(), test1.getLink21(), test1.getLink22());
    
                
    GSU3 <double> temp = GSU3 <double> (GCOMPLEX(double)(0,5.84889e-05),GCOMPLEX(double)(4.45548e-05,0.000175047),
                 GCOMPLEX(double)(-0.000279591,-0.000377919),GCOMPLEX(double)(-4.45548e-05,0.000175047),GCOMPLEX(double)(0,6.00033e-05),
                GCOMPLEX(double)(-0.000340403,0.00014374),GCOMPLEX(double)(0.000279591,-0.000377919),GCOMPLEX(double)(0.000340403,0.00014374)
                ,GCOMPLEX(double)(0,-0.000118492));

   GSU3<PREC> diff = test1 - temp;
        PREC norm = 0.0;
        PREC sum = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              norm += abs2(diff(i,j));
          }
        }
        sum += sqrt(norm);
        sum /= 16.0;
        if (sum < 1e-6) {
             rootLogger.info(CoutColors::green ,  "Test passed!");
        }
        else {
            rootLogger.info(CoutColors::red ,  "Test failed! sum: " ,  sum);
        }
    return 0;
}

