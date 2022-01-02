/* 
 * main_HisqSmearingImagmuTest.cu
 *
 * J. Goswami 
 *
 * Quick single-GPU test comparing the output of the imaginary mu smearing against the old results from the ParallelGPUCode. 
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"

#define PREC double
#define MY_BLOCKSIZE 256
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
    Gaugefield<PREC, true, HaloDepth> gauge_Lvl1(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_Lvl1_host(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_smeared_bielefeldgpu(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_u3(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_u3_host(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_Lv2(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_Lv2_host(commBase);
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
    gauge_smeared_bielefeldgpu.readconf_nersc("../test_conf/smearing_link_lv2_110_nersc");
    gauge_in.updateAll();

    smearing.SmearLvl1(gauge_Lvl1); //These two are only used in force calculations
    smearing.ProjectU3(gauge_Lvl1,gauge_u3);

    timer.start();
    smearing.SmearAll(chmp);
    timer.stop();
    rootLogger.info("Time for full smearing: ",timer);
    gauge_Lv2.writeconf_nersc("../test_conf/pgpu_lv2_smearing.nersc");

    gauge_Lvl1_host = gauge_Lvl1;
    gauge_u3_host = gauge_u3;
    gauge_Lv2_host = gauge_Lv2;

    gaugeAccessor<PREC> gaugeAcc3 = gauge_smeared_bielefeldgpu.getAccessor();

    typedef GIndexer<All,HaloDepth> GInd;
    gSite site1 = GInd::getSite(0,0,1,1);

    GSU3<PREC> test2;

    imgchmp.cREAL = cos(chmp);
    imgchmp.cIMAG = sin(chmp);

    for (int muj=0; muj<4; muj++) {
        if (muj == 3) {
            test2 = gaugeAcc3.getLink(GInd::getSiteMu(site1, muj)) * imgchmp;
        } else {
            test2 = gaugeAcc3.getLink(GInd::getSiteMu(site1, muj));
        }
    }

    GSU3<PREC> temp = gauge_Lvl1_host.getAccessor().getLink(GInd::getSiteMu(site1, 3));
    temp.su3unitarize();
    rootLogger.info("\n  level1 smeared link from parallelgpu:");
    rootLogger.info(temp.getLink00(),temp.getLink01(),temp.getLink02(),temp.getLink10());

    GSU3<PREC> temp2 = gauge_u3_host.getAccessor().getLink(GInd::getSiteMu(site1,3));
    GSU3<PREC> temp2dagger = gauge_u3_host.getAccessor().getLinkDagger(GInd::getSiteMu(site1,3));

    GSU3<PREC> unitarity_test = temp2 * temp2dagger;

    rootLogger.info("\n u3 smeared link from parallelgpu:");
    rootLogger.info(temp2.getLink00(), temp2.getLink01(),temp2.getLink02() , temp2.getLink10());

    rootLogger.info("\n unitarity test:U^{dagger}U=I");
    rootLogger.info(unitarity_test.getLink00(),unitarity_test.getLink11(),unitarity_test.getLink22());

    GSU3<PREC> temp3 = gauge_Lv2_host.getAccessor().getLink(GInd::getSiteMu(site1, 3));


    /// Simple test comparing the output of a link against the ParallelGPUCode.
    rootLogger.info( "\n Lv2 smeared link from ParallelGPUCode with mu_f = 0.4 i");
    rootLogger.info(temp3.getLink00(),temp3.getLink01(),temp3.getLink02());
    rootLogger.info(temp3.getLink10(),temp3.getLink11(),temp3.getLink12());
    rootLogger.info(temp3.getLink20(),temp3.getLink21(),temp3.getLink22());

    GSU3<PREC> test = GSU3<PREC>( GCOMPLEX(PREC)(0.732449,-0.0011968), GCOMPLEX(PREC)(-0.369535,-0.0837624), GCOMPLEX(PREC)(0.285353,0.654512) ,
                                  GCOMPLEX(PREC)(0.502689,0.694297)  , GCOMPLEX(PREC)(-0.250711,0.191546)  , GCOMPLEX(PREC)(0.330026,-0.659482),
                                  GCOMPLEX(PREC)(-0.283782,0.0796874), GCOMPLEX(PREC)(-0.790194,0.734873)  , GCOMPLEX(PREC)(-0.311241,0.150933) );

    GSU3<PREC> diff = test - temp3;
    PREC sum = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sum += abs2(diff(i,j));
        }
    }
    sum = sqrt(sum)/16.0;
    if (sum < 1e-6) {
        rootLogger.info(CoutColors::green, "Test passed!", CoutColors::reset);
    } else {
        rootLogger.error("sum:", sum); 
        throw std::runtime_error(stdLogger.fatal("Test failed!"));
    }

    return 0;
}