/*
 * main_HisqSmearingTest.cpp
 *
 */

#include "../simulateqcd.h"
#include "../modules/hisq/hisqSmearing.h"
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
    Gaugefield<PREC, true, HaloDepth> gauge_Lvl1(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_Lvl1_host(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_smeared_reference(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_u3(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_u3_host(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_Lv2(commBase);
    Gaugefield<PREC, false,HaloDepth> gauge_Lv2_host(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_naik(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_smeared_reference_device(commBase);
    timer.start();
    HisqSmearing<PREC, USE_GPU, HaloDepth,R18,R18,R18,R18> smearing(gauge_in, gauge_Lv2, gauge_naik);
    timer.stop();
    rootLogger.info("Time for initializing HisqSmearing: " ,  timer);
    timer.reset();

    rootLogger.info("Read configuration");
    gauge_in.readconf_nersc("../test_conf/gauge12750");

    gauge_smeared_reference.readconf_nersc("../test_conf/smearing_reference_conf");
    gauge_smeared_reference_device = gauge_smeared_reference;
    gauge_smeared_reference_device.su3latunitarize();
    gauge_in.updateAll();

    smearing.SmearLvl1(gauge_Lvl1); //These two are only used in force calculations
    smearing.ProjectU3(gauge_Lvl1,gauge_u3);

    timer.start();
    smearing.SmearAll();
    timer.stop();
    rootLogger.info("Time for full smearing: " ,  timer);

    gauge_Lvl1_host = gauge_Lvl1;
    gauge_u3_host = gauge_u3;
    gauge_Lv2_host = gauge_Lv2;

    SU3Accessor<PREC> gaugeAcc3 = gauge_smeared_reference.getAccessor();

    typedef GIndexer<All,HaloDepth> GInd;
    gSite site1 = GInd::getSite(0,0,1,1);
    SU3<PREC> test2 = gaugeAcc3.getLink(GInd::getSiteMu(site1, 3));

    rootLogger.info("\n" , "level2 smeared link from reference gaugefield:");
    rootLogger.info(test2.getLink00(), test2.getLink01(), test2.getLink02(), test2.getLink10());

    SU3<PREC> temp = gauge_Lvl1_host.getAccessor().getLink(GInd::getSiteMu(site1, 3));
    temp.su3unitarize();
    rootLogger.info("\n" ,  "level1 smeared link from this run:");
    rootLogger.info(temp.getLink00(), temp.getLink01(), temp.getLink02(), temp.getLink10());

    SU3<PREC> temp2 = gauge_u3_host.getAccessor().getLink(GInd::getSiteMu(site1,3));
    SU3<PREC> temp2dagger = gauge_u3_host.getAccessor().getLinkDagger(GInd::getSiteMu(site1,3));

    SU3<PREC> unitarity_test = temp2 * temp2dagger;

    rootLogger.info("\n" ,  "u3 smeared link from this run:");
    rootLogger.info(temp2.getLink00(), temp2.getLink01(), temp2.getLink02(), temp2.getLink10());

    rootLogger.info("\n" ,  "unitarity test:U^{dagger}U=I");
    rootLogger.info(unitarity_test.getLink00() ,  unitarity_test.getLink11() , unitarity_test.getLink22());

    SU3<PREC> temp3 = gauge_Lv2_host.getAccessor().getLink(GInd::getSiteMu(site1, 3));

    rootLogger.info("\n" ,  "Lv2 smeared link from this run:");
    temp3.su3unitarize();
    rootLogger.info(temp3.getLink00(), temp3.getLink01(), temp3.getLink02(), temp3.getLink10());

    gauge_Lv2.su3latunitarize();

    bool pass = compare_fields<PREC,HaloDepth,true,R18>(gauge_Lv2,gauge_smeared_reference_device);

    if (pass) {
        rootLogger.info(CoutColors::green, "Test passed!", CoutColors::reset);
    } else {
        rootLogger.error("Test failed!");
        return 1;
    }
    return 0;
}

