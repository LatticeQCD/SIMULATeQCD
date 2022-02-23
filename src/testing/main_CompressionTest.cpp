/* 
 * main_CompressionTest.cpp                                                               
 * 
 * Lukas Mazur, 9 Oct 2017
 *
 * Test and compare the various kinds of compression.
 *
 */

#include "../SIMULATeQCD.h"
#include "testing.h"

#define PREC double

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/CompressionTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 2;
    bool forceHalos = true;
    rootLogger.info("Initialize Lattice");
    initIndexer(HaloDepth,param,commBase,forceHalos);

    rootLogger.info("Initialize Gaugefields");

    StopWatch<true> timer;

    typedef GIndexer<All,HaloDepth> GInd;

    //// R18
    Gaugefield<PREC,true,HaloDepth, R18> gaugeR18( commBase);
    gaugeR18.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gaugeR18.updateAll();
    GaugeAction<PREC, true ,HaloDepth,R18> gActionR18(gaugeR18);

    //// R12
    Gaugefield<PREC,true,HaloDepth, R12> gaugeR12( commBase);
    gaugeR12.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gaugeR12.updateAll();
    GaugeAction<PREC, true ,HaloDepth,R12> gActionR12(gaugeR12);

    //// U3R14
    Gaugefield<PREC, true,HaloDepth, U3R14> gaugeU3R14( commBase);
    gaugeU3R14.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gaugeU3R14.updateAll();
    GaugeAction<PREC, true ,HaloDepth,U3R14> gActionU3R14(gaugeU3R14);

    //// R14
    Gaugefield<PREC, true,HaloDepth, R14> gaugeR14( commBase);
    gaugeR14.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gaugeR14.updateAll();
    GaugeAction<PREC, true ,HaloDepth,R14> gActionR14(gaugeR14);

    //// Compare R18 and R12
    rootLogger.info("");
    rootLogger.info("Compare R18 and R12:");
    timer.start();
    PREC plaqR18 = gActionR18.plaquette();
    timer.stop();
    rootLogger.info("R18 plaq: " ,  plaqR18 ,  ", " ,  timer);
    timer.reset();

    timer.start();
    PREC plaqR12 = gActionR12.plaquette();
    timer.stop();
    rootLogger.info("R12 plaq: " ,  plaqR12 ,  ", " ,  timer);
    timer.reset();

    compare_relative(plaqR12, plaqR18, 1e-8, 1e-7, "R12 test");

    //// Compare R18 and U3R14
    rootLogger.info("");
    rootLogger.info("Compare R18 and U3R14:");
    timer.start();
    gaugeR18 = gaugeR18*GPUcomplex<PREC>(0.7,sqrt(1-0.7*0.7));
    timer.stop();
    rootLogger.info("Multiply phase in R18. " ,  timer);
    timer.reset();

    timer.start();
    gaugeU3R14 = gaugeU3R14*GPUcomplex<PREC>(0.7,sqrt(1-0.7*0.7));
    timer.stop();
    rootLogger.info("Multiply phase in U3R14. " ,  timer);
    timer.reset();

    timer.start();
    plaqR18 = gActionR18.plaquette();
    timer.stop();
    rootLogger.info("R18 plaq: " ,  plaqR18 ,  ", " ,  timer);
    timer.reset();

    timer.start();
    PREC plaqU3R14 = gActionU3R14.plaquette();
    timer.stop();
    rootLogger.info("U3R14 plaq: " ,  plaqU3R14 ,  ", " ,  timer);
    timer.reset();

    compare_relative(plaqU3R14, plaqR18, 1e-6, 1e-6, "U3R14 test");

    //// Compare R18 and R14
    rootLogger.info("");
    rootLogger.info("Compare R18 and R14:");
    gaugeR14 = gaugeR14*GPUcomplex<PREC>(0.7,sqrt(1-0.7*0.7));

    timer.start();
    gaugeR18 = gaugeR18*GPUcomplex<PREC>(2,3);
    timer.stop();
    rootLogger.info("Multiply phase in R18. " ,  timer);
    timer.reset();

    timer.start();
    gaugeR14 = gaugeR14*GPUcomplex<PREC>(2,3);
    timer.stop();
    rootLogger.info("Multiply phase in R14. " ,  timer);
    timer.reset();

    timer.start();
    plaqR18 = gActionR18.plaquette();
    timer.stop();
    rootLogger.info("R18 plaq: " ,  plaqR18 ,  ", " ,  timer);
    timer.reset();

    timer.start();
    PREC plaqR14 = gActionR14.plaquette();
    timer.stop();
    rootLogger.info("R14 plaq: " ,  plaqR14 ,  ", " ,  timer);
    timer.reset();

    compare_relative(plaqR14, plaqR18, 1e-6, 1e-2, "R14 test");

    return 0;
}
