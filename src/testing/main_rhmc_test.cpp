/* 
 * main_rhmc_test.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/rhmc/rhmc.h"
#include "../modules/observables/PolyakovLoop.h"


template<class floatT, int HaloDepth>
bool reverse_test(CommunicationBase &commBase, RhmcParameters param, RationalCoeff rat){

    initIndexer(4,param, commBase);

    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase);

    grnd_state<true> d_rand;
    initialize_rng(param.seed(), d_rand);

    gauge.one();

    rootLogger.info("constructed gauge field");

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);

    rootLogger.info("constructed the HMC");

    HMC.init_ratapprox();

    rootLogger.info("Initialized the Rational Approximation");

    int acc = 0;
    double acceptance = 0.0;
    PolyakovLoop<floatT, true, HaloDepth, R18> ploop(gauge);

    for (int i = 1; i <= 2; ++i) {
        acc += HMC.update(true, true);
        acceptance = double(acc)/double(i);
    }

    bool ret = true;

    if (acceptance < 0.9999) {
        ret = false;
    }

    return ret;
};

template<class floatT, int HaloDepth>
bool full_test(CommunicationBase &commBase, RhmcParameters param, RationalCoeff rat)
{
    initIndexer(4,param, commBase);

    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase);

    grnd_state<true> d_rand;
    initialize_rng(param.seed(), d_rand);

    gauge.one();
    gauge.updateAll();

    rootLogger.info("constructed gauge field");

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);

    rootLogger.info("constructed the HMC");

    HMC.init_ratapprox();

    rootLogger.info("Initialized the Rational Approximation");

    int acc = 0;
    floatT acceptance = 0.0;
    PolyakovLoop<floatT, true, HaloDepth, R18> ploop(gauge);

    for (int i = 1; i <= 100; ++i) {
        acc += HMC.update();
        acceptance = floatT(acc)/floatT(i);

        rootLogger.info("|Ploop|(" ,  i , ")= " ,  abs(ploop.getPolyakovLoop()));
    }

    rootLogger.info("Run has ended. acceptance = " ,  acceptance);

    bool ret = false;

    if (acceptance > 0.7) {
        ret=true;
    }

    return ret;
};


template<class floatT, int HaloDepth>
bool no_rng_test(CommunicationBase &commBase, RhmcParameters param, RationalCoeff rat) {
    initIndexer(4,param, commBase);

    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase);
    GaugeAction<floatT, true, HaloDepth, R18> gaugeaction(gauge);

    grnd_state<true> d_rand;
    initialize_rng(param.seed(), d_rand);

    gauge.one();

    rootLogger.info("constructed gauge field");

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);

    rootLogger.info("constructed the HMC");

    HMC.init_ratapprox();

    rootLogger.info("Initialized the Rational Approximation");

    int acc = HMC.update_test();

    bool ret = false;

    if (acc == 1) {
        ret=true;
    }

    return ret;
};

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    RhmcParameters param;

    param.readfile(commBase, "../parameter/tests/rhmcTest.param", argc, argv);

    const int HaloDepth = 2;

    RationalCoeff rat;

    rat.readfile(commBase, param.rat_file(), argc, argv);

    commBase.init(param.nodeDim());

    initIndexer(HaloDepth,param, commBase);

    rootLogger.info("STARTING RHMC TESTS:\n");
    rootLogger.info("This will take some minutes. Go grab a coffee/tea.");

    rootLogger.info("STARTING REVERSIBILITY TEST:");

    typedef float floatT;

    bool revers = reverse_test<floatT, HaloDepth>(commBase, param, rat);


    if (revers)
        rootLogger.info("REVERSIBILITY TEST: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
    else
        rootLogger.error("REVERSIBILITY TEST: " ,  CoutColors::red ,  "failed" ,  CoutColors::reset);


    rootLogger.info("STARTING FULL UPDATE TEST:");
    rootLogger.info("Now, there should be some dynamics");


    bool full = full_test<floatT, HaloDepth>(commBase, param, rat);

    if (full)
        rootLogger.info("FULL UPDATE TEST: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
    else
        rootLogger.error("FULL UPDATE TEST: " ,  CoutColors::red ,  "failed" ,  CoutColors::reset);


    if (revers /*&& no_rng*/ && full)  {
        rootLogger.info(CoutColors::green ,  "ALL TESTS PASSED" ,  CoutColors::reset);
        rootLogger.warn("This only indicates that force matches action.\n");
        rootLogger.warn("Check Observables to find out if action is correct!");
    }
    else
        rootLogger.error(CoutColors::red ,  "AT LEAST ONE TEST FAILED" ,  CoutColors::reset);

    return 0;
}
