/* 
 * main_pure_gauge_hmc_test.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/rhmc/pure_gauge_hmc.h"
#include "../modules/observables/PolyakovLoop.h"
#include <chrono>


template<int HaloDepth>
bool no_rng_test(CommunicationBase &commBase, RhmcParameters param){

    Gaugefield<double, true, HaloDepth> gauge(commBase);
    GaugeAction<double, true, HaloDepth> gaugeaction(gauge);
    grnd_state<true> d_rand;
    
    gauge.one();

    rootLogger.info() << "constructed gauge field";

    pure_gauge_hmc<double, All, HaloDepth> HMC(param, gauge, d_rand.state);

    rootLogger.info() << "constructed hmc";

    int acc =HMC.update_test();

    bool ret = false;

    if (acc == 1) {
        ret=true;
    }

    return ret;
};


template<int HaloDepth>
bool reverse_test(CommunicationBase &commBase, RhmcParameters param){

    Gaugefield<double, false, HaloDepth, R18> h_gauge(commBase);
    Gaugefield<double, true, HaloDepth, R18> gauge(commBase);
    GaugeAction<double, true, HaloDepth, R18> gaugeaction(gauge);

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    h_rand.make_rng_state(param.seed());

    d_rand = h_rand;
    gauge.random(d_rand.state);

    rootLogger.info() << "constructed gauge field";

    pure_gauge_hmc<double, All, HaloDepth> HMC(param, gauge, d_rand.state);

    rootLogger.info() << "constructed hmc";

    int acc = 0;
    double acceptance = 0.0;
    PolyakovLoop<double, true, HaloDepth> ploop(gauge);

    for (int i = 1; i <= 10; ++i) {
        acc += HMC.update(true, true);
        acceptance = double(acc)/double(i);
    }

    bool ret = true;

    if (acceptance < 0.9999) {
        ret = false;
    }

    return ret;
};


template<int HaloDepth>
bool full_test(CommunicationBase &commBase, RhmcParameters param) {

    const CompressionType COMP = R18;

    Gaugefield<double, true, HaloDepth, COMP> gauge(commBase);
    GaugeAction<double, true, HaloDepth, COMP> gaugeaction(gauge);
    
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    h_rand.make_rng_state(param.seed());

    d_rand = h_rand;
    rootLogger.info() << "after the fill with rand";
    gauge.random(d_rand.state);
    gauge.updateAll();

    rootLogger.info() << "constructed gauge field";

    pure_gauge_hmc<double, All, HaloDepth, COMP> HMC(param, gauge, d_rand.state);

    rootLogger.info() << "constructed hmc";

    int acc = 0;
    double acceptance = 0.0;
    PolyakovLoop<double, true, HaloDepth> ploop(gauge);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 1; i <= 100; ++i)  {
        acc += HMC.update();
        acceptance = double(acc)/double(i);

        rootLogger.info() << "|Ploop|= " << abs(ploop.getPolyakovLoop());
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;

    rootLogger.info() << "Elapsed time: " << elapsed.count();

    rootLogger.info() << "Run has ended. acceptance = " << acceptance;

    bool ret = false;

    if (acceptance > 0.7 && acceptance <= 1.0){
        ret=true;
    }

    return ret;
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    RhmcParameters param;

    param.readfile(commBase, "../parameter/tests/pure_gauge_hmc.param", argc, argv);

    const int HaloDepth = 2;
    
    commBase.init(param.nodeDim());

    initIndexer(HaloDepth,param, commBase);

    rootLogger.info() << "STARTING PURE GAUGE HMC TESTS:\n";
    rootLogger.info() << "This will take some minutes. Go grab a coffee/tea.";

    rootLogger.info() << "STARTING REVERSIBILITY TEST:";

    bool revers = reverse_test<HaloDepth>(commBase, param);

    if (revers)
        rootLogger.info() << "REVERSIBILITY TEST: passed";
    else
        rootLogger.error() << "REVERSIBILITY TEST: failed";

    rootLogger.info() << "STARTING NO_RNG TEST:";
    rootLogger.info() << "There should be no dynamics!";

    bool no_rng = no_rng_test<HaloDepth>(commBase, param);

    if (no_rng)
        rootLogger.info() << "UPDATE WITHOUT RNG: passed";
    else
        rootLogger.error() << "UPDATE WITHOUT RNG: failed";

    rootLogger.info() << "STARTING FULL UPDATE TEST:";
    rootLogger.info() << "Now, there should be some dynamics"; 

    bool full = full_test<HaloDepth>(commBase, param);

    if (full)
        rootLogger.info() << "FULL UPDATE TEST: passed";
    else
        rootLogger.error() << "FULL UPDATE TEST: failed";

    int ret = 1;

    if (revers && no_rng && full)  {
        ret = 0;
        rootLogger.info() << "ALL TESTS PASSED";
        rootLogger.warn() << "This only indicates that force matches action.\n";
        rootLogger.warn() << "Check Observables to find out if action is correct!";
    }
    else
        rootLogger.error() << "AT LEAST ONE TEST FAILED";

    return ret;
}
