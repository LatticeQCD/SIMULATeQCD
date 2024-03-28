/*
 * main_HBOR_benchmark.cpp
 *
 * D. Clarke
 *
 * To benchmark the pure gauge updates.
 *
 */

#include "../modules/gauge_updates/pureGaugeUpdates.h"
#include "../gauge/gaugeAction.h"
#include "../testing/testing.h"

#define PREC double
#define MAKE_TEST_CONF 1
#define READ_TEST_CONF 0


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 1;
    const int nsweep = 50;              /// Number of sweeps
    LatticeParameters param;
    const int LatDim[] = {68, 272, 68, 68};
    const int NodeDim[] = {1, 4, 1, 1};
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);
    const PREC beta0 = 5.8;

    /// Initialize communication base, indexer, timer.
    rootLogger.info("Initialization");
    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param, commBase);
    typedef GIndexer<All, HaloDepth> GInd;
    StopWatch<true> timer;

    /// Initialize objects related to the gauge fields.
    Gaugefield<PREC, true, HaloDepth> gauge(commBase);
    Gaugefield<PREC, false, HaloDepth> hostgauge(commBase);
    GaugeAction<PREC, true, HaloDepth> gAction(gauge);
    GaugeUpdate<PREC, true, HaloDepth> gUpdate(gauge);

    /// You only need a reference configuration if you're comparing on multiGPU.
    IF(READ_TEST_CONF) (
        Gaugefield<PREC, false, HaloDepth> refgauge(commBase);
    )

    /// Initialize RNG.
    rootLogger.info("Initialize RNG and make random lattice...");
    int seed = 0;
    grnd_state<false> host_state;
    grnd_state<true> dev_state;
    host_state.make_rng_state(seed);
    dev_state = host_state;
    rootLogger.info("Random seed is: " ,  seed);

    /// Create random lattice, then hit it with some updates.
    rootLogger.info("Performing the benchmark... ");
    timer.start();
    gauge.random(dev_state.state);
    gauge.updateAll();
    for (int i=0; i<nsweep; i++) {
        gUpdate.updateHB(dev_state.state,beta0);
        gUpdate.updateOR();
        gUpdate.updateOR();
        gUpdate.updateOR();
        gUpdate.updateOR();
        rootLogger.info("Sweep " ,  i);
    }
    timer.stop();
    rootLogger.info("Time for updates: " ,  timer);
    hostgauge=gauge;

    /// Compare against test configuration mode.
    IF(READ_TEST_CONF) (
        rootLogger.info("Read from file...");
        refgauge.readconf_nersc("config_HBOR_benchmark");
        rootLogger.info("Running comparison test...");
        bool pass=compare_fields<PREC,HaloDepth,false,R18>(hostgauge,refgauge,1e-13)
        if (pass) {
            rootLogger.info("Comparison test " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
        } else {
            rootLogger.error("Comparison test failed!");
            return -1;
        }
    )
    return 0;
}

