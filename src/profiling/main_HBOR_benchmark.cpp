/* 
 * main_HBOR_benchmark.cu
 *
 * D. Clarke
 *
 * To benchmark the pure gauge updates.
 *
 */

#include "../modules/gauge_updates/PureGaugeUpdates.h"
#include "../gauge/GaugeAction.h"

#include <unistd.h>

#define PREC double
#define MY_BLOCKSIZE 256
#define MAKE_TEST_CONF 1
#define READ_TEST_CONF 0

template<class floatT, size_t HaloDepth>
bool test_function(Gaugefield<floatT, false, HaloDepth> &gauge, Gaugefield<floatT, false, HaloDepth> &refgauge,
                   floatT tol)
{
    size_t totalchecks=0;
    size_t failedchecks=0;
    typedef GIndexer<All, HaloDepth> GInd;
    bool lpassed=true;

    for (int ix=0; ix<(int)GInd::getLatData().lx; ix++)
    for (int iy=0; iy<(int)GInd::getLatData().ly; iy++)
    for (int iz=0; iz<(int)GInd::getLatData().lz; iz++)
    for (int it=0; it<(int)GInd::getLatData().lt; it++) {
        for (int mu=0; mu<4; mu++) {
            totalchecks++;
            gSiteMu siteMu=GInd::getSiteMu(ix,iy,iz,it,mu);
            if( !compareGSU3<floatT>(gauge.getAccessor().getLink(siteMu),refgauge.getAccessor().getLink(siteMu),tol) )
                failedchecks++;
        }
    }
    floatT failedfrac=1.0*failedchecks/totalchecks;
    rootLogger.info("test_function: " ,  failedfrac*100 ,  "% of tests failed with tolerance " ,  tol);
    if(failedfrac>0.01) lpassed=false;
    return lpassed;
}

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
    MicroTimer timer;

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
        bool lpassed=test_function(hostgauge,refgauge,1e-13);
        if (lpassed) {
            rootLogger.info("Comparison test " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
        } else {
            rootLogger.error("Comparison test failed!");
        }
    )
    return 0;
}

