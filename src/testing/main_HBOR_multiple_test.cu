/* 
 * main_HBOR_multiple_test.cu
 *
 * D. Clarke
 *
 * Test that overrelaxation and heatbath updates agree with high precision to the single processor results.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gauge_updates/PureGaugeUpdates.h"

#include <iostream>
#include <iomanip>
#include <unistd.h>

#define PREC double
#define MY_BLOCKSIZE 256

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

    /// You shouldn't adjust any of these parameters.
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 1;
    const int nsweep = 300;              /// Number of sweeps
    const int nskip  = 50;               /// Number of sweeps to skip before calculating plaquette average
    LatticeParameters param;
    const int LatDim[] = {8, 8, 8, 4};
    param.latDim.set(LatDim);
    const PREC beta0 = 3.36;
    bool lpassed = true;

    /// Initialize communication base, indexer, timer.
    rootLogger.info("Initialization");
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/HBOR_multiple_test.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param, commBase);
    typedef GIndexer<All, HaloDepth> GInd;
    StopWatch<true> timer;

    /// Initialize objects related to the gauge fields.
    Gaugefield<PREC, true, HaloDepth> gauge(commBase);
    Gaugefield<PREC, false, HaloDepth> hostgauge(commBase);
    Gaugefield<PREC, false, HaloDepth> refgauge(commBase);  /// reference gauge field from single processor code.
    GaugeAction<PREC, true, HaloDepth> gAction(gauge);
    GaugeUpdate<PREC, true, HaloDepth> gUpdate(gauge);

    /// Initialize RNG. Make sure the seed agrees with the single processor run.
    int seed = 0;
    grnd_state<false> host_state;
    grnd_state<true> dev_state;
    host_state.make_rng_state(seed);
    dev_state = host_state;
    rootLogger.info("Random seed is: " ,  seed);

    /// Read in gauge12750 and compare what the single GPU code read in with what the multiGPU code read in.
    gauge.readconf_nersc("../test_conf/gauge12750");
    gauge.updateAll();
    hostgauge=gauge;
    refgauge.readconf_nersc("config_HBORTestRD");
    refgauge.updateAll();
    if(test_function(hostgauge,refgauge,(PREC)1e-15)) {
        rootLogger.info("Direct link check (read) " ,  CoutColors::green ,  "passed." ,  CoutColors::reset);
    } else {
        rootLogger.error("Direct link check (read)  failed.");
        lpassed=false;
    }

    /// Reset the configuration and repeat test, this time after one HB sweep.
    gauge.readconf_nersc("../test_conf/gauge12750");
    gauge.updateAll();
    gUpdate.updateHB(dev_state.state, beta0); /// HB update the entire gauge field
    hostgauge=gauge;
    refgauge.readconf_nersc("config_HBORTestHB");
    refgauge.updateAll();
    if(test_function(hostgauge,refgauge,(PREC)1e-15)) {
        rootLogger.info("Direct link check (HB) " ,  CoutColors::green ,  "passed." ,  CoutColors::reset);
    } else {
        rootLogger.error("Direct link check (HB) failed." );
        lpassed=false;
    }

    /// Reset the configuration and repeat test, this time after one OR sweep.
    gauge.readconf_nersc("../test_conf/gauge12750");
    gauge.updateAll();
    gUpdate.updateOR(); /// OR update the entire gauge field
    hostgauge=gauge;
    refgauge.readconf_nersc("config_HBORTestOR");
    refgauge.updateAll();
    if(test_function(hostgauge,refgauge,(PREC)1e-15)) {
        rootLogger.info("Direct link check (OR) " ,  CoutColors::green ,  "passed." ,  CoutColors::reset);
    } else {
        rootLogger.error("Direct link check (OR) failed.");
        lpassed=false;
    }

    /// Reset the configuration.
    gauge.readconf_nersc("../test_conf/gauge12750");
    gauge.updateAll();

    /// Average plaquette test and timing.
    timer.start();
    PREC plaqav = 0.0;
    PREC plaq = gAction.plaquette();
    rootLogger.info("Starting plaquette: " ,  plaq);
    if (commBase.MyRank() == 0) std::cout << "\nisweep \t  plaquette\n";
    for (int isweep = 0; isweep < nsweep; isweep++) {
        gUpdate.updateHB(dev_state.state, beta0);  /// HB with RNG
        gUpdate.updateOR();
        gUpdate.updateOR();
        /// Print plaquette to screen.
        plaq = gAction.plaquette();
        if (commBase.MyRank() == 0) {
            std::cout << std::setw(7) << isweep << "  " << std::setw(13) << std::scientific << plaq << std::endl;
            if (isweep >= nskip) {
                plaqav += plaq;
            }
        }
    }
    if (commBase.MyRank() == 0 ) std::cout << std::endl;
    plaqav /= (nsweep - nskip);
    timer.stop();
    rootLogger.info("Time for updates: " ,  timer);
    rootLogger.info("Average plaquette: " ,  plaqav);

    /// With lattice 8^3x4, beta=3.36, and 300 1HB+2OR sweeps, skipping the first 50 sweeps,
    /// and making 5 jackknife blocks, the old Bielefeld GPU code yields:
    ///    plaquette = 0.23439(18)
    PREC bieplaqav = 0.23439;
    PREC bieplaqer = 0.00018;
    PREC lower = bieplaqav - 2 * bieplaqer;
    PREC upper = bieplaqav + 2 * bieplaqer;
    if (plaqav < lower || plaqav > upper) {
        rootLogger.error("Unlikely plaquette value! Compare with 0.23439(18)");
        rootLogger.error(plaqav ,  " " ,  lower ,  " " ,  upper);
        lpassed=false;
    } else {
        rootLogger.info("Plaquette test " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }

    /// Close up shop.
    if (lpassed) {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    } else {
        rootLogger.error("At least one test failed!");
        return -1;
    }

    return 0;
}

