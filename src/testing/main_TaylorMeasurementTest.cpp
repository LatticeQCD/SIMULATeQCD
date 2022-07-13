/*
 * main_GeneralFunctorTest.cpp
 *
 * This file is a test for the measurement of the
 * taylor coefficients of Z(mu)
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/TaylorMeasurement.h"
//#include "testing.h" // for comparing stuff

#define PREC double

// TODO define test class


int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/TaylorMeasurementTest.param", argc, argv);

    commBase.init(param.nodeDim());

    const int HaloDepth = 0; // >= 1 for multi gpu
    initIndexer(HaloDepth, param, commBase);
    const int HaloDepthSpin = 0;
    initIndexer(HaloDepthSpin, param, commBase);
    stdLogger.setVerbosity(INFO);

    /// More initialization.
    StopWatch<true> timer;
    Gaugefield<PREC,true,HaloDepth>  gauge(commBase);      /// gauge field
    GaugeAction<PREC,true,HaloDepth> gauge_action(gauge);  /// gauge action

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc("../test_conf/l328f21b6285m0009875m0790a_019.995");
    gauge.updateAll();

#if 0
    rootLogger.info("Measure Plaquette");
    timer.start();

    PREC plaq = gauge_action.plaquette();

    rootLogger.info("Plaquette ", plaq);
#else
    if (param.valence_masses.numberValues() == 0) {
        rootLogger.error("No valence masses specified, aborting");
        return 1;
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    PREC mass = (PREC)param.valence_masses[0];

    rootLogger.info("Rng initialized");

    const int NStacks = 1;
    TaylorMeasurement<PREC, true, All, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(commBase, param, mass, gauge, d_rand);
    rootLogger.info("Class initialized");
    taylor_measurement.insertOperator(1);
    //taylor_measurement.insertOperator(3);
    taylor_measurement.insertOperator(11);
    //taylor_measurement.insertOperator(13);
    //taylor_measurement.insertOperator(23);
    rootLogger.info("Operators added");

    taylor_measurement.computeOperators();
    rootLogger.info("Operators computed");

    std::vector<DerivativeOperatorMeasurement> results;
    taylor_measurement.collectResults(results);
    rootLogger.info("Results collected");
    for (DerivativeOperatorMeasurement &meas : results) {
        rootLogger.info("ID: ", meas.operatorId, ", Measurement: ", meas.measurement);
    }
#endif
    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
