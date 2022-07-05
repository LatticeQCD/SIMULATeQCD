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
    grnd_state<false> d_rand;
    d_rand.make_rng_state(1337);

    const int NStacks = 6;
    TaylorMeasurement<PREC, true, All, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(commBase, param, param.valence_masses[0], gauge, d_rand);
    taylor_measurement.insertOperator(1);
    taylor_measurement.insertOperator(3);
    taylor_measurement.insertOperator(10);

    taylor_measurement.computeOperators();

    std::vector<DerivativeOperatorMeasurement> results;
    taylor_measurement.collectResults(results);
    for (DerivativeOperatorMeasurement &meas : results) {
        printf(("ID: %d, Measurement %." + std::to_string(param.prec_out()) + "e").c_str(), meas.operatorId, meas.measurement);
    }
#endif
    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr

    return 0;
}
