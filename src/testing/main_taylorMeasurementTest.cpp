/*
 * main_TaylorMeasurementTest.cpp
 *
 * This file is a test for the measurement of the
 * taylor coefficients of Z(mu)
 *
 */

#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"
#include "../modules/dslash/condensate.h"
#include "testing.h" // for comparing stuff

#define PREC double

// extend the functionallity of testing by a fail only check
void assert_check(bool condition, const std::string fail_text) {
    if (!condition) {
        // NOTE: the text has already been printed by check, just additionally throw an exception here to abort the program
        throw std::runtime_error(rootLogger.fatal(fail_text));
    }
}

int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    bool is_test = false;
    try {
        // test using predefined test setup
        param.readfile(commBase, "../parameter/tests/taylorMeasurementTest.param", argc, argv);
        is_test = true;
    }
    catch (std::runtime_error e) {
        // just try reading from the same directory (this makes it useful even when not in this fixed folder structure used as a test)
        rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
        param.readfile(commBase, "./TaylorMeasurement.param", argc, argv);
    }

    commBase.init(param.nodeDim());

    const int HaloDepth = 2; // >= 1 for multi gpu
    const int HaloDepthSpin = 4;
    initIndexer(HaloDepth, param, commBase);
    stdLogger.setVerbosity(INFO);

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);      /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    if (param.valence_masses.numberValues() == 0) {
        rootLogger.error("No valence masses specified, aborting");
        return 1;
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    rootLogger.info("Rng initialized with seed ", param.seed());


    if (is_test) {
        // run the test and compare the result to a known value
        rootLogger.info("Starting in Test Mode");

        PREC mass = (PREC)param.valence_masses[0];
        rootLogger.info("Using mass ", mass);

        // test with two NStacks values
        const int NStacks1 = 1;
        std::vector<DerivativeOperatorMeasurement> results1;
        {
            TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks1> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
            rootLogger.info("Class initialized");
            for (const auto& id : param.operator_ids.get()) {
                taylor_measurement.insertOperator(id);
            }
            taylor_measurement.computeOperators();
            rootLogger.info("Operators computed with NStacks=", NStacks1);
            taylor_measurement.collectResults(results1);
        }

        const int NStacks2 = 2;
        std::vector<DerivativeOperatorMeasurement> results2;
        {
            TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks2> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
            rootLogger.info("Class initialized");
            for (const auto& id : param.operator_ids.get()) {
                taylor_measurement.insertOperator(id);
            }
            taylor_measurement.computeOperators();
            rootLogger.info("Operators computed with NStacks=", NStacks2);
            taylor_measurement.collectResults(results2);
        }

        // results for NStacks1
        COMPLEX(double) reference_results1[16] = {
            COMPLEX(double)(0.0), // op 0
            COMPLEX(double)(0.0727592, -0.000190948), // op 1
            COMPLEX(double)(-0.602464, 0.0012473), // op 11
            COMPLEX(double)(0.000188563, -0.000214152), // op 12
            COMPLEX(double)(0.0243022, 0.000199526), // op 13
            COMPLEX(double)(0.000398233, 5.24191e-05), // op 14
            COMPLEX(double)(-0.0350572, 0.000933468), // op 15
            COMPLEX(double)(0.000109375, -1.63429e-05), // op 2
            COMPLEX(double)(-0.00040642, 0.000370748), // op 21
            COMPLEX(double)(0.233341, -0.000174791), // op 22
            COMPLEX(double)(-6.4219e-05, -9.00177e-07), // op 23
            COMPLEX(double)(0.130463, -0.000368956), // op 24
            COMPLEX(double)(-0.000579647, -0.000647879), // op 25
            COMPLEX(double)(0.236625, 3.62528e-06), // op 3
            COMPLEX(double)(0.000120989, -6.90832e-05), // op 4
            COMPLEX(double)(0.129838, 0.000275378), // op 5
        };
        // results for NStacks2 (these are different for some reason, probably related to the RNG)
        COMPLEX(double) reference_results2[16] = {
            COMPLEX(double)(0.0), // op 0
            COMPLEX(double)(0.0729049, 0.000308857), // op 1
            COMPLEX(double)(-0.603425, 9.80468e-05), // op 11
            COMPLEX(double)(-6.17861e-05, 0.000590835), // op 12
            COMPLEX(double)(0.024296, -5.7674e-05), // op 13
            COMPLEX(double)(-0.000119477, 0.00150001), // op 14
            COMPLEX(double)(-0.036802, -0.00110223), // op 15
            COMPLEX(double)(-0.000152372, -1.39467e-05), // op 2
            COMPLEX(double)(0.000267956, -0.000175591), // op 21
            COMPLEX(double)(0.2335, -0.000496516), // op 22
            COMPLEX(double)(-2.80293e-05, 8.53035e-05), // op 23
            COMPLEX(double)(0.130606, -0.000356297), // op 24
            COMPLEX(double)(0.00100031, 0.000593865), // op 25
            COMPLEX(double)(0.236812, -2.5006e-05), // op 3
            COMPLEX(double)(-3.74247e-05, 9.02047e-05), // op 4
            COMPLEX(double)(0.129435, 6.73965e-05), // op 5
        };
        assert_check(results1.size() == results2.size(), "Collected result sizes are not consistent");
        for (int i = 0; i < results1.size(); i++) {
            DerivativeOperatorMeasurement &meas1 = results1[i];
            DerivativeOperatorMeasurement &meas2 = results2[i];
            assert_check(meas1.operatorId == meas2.operatorId, "Operator IDs are not consistent");
            if (meas1.operatorId == 0) {
                continue; // the checks would fail here because of 0.0/0.0
            }
            // 3 sigma intervals check. Only if something is really off, we will know
            // Note, that since the seed is fixed this is not randomly true or false. If this fails, we know something went wrong!
            compare_relative(real(meas1.measurement), real(meas2.measurement), 100.0, 3.0 * sqrt(real(meas1.std)*real(meas1.std) + real(meas2.std)*real(meas2.std)), std::string("Consistency check between NStacks with operator ") + std::to_string(meas1.operatorId) + std::string(" (real)"));
            compare_relative(imag(meas1.measurement), imag(meas2.measurement), 100.0, 3.0 * sqrt(imag(meas1.std)*imag(meas1.std) + imag(meas2.std)*imag(meas2.std)), std::string("Consistency check between NStacks with operator ") + std::to_string(meas1.operatorId) + std::string(" (imag)"));
            // precise checks with reference
            compare_relative(real(reference_results1[i]), real(meas1.measurement), 1e-5, 0.1 * real(meas1.std), std::string("Check 1 with reference for operator ") + std::to_string(meas1.operatorId) + std::string(" (real)"));
            compare_relative(imag(reference_results1[i]), imag(meas1.measurement), 1e-5, 0.1 * imag(meas1.std), std::string("Check 1 with reference for operator ") + std::to_string(meas1.operatorId) + std::string(" (imag)"));
            compare_relative(real(reference_results2[i]), real(meas2.measurement), 1e-5, 0.1 * real(meas2.std), std::string("Check 2 with reference for operator ") + std::to_string(meas2.operatorId) + std::string(" (real)"));
            compare_relative(imag(reference_results2[i]), imag(meas2.measurement), 1e-5, 0.1 * imag(meas2.std), std::string("Check 2 with reference for operator ") + std::to_string(meas2.operatorId) + std::string(" (imag)"));
        }
    }
    else {
        // run as a standalone programm using the parameter file
        rootLogger.info("Starting in Standalone Mode");

        const int NStacks = 8; // NOTE: this only works after the blocksize fix

        for (double mass : param.valence_masses.get()) {
            rootLogger.info("Using mass ", mass);

            TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
            try {
                for (const auto& id : param.operator_ids.get()) {
                    taylor_measurement.insertOperator(id);
                }
            }
            catch (std::runtime_error e) {
                rootLogger.error(e.what());
                return 1;
            }
            rootLogger.info("Operators added");
            taylor_measurement.write_output_file_header();
            taylor_measurement.computeOperators();
            rootLogger.info("Operators computed");

            std::vector<DerivativeOperatorMeasurement> results;
            taylor_measurement.collectResults(results);
            rootLogger.info("Results collected");
            for (DerivativeOperatorMeasurement &meas : results) {
                rootLogger.info("ID: ", meas.operatorId, ", Measurement: ", meas.measurement, ", Error: ", meas.std);
            }
        }
    }

    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
