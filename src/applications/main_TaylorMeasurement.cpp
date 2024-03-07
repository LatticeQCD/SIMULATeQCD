/*
 * main_TaylorMeasurement.cpp
 *
 * This file is the main file for the application for the measurement of the
 * taylor coefficients of Z(mu)
 *
 */
#include "../SIMULATeQCD.h"
#include "../modules/observables/TaylorMeasurement.h"
#include "../modules/dslash/condensate.h"
#include "../testing/testing.h" // for comparing stuff

#define PREC double
// main
int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    
    // try reading parameter file from the same directory 
    rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
    param.readfile(commBase, "/home/finn/_s0_stream_0/TaylorMeasurement.param", argc, argv);


    commBase.init(param.nodeDim());

    const int HaloDepth = 2; // >= 1 for multi gpu
    const int HaloDepthSpin = 4;
    // const int nvec= 304; // number of vectors to be read
    const int NStacks = 8; // NOTE: this only works for NStacks=8 after the blocksize fix
    typedef float floatT; // Define the precision here

    rootLogger.info("STARTING Taxlor Measurement:");

    if (sizeof(floatT)==4) {
      rootLogger.info("update done in single precision");
    } else if(sizeof(floatT)==8) {
      rootLogger.info("update done in double precision");
    } else {
      rootLogger.info("update done in unknown precision");
    }

    initIndexer(HaloDepth, param, commBase);
    stdLogger.setVerbosity(INFO);

    const int sizeh = param.latDim[0]*param.latDim[1]*param.latDim[2]*param.latDim[3]/2;
    rootLogger.info("sizeh: ", sizeh);

    // Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);      /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();
    
    // Read the Eigenvalues and Eigenvectors
    new_eigenpairs<PREC,true,Even,HaloDepthSpin,NStacks> eigenpairs(commBase);
    rootLogger.info("Read eigenvectors and eigenvalues from ", param.EigenvectorfileName());
    eigenpairs.readconf_evnersc(param.EigenvectorfileName());
    eigenpairs.updateAll();

    // if (param.valence_masses.numberValues() == 0) {
    //     rootLogger.error("No valence masses specified, a);
    //     return 1;
    // }

    // grnd_state<false> h_rand;
    // grnd_state<true> d_rand;
    // h_rand.make_rng_state(param.seed());
    // d_rand = h_rand;borting"

    // rootLogger.info("Rng initialized with seed ", param.seed());


    //     // run as a standalone programm using the parameter file
    //     rootLogger.info("Starting in Standalone Mode");

        

    //     for (double mass : param.valence_masses.get()) {
    //         rootLogger.info("Using mass ", mass);

    //         TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
    //         try {
    //             for (const auto& id : param.operator_ids.get()) {
    //                 taylor_measurement.insertOperator(id);
    //             }
    //         }
    //         catch (std::runtime_error e) {
    //             rootLogger.error(e.what());
    //             return 1;
    //         }
    //         rootLogger.info("Operators added");
    //         taylor_measurement.write_output_file_header();
    //         taylor_measurement.computeOperators();
    //         rootLogger.info("Operators computed");

    //         std::vector<DerivativeOperatorMeasurement> results;
    //         taylor_measurement.collectResults(results);
    //         rootLogger.info("Results collected");
    //         for (DerivativeOperatorMeasurement &meas : results) {
    //             rootLogger.info("ID: ", meas.operatorId, ", Measurement: ", meas.measurement, ", Error: ", meas.std);
    //         }
        
    // }

    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
