/*
 * main_TaylorMeasurement.cpp
 *
 * This file is the main file for the application for the measurement of the
 * taylor coefficients of Z(mu)
 *
 */
#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"
#include "../modules/dslash/condensate.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../testing/testing.h" // for comparing stuff

// main
int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    
    // try reading parameter file from the same directory 
    rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);


    commBase.init(param.nodeDim());

    const int HaloDepth = 2; // >= 1 for multi gpu
    const int HaloDepthSpin = 4;
    const int NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    typedef float floatT; // Define the precision here
    typedef float PREC;

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

    // const int sizeh = param.latDim[0]*param.latDim[1]*param.latDim[2]*param.latDim[3]/2;

    // // Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    Gaugefield<floatT,true,HaloDepth,R14> gauge(commBase);  
    
    // naik_epsilon(use_naik_epsilon ? get_naik_epsilon_from_amc(mass) : 0.0);  /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    Gaugefield<floatT,true,HaloDepth,R18> gauge_smeared(commBase);
    Gaugefield<floatT,true,HaloDepth,U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, true, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    smearing.SmearAll();
    
    // Read the Eigenvalues and Eigenvectors
    eigenpairs<PREC,true,Even,HaloDepthSpin,NStacks> eigenpairs(commBase);
    rootLogger.info("Read eigenvectors and eigenvalues from ", param.eigen_file());
    eigenpairs.read_evnersc(param.num_toread_vectors(), param.eigen_file());
    eigenpairs.updateAll();

    HisqDSlash<floatT,true,Even,HaloDepth,HaloDepthSpin,NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

    eigenpairs.tester(dslash, param.num_toread_vectors());

    // if (param.valence_masses.numberValues() == 0) {
    //     rootLogger.error("No valence masses specified, aborting");
    //     return 1;
    // }

    // grnd_state<false> h_rand;
    // grnd_state<true> d_rand;
    // h_rand.make_rng_state(param.seed());
    // d_rand = h_rand;

    // rootLogger.info("Rng initialized with seed ", param.seed());


    // // run as a standalone programm using the parameter file
    // rootLogger.info("Starting in Standalone Mode");

        

    // for (double mass : param.valence_masses.get()) {
    //     rootLogger.info("Using mass ", mass);

    //     TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
    //     try {
    //         for (const auto& id : param.operator_ids.get()) {
    //             taylor_measurement.insertOperator(id);
    //         }
    //     }
    //     catch (std::runtime_error e) {
    //         rootLogger.error(e.what());
    //         return 1;
    //     }
    //     rootLogger.info("Operators added");
    //     taylor_measurement.write_output_file_header();
    //     taylor_measurement.computeOperators();
    //     rootLogger.info("Operators computed");

    //     std::vector<DerivativeOperatorMeasurement> results;
    //     taylor_measurement.collectResults(results);
    //     rootLogger.info("Results collected");
    //     for (DerivativeOperatorMeasurement &meas : results) {
    //         rootLogger.info("ID: ", meas.operatorId, ", Measurement: ", meas.measurement, ", Error: ", meas.std);
    //     }
        
    // }

    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
