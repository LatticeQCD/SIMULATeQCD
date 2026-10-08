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

const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
const size_t HaloDepthSpin = 4;
typedef float floatT; // Define the precision here
typedef float PREC;

// Everything that depends on the number of simultaneous right-hand sides (NStacks)
template<size_t NStacks>
int runTaylorMeasurement(TaylorMeasurementParameters &param, CommunicationBase &commBase,
                         Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge, grnd_state<true> &d_rand) {

    rootLogger.info("Solving for ", NStacks, " right-hand sides simultaneously");

    // Read the Eigenvalues and Eigenvectors
    Eigenpairs<PREC,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairs(commBase);
    rootLogger.info("Read eigenvectors and eigenvalues from ", param.eigen_file());
    eigenpairs.readEigenpairsFromFile(param.eigen_file());
    eigenpairs.updateAll();

    for (double mass : param.valence_masses.get()) {
        rootLogger.info("Using mass ", mass);

        TaylorMeasurement<floatT, true, HaloDepthGauge, HaloDepthSpin, NStacks> taylor_measurement(gauge, eigenpairs, param, mass, param.use_naik_epsilon(), d_rand);
        try {
            for (const auto& id : param.operator_ids.get()) {
                taylor_measurement.insertOperator(id);
            }
        }
        catch (const std::runtime_error& e) {
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
    return 0;
}

// main
int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);

    // try reading parameter file from the same directory 
    rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);

    
    commBase.init(param.nodeDim());
    // if (commBase.getNumberProcesses() == 1) {
    //     commBase.forceHalos(true);
    // }

    rootLogger.info("STARTING Taylor Measurement:");

    if (sizeof(floatT)==4) {
      rootLogger.info("update done in single precision");
    } else if(sizeof(floatT)==8) {
      rootLogger.info("update done in double precision");
    } else {
      rootLogger.info("update done in unknown precision");
    }


    initIndexer(HaloDepthGauge, param, commBase);
    stdLogger.setVerbosity(INFO);

    // const int sizeh = param.latDim[0]*param.latDim[1]*param.latDim[2]*param.latDim[3]/2;

    // // Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge(commBase);  
    // naik_epsilon(use_naik_epsilon ? get_naik_epsilon_from_amc(mass) : 0.0);  /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge_smeared(commBase);
    Gaugefield<floatT,true,HaloDepthGauge,U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, true, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
    smearing.SmearAll();

    
    if (param.valence_masses.numberValues() == 0) {
        rootLogger.error("No valence masses specified, aborting");
        return 1;
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    rootLogger.info("Rng initialized with seed ", param.seed());


    // run as a standalone programm using the parameter file
    rootLogger.info("Starting in Standalone Mode");

        

    // NStacks is a template parameter, so only the values instantiated here can be chosen
    int status;
    switch (param.num_rhs()) {
        case 1: status = runTaylorMeasurement<1>(param, commBase, gauge, d_rand); break;
        case 2: status = runTaylorMeasurement<2>(param, commBase, gauge, d_rand); break;
        case 4: status = runTaylorMeasurement<4>(param, commBase, gauge, d_rand); break;
        case 5: status = runTaylorMeasurement<5>(param, commBase, gauge, d_rand); break;
        case 8: status = runTaylorMeasurement<8>(param, commBase, gauge, d_rand); break;
        default:
            rootLogger.error("num_rhs = ", param.num_rhs(), " not supported; choose one of 1, 2, 4, 5, 8");
            return 1;
    }
    if (status != 0) {
        return status;
    }

    // output file

    ///prepare output file
    std::stringstream outputfilename,cbeta,cstream;
    outputfilename << param.measurements_dir() << "TaylorMeasurement_l" << param.latDim[0] << param.latDim[3] << "f21";

    outputfilename << ".d";

    std::ofstream resultfile;
    if (true) {
        resultfile.open(outputfilename.str());
        rootLogger.info("output_file_name: " ,  outputfilename.str());
    }

    // output_file_name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
