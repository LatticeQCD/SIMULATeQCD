/* 
 * main_gradientFlowTest.cpp                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/Topology.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include <cstdio>
#include "refValues_gradFlow.h"
#include "../modules/gauge_updates/PureGaugeUpdates.h"

#define USE_GPU true

template<class floatT>
struct gradientFlowParam : LatticeParameters {
    Parameter<floatT> start_step_size;

    // acceptance range
    Parameter<floatT> accuracy;
    Parameter<std::string> measurements_dir;
    Parameter<std::string> RK_method;
    Parameter<floatT, 2> measurement_intervall;
    DynamicParameter<floatT> necessary_flow_times;

    // constructor
    gradientFlowParam() {
        add(start_step_size, "start_step_size");
        add(accuracy, "accuracy");
        add(measurements_dir, "measurements_dir");
        addOptional(necessary_flow_times, "necessary_flow_times");
        add(measurement_intervall, "measurement_intervall");
    }
};

template<typename floatT>
bool compare(std::stringstream &logStream, const std::string& obs, floatT value, floatT reference, floatT acceptance){
    logStream << " " << obs << " = " << value;
    bool success = isApproximatelyEqual(value, reference, acceptance);
    if (!success) {
        logStream << CoutColors::red << " != " << reference << CoutColors::reset << "|";
    }
    return !success;
}

template<class floatT, size_t HaloDepth, RungeKuttaMethod RKmethod, Force force>
bool run(Gaugefield<floatT, USE_GPU, HaloDepth> &gauge,
        GaugeAction<floatT, USE_GPU, HaloDepth> &gAction,
        Topology<floatT, USE_GPU, HaloDepth> &topology,
        gradientFlowParam<floatT> &lp,
        std::vector<std::vector<double>> &reference_values,
        const floatT acceptance){

    gradientFlow<floatT, HaloDepth, RKmethod, force> gFlow(gauge, lp.start_step_size(), lp.measurement_intervall()[0],
    lp.measurement_intervall()[1], lp.necessary_flow_times.get(), lp.accuracy());

    // initialize some values for the measurement
    floatT flow_time = 0;
    unsigned int flow_time_count = 0;
    floatT plaq, clov, topChar;
    std::stringstream logStream;
    logStream << std::fixed << std::setprecision(std::numeric_limits<floatT>::digits10 + 1);

    bool failed = false;
    bool continueFlow = true;

    // flow the field until max flow time
    while(continueFlow) {

        if (flow_time_count >= reference_values.size()) {
            rootLogger.info("End of reference values reached!");
            failed = true;
            return failed;
        }

        continueFlow = gFlow.continueFlow();
        logStream.str("");
        logStream << "   t = " << flow_time;

        compare<floatT>(logStream, "", flow_time, floatT(reference_values[flow_time_count][0]), floatT(acceptance));
        compare<floatT>(logStream, "Plaquette", gAction.plaquette(), floatT(reference_values[flow_time_count][1]), floatT(acceptance));
        compare<floatT>(logStream, "Clover", gAction.clover(), floatT(reference_values[flow_time_count][2]), floatT(acceptance));
        compare<floatT>(logStream, "topCharge", topology.topCharge(), floatT(reference_values[flow_time_count][3]), floatT(acceptance));

        rootLogger.info(logStream.str());

        flow_time += gFlow.updateFlow();
        flow_time_count++;
        gauge.updateAll();
    }
    return failed;
}

template<class floatT>
bool run_test(int argc, char* argv[], CommunicationBase &commBase, const floatT acceptance, const floatT accuracy) {
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 3;

    gradientFlowParam<floatT> lp;

    //! do not change these! they are hardcoded to match the reference values.
    lp.accuracy.set(accuracy);
    lp.latDim.set({32, 32, 32, 16});
    lp.start_step_size.set(0.01);
    lp.measurements_dir.set("./");
    lp.measurement_intervall.set({0.0, 1.0});
    lp.necessary_flow_times.set({0.0, 0.1, 0.5, 1.0});

    lp.readfile(commBase, "../parameter/tests/gradientFlowTest.param", argc, argv);

    commBase.init(lp.nodeDim());

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, USE_GPU, HaloDepth> gauge(commBase);
    GaugeUpdate<floatT, USE_GPU, HaloDepth> gaugeUpdate(gauge);
    gaugeUpdate.set_gauge_to_reference();

    LatticeContainer<USE_GPU, floatT> latticeContainer(commBase);
    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);
    rootLogger.info(gAction.plaquette());
    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);

    rootLogger.info("Comparison-tolerance to reference is ", acceptance);

    bool failed = false;

    //! loop over RK methods and forces.
     static_for<0, 3>::apply([&](auto i){
         const auto RKmethod = static_cast<RungeKuttaMethod>(static_cast<int>(i));
         static_for<0, 2>::apply([&](auto j){
             const auto myforce = static_cast<Force>(static_cast<int>(j));
             failed = failed || run<floatT, HaloDepth, RKmethod, myforce>(gauge, gAction, topology, lp,wilson_values, acceptance);
         });
         gaugeUpdate.set_gauge_to_reference();
         gauge.updateAll();
     });
     return failed;
}

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    //how large can the difference to the reference values be?
    const double double_acceptance = 1e-9;

    //adaptive stepsize accuracy. just a parameter for reference values, do not change!
    const double double_accuracy = 1e-9;

    stdLogger.info("TEST DOUBLE PRECISION");
    bool passfail_double = run_test<double>(argc, argv, commBase, double_acceptance, double_accuracy);

    rootLogger.info(CoutColors::green ,  "           ");
    if (passfail_double) { // || passfail_float
        rootLogger.error("At least one test failed!");
        return 1;
    } else {
        rootLogger.info(CoutColors::green ,  "All Tests passed!" ,  CoutColors::reset);
    }

    return 0;
}