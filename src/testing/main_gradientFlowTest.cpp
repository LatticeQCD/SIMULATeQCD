/*
 * main_gradientFlowTest.cpp
 *
 * Flows the stored configuration l20t20b06498a_nersc.302500 for all combinations of RK_method and force and compares
 * flow time, plaquette, clover and topological charge against refValues_gradFlow.h. Starting from a stored
 * configuration keeps the test independent of the gauge-update code.
 *
 * It also checks that the flow lands exactly once on every measured flow time (see runMeasuredTimes), also when that
 * time is not a necessary flow time or not a multiple of the step size.
 *
 * To regenerate the reference values, run with print_reference=1: the comparison is skipped and the values are
 * written to <measurements_dir>/refValues_gradFlow.h, which replaces src/testing/refValues_gradFlow.h.
 *
 */

#include "../simulateqcd.h"
#include "../modules/observables/topology.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include <cstdio>
#include <fstream>
#include "refValues_gradFlow.h"

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
    Parameter<bool> print_reference;

    // constructor
    gradientFlowParam() {
        add(start_step_size, "start_step_size");
        add(accuracy, "accuracy");
        add(measurements_dir, "measurements_dir");
        addOptional(necessary_flow_times, "necessary_flow_times");
        add(measurement_intervall, "measurement_intervall");
        addDefault(print_reference, "print_reference", false);
    }
};

template<typename floatT>
bool compare(std::stringstream &logStream, std::stringstream &logStream_ref, floatT value, floatT reference, floatT tolerance){
    logStream << value << "  " ;
    bool success = isApproximatelyEqual(value, reference, tolerance);
    if (!success) {
        logStream_ref << CoutColors::red << reference << CoutColors::reset << "  ";
    } else {
        logStream_ref << CoutColors::green << reference << CoutColors::reset << "  ";
    }
    return !success;
}

// with generated != nullptr the values are collected there instead of being compared to reference_values
template<class floatT, size_t HaloDepth, RungeKuttaMethod RKmethod, Force force>
bool run(Gaugefield<floatT, USE_GPU, HaloDepth> &gauge,
        GaugeAction<floatT, USE_GPU, HaloDepth> &gAction,
        Topology<floatT, USE_GPU, HaloDepth> &topology,
        gradientFlowParam<floatT> &lp,
        std::vector<std::vector<double>> &reference_values,
        const floatT acceptance,
        unsigned int &flow_time_count,
        std::vector<std::vector<double>> *generated){

    gradientFlow<floatT, HaloDepth, RKmethod, force> gFlow(gauge, lp.start_step_size(), lp.measurement_intervall()[0],
    lp.measurement_intervall()[1], lp.necessary_flow_times.get(), std::vector<floatT>(), lp.accuracy());

    //! initialize some values for the measurement
    floatT flow_time = 0;
    floatT plaq, clov, topChar;
    std::stringstream logStream, logStream_ref;
    logStream << std::fixed << std::setprecision(std::numeric_limits<floatT>::digits10 + 1);
    logStream_ref << std::fixed << std::setprecision(std::numeric_limits<floatT>::digits10 + 1);

    bool failed = false;
    bool continueFlow = true;

    rootLogger.info("Flowtime            Plaquette           Clover              topCharge");

    //! flow the field until max flow time
    while(continueFlow) {

        if (generated == nullptr && flow_time_count >= reference_values.size()) {
            rootLogger.info("End of reference values reached! Is the lattice size correct?");
            failed = true;
            return failed;
        }

        continueFlow = gFlow.continueFlow();

        const size_t n_obs = 4;
        floatT values[n_obs] = {flow_time, gAction.plaquette(), gAction.clover(), topology.topCharge()};

        if (generated != nullptr) {
            generated->push_back(std::vector<double>(values, values + n_obs));
            for (int i=0; i<n_obs; i++){
                logStream << values[i] << "  ";
            }
            rootLogger.info(logStream.str());
            logStream.str(std::string()); logStream.clear();
        } else {
            logStream << "     ";
            logStream_ref << "ref: ";
            for (int i=0; i<n_obs; i++){
                bool tmpfailed = compare<floatT>(logStream, logStream_ref, values[i], floatT(reference_values[flow_time_count][i]), floatT(acceptance));
                failed = failed || tmpfailed;
            }
            rootLogger.info(logStream.str());
            rootLogger.info(logStream_ref.str());
            logStream.str(std::string()); logStream.clear();
            logStream_ref.str(std::string()); logStream_ref.clear();
        }

        flow_time += gFlow.updateFlow();
        flow_time_count++;
        gauge.updateAll();
    }
    return failed;
}

// The flow has to land exactly on every measured flow time, also when it is not a necessary flow time, not a multiple of
// the step size, or given in both lists. Needs no reference values.
template<class floatT, size_t HaloDepth, RungeKuttaMethod RKmethod, Force force>
bool runMeasuredTimes(Gaugefield<floatT, USE_GPU, HaloDepth> &gauge,
                      gradientFlowParam<floatT> &lp,
                      const std::vector<floatT> &necessaryTimes,
                      const std::vector<floatT> &measuredTimes,
                      const floatT tolerance) {

    gradientFlow<floatT, HaloDepth, RKmethod, force> gFlow(gauge, lp.start_step_size(), lp.measurement_intervall()[0],
    lp.measurement_intervall()[1], necessaryTimes, measuredTimes, lp.accuracy());

    floatT flow_time = lp.measurement_intervall()[0];
    std::vector<floatT> hitTimes;
    bool continueFlow = true;

    while (continueFlow) {
        continueFlow = gFlow.continueFlow();
        if (gFlow.checkIfMeasuredTime()) {
            hitTimes.push_back(flow_time);
        }
        flow_time += gFlow.updateFlow();
        gauge.updateAll();
    }

    bool failed = false;
    for (floatT measured : measuredTimes) {
        int hits = 0;
        for (floatT hit : hitTimes) {
            if (isApproximatelyEqual(hit, measured, tolerance)) hits++;
        }
        if (hits != 1) {
            rootLogger.error("Measured flow time ", measured, " was hit ", hits, " times, expected once.");
            failed = true;
        }
    }
    if (hitTimes.size() != measuredTimes.size()) {
        rootLogger.error("checkIfMeasuredTime() was true ", hitTimes.size(), " times, expected ", measuredTimes.size());
        failed = true;
    }
    return failed;
}

void writeReferenceFile(const std::string &fileName, const std::vector<std::vector<double>> &values) {
    std::ofstream file(fileName);
    file << "/*\n"
            " * refValues_gradFlow.h\n"
            " *\n"
            " * Generated by gradientFlowTest with print_reference=1.\n"
            " *\n"
            " * Contains ordered reference values {flow time, plaquette, clover, topological charge} for the\n"
            " * gradientFlowTest for all combinations of RK_method and force, starting from l20t20b06498a_nersc.302500.\n"
            " */\n"
            "\n"
            "#pragma once\n"
            "#include <vector>\n"
            "\n"
            "std::vector<std::vector<double>> refValues_gradFlow = {\n";
    for (size_t row = 0; row < values.size(); row++) {
        char line[128];
        std::snprintf(line, sizeof(line), "    {%.16f, %.16f, %.16f, %.16f}%s\n",
                      values[row][0], values[row][1], values[row][2], values[row][3], row + 1 < values.size() ? "," : "");
        file << line;
    }
    file << "};\n";
}

template<class floatT>
bool run_test(int argc, char* argv[], CommunicationBase &commBase, const floatT tolerance) {
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 3;

    gradientFlowParam<floatT> lp;

    //! do not change these! they are hardcoded to match the reference values.
    lp.accuracy.set(1e-5);
    lp.latDim.set({20, 20, 20, 20});
    lp.start_step_size.set(0.05);
    lp.measurements_dir.set("./");
    lp.measurement_intervall.set({0.0, 1.0});
    lp.necessary_flow_times.set({0.0, 0.1, 0.5, 1.0});

    lp.readfile(commBase, "../parameter/tests/gradientFlowTest.param", argc, argv);

    commBase.init(lp.nodeDim());

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, USE_GPU, HaloDepth> gauge(commBase);

    LatticeContainer<USE_GPU, floatT> latticeContainer(commBase);
    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);

    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);

    const bool printReference = lp.print_reference();
    std::vector<std::vector<double>> generated;
    if (printReference) {
        rootLogger.info("print_reference=1: generating reference values, no comparison");
    } else {
        rootLogger.info("Comparison-tolerance to reference is ", tolerance);
    }

    bool failed = false;
    unsigned int flow_time_count = 0;  // running index for the reference values

    //! loop over RK methods and forces.
    static_for<0, 3>::apply([&](auto i){
        const auto RK_method = static_cast<RungeKuttaMethod>(static_cast<int>(i));
        static_for<0, 2>::apply([&](auto j){
            ///Reset Gaugefield to the stored configuration.
            gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
            gauge.updateAll();
            rootLogger.info("Plaquette = ", gAction.plaquette());

            ///Run test
            rootLogger.info(">>>>>>>>>>> RK_method=", RungeKuttaMethods[i], ", Force=", Forces[j], " <<<<<<<<<<<");
            const auto force = static_cast<Force>(static_cast<int>(j));
            bool tmpfailed = run<floatT, HaloDepth, RK_method, force>
                    (gauge, gAction, topology, lp, refValues_gradFlow, tolerance, flow_time_count, printReference ? &generated : nullptr);
            failed = failed || tmpfailed;
        });
    });

    if (printReference && commBase.IamRoot()) {
        const std::string fileName = lp.measurements_dir() + "refValues_gradFlow.h";
        writeReferenceFile(fileName, generated);
        rootLogger.info("Wrote ", generated.size(), " reference rows to ", fileName);
    }

    //! measured flow times: 0.123 is neither a necessary time nor a multiple of the step size 0.05,
    //! 0.5 is in both lists and must not produce a zero-length step.
    const std::vector<floatT> necessaryTimes = {0.5, 1.0};
    const std::vector<floatT> measuredTimes = {0.123, 0.5};
    static_for<0, 3>::apply([&](auto i){
        const auto RK_method = static_cast<RungeKuttaMethod>(static_cast<int>(i));
        gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
        gauge.updateAll();
        rootLogger.info(">>>>>>>>>>> measured flow times, RK_method=", RungeKuttaMethods[i], ", Force=", Forces[0], " <<<<<<<<<<<");
        bool tmpfailed = runMeasuredTimes<floatT, HaloDepth, RK_method, wilson>
                (gauge, lp, necessaryTimes, measuredTimes, tolerance);
        failed = failed || tmpfailed;
    });
    return failed;
}

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    //! how large can the difference to the reference values be?
    const double double_tolerance = 1e-9;

    stdLogger.info("TEST DOUBLE PRECISION");
    bool passfail_double = run_test<double>(argc, argv, commBase, double_tolerance);

    if (passfail_double) {
        rootLogger.error("At least one test failed!");
        return 1;
    } else {
        rootLogger.info(CoutColors::green ,  "All Tests passed!" ,  CoutColors::reset);
    }

    return 0;
}
