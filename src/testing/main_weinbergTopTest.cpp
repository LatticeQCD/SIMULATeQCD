/* 
 * main_weinbergTopTest.cpp                                                               
 * 
 * Jangho Kim
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/Topology.h"
#include "../modules/observables/Weinberg.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include <cstdio>
#include "refValues_weinbergTop.h"
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

template<class floatT, size_t HaloDepth, RungeKuttaMethod RKmethod, Force force>
bool run(Gaugefield<floatT, USE_GPU, HaloDepth> &gauge,
        GaugeAction<floatT, USE_GPU, HaloDepth> &gAction,
        Topology<floatT, USE_GPU, HaloDepth> &topology,
        Weinberg<floatT, USE_GPU, HaloDepth> &weinberg,
        gradientFlowParam<floatT> &lp,
        std::vector<std::vector<double>> &reference_values,
        const floatT acceptance,
        unsigned int &flow_time_count){

    gradientFlow<floatT, HaloDepth, RKmethod, force> gFlow(gauge, lp.start_step_size(), lp.measurement_intervall()[0],
    lp.measurement_intervall()[1], lp.necessary_flow_times.get(), lp.accuracy());

    //! initialize some values for the measurement
    floatT flow_time = 0;
    floatT plaq, clov, topChar, topChar_imp, topChar_imp_imp, wb, wb_imp, wb_imp_imp;
    std::stringstream logStream, logStream_ref;
    logStream << std::fixed << std::setprecision(std::numeric_limits<floatT>::digits10 + 1);
    logStream_ref << std::fixed << std::setprecision(std::numeric_limits<floatT>::digits10 + 1);

    bool failed = false;
    bool continueFlow = true;

    rootLogger.info("     Flowtime            topCharge            topCharge_imp        top_Charge_imp_imp   weinberg              weinberg_imp          weinberg_imp_imp");

    //! flow the field until max flow time
    while(continueFlow) {

        if (flow_time_count >= reference_values.size()) {
            rootLogger.info("End of reference values reached!");
            failed = true;
            return failed;
        }

        continueFlow = gFlow.continueFlow();
        logStream << "     ";
        logStream_ref << "ref: ";

        const size_t n_obs = 7;
        floatT values[n_obs] = {flow_time, topology.topCharge(), topology.template topCharge<true,false>(), topology.template topCharge<false,true>(), weinberg.WB(), weinberg.template WB<true,false>(), weinberg.template WB<false,true>()};

        for (int i=0; i<n_obs; i++){
            bool tmpfailed = compare<floatT>(logStream, logStream_ref, values[i], floatT(reference_values[flow_time_count][i]), floatT(acceptance));
            failed = failed || tmpfailed;
        }

        rootLogger.info(logStream.str());
        rootLogger.info(logStream_ref.str());
        logStream.str(std::string()); logStream.clear();
        logStream_ref.str(std::string()); logStream_ref.clear();

        flow_time += gFlow.updateFlow();
        flow_time_count++;
        gauge.updateAll();
    }
    return failed;
}

template<class floatT>
bool run_test(int argc, char* argv[], CommunicationBase &commBase, const floatT tolerance) {
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 3;

    gradientFlowParam<floatT> lp;

    //! do not change these! they are hardcoded to match the reference values.
    lp.accuracy.set(1e-6);
    lp.latDim.set({4, 4, 4, 4});
    lp.start_step_size.set(0.01);
    lp.measurements_dir.set("./");
    lp.measurement_intervall.set({0.0, 1.0});
    lp.necessary_flow_times.set({0.0, 0.01, 0.02});

    lp.readfile(commBase, "../parameter/tests/weinbergTopTest.param", argc, argv);

    commBase.init(lp.nodeDim());

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, USE_GPU, HaloDepth> gauge(commBase);
    gauge.readconf_openqcd(lp.GaugefileName());

    GaugeUpdate<floatT, USE_GPU, HaloDepth> gaugeUpdate(gauge);

    LatticeContainer<USE_GPU, floatT> latticeContainer(commBase);
    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);

    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);
    Weinberg<floatT, USE_GPU, HaloDepth> weinberg(gauge);

    rootLogger.info("Comparison-tolerance to reference is ", tolerance);

    bool failed = false;
    unsigned int flow_time_count = 0;  // running index for the reference values

    //! loop over RK methods and forces.
    static_for<0, 1>::apply([&](auto i){ //only fixed_stepsize
        const auto RK_method = static_cast<RungeKuttaMethod>(static_cast<int>(i));
        static_for<0, 1>::apply([&](auto j){ //only Wilson flow
            ///Reset Gaugefield to reference.
            gauge.updateAll();
            rootLogger.info("Plaquette = ", gAction.plaquette());
            ///Run test
            rootLogger.info(">>>>>>>>>>> RK_method=", RungeKuttaMethods[i], ", Force=", Forces[j], " <<<<<<<<<<<");
            const auto force = static_cast<Force>(static_cast<int>(j));
            bool tmpfailed = run<floatT, HaloDepth, RK_method, force>
                    (gauge, gAction, topology, weinberg, lp, refValues_weinbergTop, tolerance, flow_time_count);
            failed = failed || tmpfailed;
        });
    });
    return failed;
}

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    //! how large can the difference to the reference values be?
    const double double_tolerance = 1e-4;

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
