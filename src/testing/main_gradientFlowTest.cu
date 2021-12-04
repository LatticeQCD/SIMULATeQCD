/* 
 * main_gradientFlowTest.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/Topology.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include <cstdio>
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

    // constructor
    gradientFlowParam() {
        add(start_step_size, "start_step_size");
        add(accuracy, "accuracy");
        add(measurements_dir, "measurements_dir");
        addOptional(necessary_flow_times, "necessary_flow_times");
        add(measurement_intervall, "measurement_intervall");
    }
};

template<class floatT, size_t HaloDepth, typename gradFlowClass>
bool run(gradFlowClass &gradFlow,
        Gaugefield<floatT, USE_GPU, HaloDepth> &gauge,
        GaugeAction<floatT, USE_GPU, HaloDepth> &gAction,
        Topology<floatT, USE_GPU, HaloDepth> &topology,
        std::vector<std::vector<double>> &reference_values, const floatT acceptance){

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

        continueFlow = gradFlow.continueFlow();
        logStream.str("");
        logStream << "   t = " << flow_time << ": ";

        bool compareFlowTime = isApproximatelyEqual(flow_time, floatT(reference_values[flow_time_count][0]), floatT(acceptance));
        if (!compareFlowTime) {
            failed = true;
            logStream << CoutColors::red << " != " << reference_values[flow_time_count][0] << CoutColors::reset <<  ":";
        }

        // compute the observables on the smeared field

        plaq = gAction.plaquette();
        logStream << "   Plaquette = " << plaq;
        bool comparePlaquette = isApproximatelyEqual(plaq, floatT(reference_values[flow_time_count][1]), floatT(acceptance));
        if (!comparePlaquette) {
            failed = true;
            logStream << CoutColors::red << " != " << reference_values[flow_time_count][1] << CoutColors::reset;
        }

        clov = gAction.clover();
        logStream << "   Clover = " << clov;
        bool compareClover = isApproximatelyEqual(clov, floatT(reference_values[flow_time_count][2]), floatT(acceptance));
        if (!compareClover) {
            failed = true;
            logStream << CoutColors::red << " != " << reference_values[flow_time_count][2] << CoutColors::reset;
        }

        topChar = topology.topCharge();
        logStream << "   topCharge = " << topChar;
        bool compareTopCh = isApproximatelyEqual(topChar, floatT(reference_values[flow_time_count][3]), floatT(acceptance));
        if (!compareTopCh) {
            failed = true;
            logStream << CoutColors::red << " != " << reference_values[flow_time_count][3] << CoutColors::reset;
        }
        rootLogger.info(logStream.str());

        flow_time += gradFlow.updateFlow();
        flow_time_count++;
        gauge.updateAll();
    }
    return failed;
}


template<class floatT> bool run_test(int argc, char* argv[], CommunicationBase &commBase, const floatT acceptance, const floatT accuracy) {
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 3;

    MicroTimer timer;
    gradientFlowParam<floatT> lp;

    //do not change these!
    lp.accuracy.set(accuracy);
    lp.latDim.set({8, 8, 8, 8});
    lp.GaugefileName.set("../test_conf/oneInstantonConf_s8t8_nersc");
    lp.beta.set(6.498);
    lp.format.set("nersc");
    lp.endianness.set("auto");
    lp.start_step_size.set(0.01);
    lp.measurements_dir.set("./");
    lp.measurement_intervall.set({0.0, 1.0});
    lp.necessary_flow_times.set({0.5, 1.0});

    //in case you want to change the nodeDim for multi-gpu testing (actually this is broken because the test lattice is too small...)
    lp.readfile(commBase, "../parameter/tests/gradientFlowTest.param", argc, argv);

    commBase.init(lp.nodeDim());

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, USE_GPU, HaloDepth> gauge(commBase);
    Gaugefield<floatT, USE_GPU, HaloDepth> gauge_backup(commBase);
    LatticeContainer<USE_GPU, floatT> redBase(commBase);

    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);
    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);

    rootLogger.info("Comparison-tolerance to reference is " ,  acceptance);
    rootLogger.info("Read configuration");
    gauge.readconf_nersc(lp.GaugefileName());
    gauge_backup = gauge;
    gauge.updateAll();

    bool zFlowAdFailed;
    bool zFlowAdAllGPUFailed;
    bool wFlowFailed;
    bool zFlowFailed;
    bool wFlowAdFailed;
    bool wFlowAdAllGPUFailed;

    wilsonFlow<floatT, HaloDepth, fixed_stepsize> wFlow(gauge, lp.start_step_size(),
                                                        lp.measurement_intervall()[0], lp.measurement_intervall()[1],
                                                        lp.necessary_flow_times.get());
    wilsonFlow<floatT, HaloDepth, adaptive_stepsize> wFlowAd(gauge, lp.start_step_size(),
                                                             lp.measurement_intervall()[0],
                                                             lp.measurement_intervall()[1],
                                                             lp.necessary_flow_times.get(), lp.accuracy());
    wilsonFlow<floatT, HaloDepth, adaptive_stepsize_allgpu> wFlowAdAllGPU(gauge, lp.start_step_size(),
                                                                          lp.measurement_intervall()[0],
                                                                          lp.measurement_intervall()[1],
                                                                          lp.necessary_flow_times.get(),
                                                                          lp.accuracy());
    zeuthenFlow<floatT, HaloDepth, fixed_stepsize> zFlow(gauge, lp.start_step_size(),
                                                         lp.measurement_intervall()[0], lp.measurement_intervall()[1],
                                                         lp.necessary_flow_times.get());
    zeuthenFlow<floatT, HaloDepth, adaptive_stepsize> zFlowAd(gauge, lp.start_step_size(),
                                                              lp.measurement_intervall()[0],
                                                              lp.measurement_intervall()[1],
                                                              lp.necessary_flow_times.get(), lp.accuracy());
    zeuthenFlow<floatT, HaloDepth, adaptive_stepsize_allgpu> zFlowAdAllGPU(gauge, lp.start_step_size(),
                                                                           lp.measurement_intervall()[0],
                                                                           lp.measurement_intervall()[1],
                                                                           lp.necessary_flow_times.get(),
                                                                           lp.accuracy());

    timer.start();
    wFlowFailed = run<floatT, HaloDepth, wilsonFlow<floatT, HaloDepth, fixed_stepsize>>(wFlow, gauge, gAction, topology,
                                                                                         wilson_values, acceptance);
    timer.stop();
    rootLogger.info("complete time for standard Runge Kutta 3 and Wilson Force = " ,  timer);
    timer.reset();
    gauge = gauge_backup;
    gauge.updateAll();

    timer.start();
    wFlowAdFailed = run<floatT, HaloDepth, wilsonFlow<floatT, HaloDepth, adaptive_stepsize>>(wFlowAd, gauge, gAction,
                                                                                              topology,
                                                                                              std::is_same<floatT, double>::value ? wilsonAd_values : wilson_ad_values_float,
                                                                                              acceptance);
    timer.stop();
    rootLogger.info("complete time for adaptive Runge Kutta 3 and Wilson Force = " ,  timer);
    timer.reset();
    gauge = gauge_backup;
    gauge.updateAll();

    timer.start();
    wFlowAdAllGPUFailed = run<floatT, HaloDepth, wilsonFlow<floatT, HaloDepth, adaptive_stepsize_allgpu>>(
            wFlowAdAllGPU, gauge, gAction, topology,
            std::is_same<floatT, double>::value ? wilsonAd_values : wilson_ad_values_float, acceptance);
    timer.stop();
    rootLogger.info("complete time for adaptive all GPU Runge Kutta 3 and Wilson Force = " ,  timer);
    timer.reset();
    gauge = gauge_backup;
    gauge.updateAll();

    timer.start();
    zFlowFailed = run<floatT, HaloDepth, zeuthenFlow<floatT, HaloDepth, fixed_stepsize>>(zFlow, gauge, gAction, topology,
                                                                                          zeuthen_values, acceptance);
    timer.stop();
    rootLogger.info("complete time for standard Runge Kutta 3 and Zeuthen Force = " ,  timer);
    timer.reset();
    gauge.updateAll();
    gauge = gauge_backup;

    timer.start();
    zFlowAdFailed = run<floatT, HaloDepth, zeuthenFlow<floatT, HaloDepth, adaptive_stepsize>>(zFlowAd, gauge,
                                                                                               gAction, topology,
                                                                                               std::is_same<floatT, double>::value ? zeuthenAd_values : zeuthen_ad_values_float,
                                                                                               acceptance);
    timer.stop();
    rootLogger.info("complete time for adaptive Runge Kutta 3 and Zeuthen Force = " ,  timer);
    timer.reset();
    gauge = gauge_backup;
    gauge.updateAll();

    timer.start();
    zFlowAdAllGPUFailed = run<floatT, HaloDepth, zeuthenFlow<floatT, HaloDepth, adaptive_stepsize_allgpu>>(
            zFlowAdAllGPU, gauge, gAction,
            topology, std::is_same<floatT, double>::value ? zeuthenAd_values : zeuthen_ad_values_float, acceptance);
    timer.stop();
    rootLogger.info("complete time for adaptive all GPU Runge Kutta 3 and Zeuthen Force = " ,  timer);
    timer.reset();

    bool failed = (zFlowAdFailed || zFlowAdAllGPUFailed || wFlowFailed || zFlowFailed || wFlowAdFailed || wFlowAdAllGPUFailed);
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
        throw PGCError("At least one test failed!");
    } else {
        rootLogger.info(CoutColors::green ,  "All Tests passed!" ,  CoutColors::reset);
    }

    return 0;
}



