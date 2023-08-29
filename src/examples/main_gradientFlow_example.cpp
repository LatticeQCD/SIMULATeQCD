/*
 * main_gradientFlow_example.cpp
 *
 * Lukas Mazur, 25 Nov 2018
 *
 */

#include "../simulateqcd.h"
#include "../modules/gradientFlow/gradientFlow.h"

#define USE_GPU true
//define precision
#if SINGLEPREC
#define PREC float
#else
#define PREC double
#endif

template<class floatT>
struct gradientFlowParam : LatticeParameters {

    Parameter<std::string> force;
    Parameter<floatT> start_step_size;
    Parameter<std::string> RK_method;
    Parameter<floatT> accuracy;
    Parameter<floatT, 2> measurement_intervall;
    DynamicParameter<floatT> necessary_flow_times;

    // constructor
    gradientFlowParam() {
        addDefault(force, "force", std::string("zeuthen"));
        add(start_step_size, "start_step_size");
        addDefault(RK_method, "RK_method", std::string("adaptive_stepsize_allgpu"));
        addDefault(accuracy, "accuracy", floatT(1e-5));
        addOptional(necessary_flow_times, "necessary_flow_times");
        add(measurement_intervall, "measurement_intervall");
    }
};

template<class floatT, size_t HaloDepth, typename gradFlowClass>
void run(gradFlowClass &gradFlow, Gaugefield<floatT, USE_GPU, HaloDepth> &gauge, gradientFlowParam<floatT> &lp) {

    std::stringstream logStream;
    logStream << std::fixed << std::setprecision(6);

    rootLogger.info("Read configuration");
    gauge.readconf_nersc(lp.GaugefileName());
    gauge.updateAll();

    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);
    floatT plaq;
    floatT flow_time = 0;

    /// flow the field until max flow time
    bool continueFlow = true;
    while (continueFlow) {
        continueFlow = gradFlow.continueFlow();

        logStream.str("");
        logStream << "t = " << floatT(flow_time) << ": ";

        ///Calculate observables
        plaq = gAction.plaquette();
        logStream << "   Plaquette = " << plaq;

        rootLogger.info(logStream.str());
        flow_time += gradFlow.updateFlow();

        gauge.updateAll();
    }

    rootLogger.info("done");
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    gradientFlowParam<PREC> lp;
    lp.readfile(commBase, "../parameter/applications/gradientFlow.param", argc, argv);
    commBase.init(lp.nodeDim());

    if (lp.force() != "zeuthen"
        or (lp.RK_method() != "adaptive_stepsize" and lp.RK_method() != "adaptive_stepsize_allgpu")) {
        throw std::runtime_error(stdLogger.fatal("Force is not zeuthen or RK_method is not adaptive stepsize!"));
    }
    const size_t HaloDepth = 3;

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<PREC, USE_GPU, HaloDepth> gauge(commBase);

    ///initialize gradient flow class & run with the given parameters
    if (lp.RK_method() == "adaptive_stepsize") {

        gradientFlow<PREC, HaloDepth, adaptive_stepsize, zeuthen> gradFlow(gauge, lp.start_step_size(),
                                              lp.measurement_intervall()[0], lp.measurement_intervall()[1],
                                              lp.necessary_flow_times.get(), lp.accuracy());
        run<PREC, HaloDepth, gradientFlow<PREC, HaloDepth, adaptive_stepsize, zeuthen> >(gradFlow, gauge, lp);

    } else if (lp.RK_method() == "adaptive_stepsize_allgpu") {

        gradientFlow<PREC, HaloDepth, adaptive_stepsize_allgpu, zeuthen> gradFlow(gauge, lp.start_step_size(),
                                              lp.measurement_intervall()[0], lp.measurement_intervall()[1],
                                              lp.necessary_flow_times.get(), lp.accuracy());
        run<PREC, HaloDepth, gradientFlow<PREC, HaloDepth, adaptive_stepsize_allgpu, zeuthen> >(gradFlow, gauge, lp);
    }

    return 0;
}

