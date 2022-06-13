/*
 * main_GeneralFunctorTest.cpp
 *
 * This file is a test for the measurement of the
 * taylor coefficients of Z(mu)
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/TaylorMeasurement.h"
#include "../modules/dslash/condensate.h"
//#include "testing.h" // for comparing stuff

#define PREC double

// TODO define test class


int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    try {
        param.readfile(commBase, "../parameter/tests/TaylorMeasurementTest.param", argc, argv);
    }
    catch (std::runtime_error e) {
        // just try reading from the same directory (this makes it useful even when not in this fixed folder structure)
        rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
        param.readfile(commBase, "./TaylorMeasurement.param", argc, argv);
    }

    commBase.init(param.nodeDim());

    const int HaloDepth = 2; // >= 1 for multi gpu
    const int HaloDepthSpin = 4;
    initIndexer(HaloDepth, param, commBase);
    stdLogger.setVerbosity(INFO);

    /// More initialization.
    Gaugefield<PREC,true,HaloDepth>  gauge(commBase);      /// gauge field

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

#if 0
    GaugeAction<PREC,true,HaloDepth> gauge_action(gauge);
    rootLogger.info("Measure Plaquette");
    StopWatch<true> timer;
    timer.start();

    PREC plaq = gauge_action.plaquette();

    rootLogger.info("Took: ", timer.stop(), "ms");
    rootLogger.info("Plaquette ", plaq);

    rootLogger.info("Do a timed dotProduct of spinorfields");

    SpinorfieldAll<double, true, HaloDepthSpin, 1> spinors(commBase);
    {
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(param.seed());
        d_rand = h_rand;
        spinors.gauss(d_rand.state);
    }
    timer.reset();
    timer.start();
    const int runs = 1000;
    for (int i = 0; i < runs; i++) {
        GPUcomplex<double> dot = spinors.dotProduct(spinors);
    }
    rootLogger.info("One dotProduct takes ", timer.stop()/runs*1000.0, "µs");
#endif

    if (param.valence_masses.numberValues() == 0) {
        rootLogger.error("No valence masses specified, aborting");
        return 1;
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    rootLogger.info("Rng initialized");

    PREC mass = (PREC)param.valence_masses[0];

    rootLogger.info("Using mass ", mass);

#if 0
    const bool onDevice = true;
    const int NStacks = 1;
#if 1
    Gaugefield<PREC, onDevice, HaloDepth, U3R14> gauge_Naik(gauge.getComm());
    Gaugefield<PREC, onDevice, HaloDepth, R18> gauge_smeared(gauge.getComm());
    // To test the basics, compute tr(D^-1) here explicitly (it was giving different results as the densecode)
    HisqSmearing<PREC, onDevice, HaloDepth, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
    smearing.SmearAll(param.mu_f());

    SpinorfieldAll<PREC, onDevice, HaloDepthSpin, NStacks> spinorIn(commBase);
    SpinorfieldAll<PREC, onDevice, HaloDepthSpin, NStacks> spinorOut(commBase);
    Spinorfield<PREC, false, Odd, HaloDepthSpin, NStacks> spinorOdd(commBase);

    typedef GIndexer<LayoutSwitcher<Even>(), HaloDepthSpin> GIndE;
    typedef GIndexer<LayoutSwitcher<Odd>(), HaloDepthSpin> GIndO;
    typedef GIndexer<LayoutSwitcher<All>(), HaloDepthSpin> GInd;
    spinorIn.even.setPointSource(GIndE::getSiteFull(16, 16, 16, 4).coord, 0, 1.0);
    spinorOut.odd.template iterateOverBulk<>(dDdmuFunctor<PREC, true, Even, HaloDepth, HaloDepthSpin, NStacks>(spinorIn.even, gauge_smeared, gauge_Naik, 2));
    rootLogger.info("operator computed");
    spinorOdd = spinorOut.odd;
    rootLogger.info(spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 16, 1)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 16, 3)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 16, 5)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 16, 7)), "\n");
    rootLogger.info(spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 14, 1)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 14, 3)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 14, 5)), "\n",
                    spinorOdd.getAccessor().getElement(GIndO::getSiteFull(16, 16, 14, 7)), "\n");
    
    {
        rootLogger.info("Gaugefield Naik:");
        Gaugefield<PREC, false, HaloDepth, U3R14> gauge_Naik_CPU(gauge.getComm(), "gaugenaik_copy");
        gauge_Naik_CPU = gauge_Naik;
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 0, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 1, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 2, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 3, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 4, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 5, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 6, 3)));
        rootLogger.info(gauge_Naik_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 7, 3)));
    }
    {
        rootLogger.info("Gaugefield Smeared:");
        Gaugefield<PREC, false, HaloDepth, R18> gauge_smeared_CPU(gauge.getComm(), "gauge_smeared_copy");
        gauge_smeared_CPU = gauge_smeared;
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 0, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 1, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 2, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 3, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 4, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 5, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 6, 3)));
        rootLogger.info(gauge_smeared_CPU.getAccessor().getElement(GInd::getSiteMu(16, 16, 16, 7, 3)));
    }

    double norm = GIndexer<All, HaloDepth>::getLatData().globvol4;// * 3;
    //GCOMPLEX(double) tr = spinorIn.dotProduct(spinorOut) / norm;
    double tr = spinorIn.realdotProduct(spinorOut) / norm;
    rootLogger.info("Measured tr(D^-1) = ", tr);
#else
    RhmcParameters rhmc_param;
    rhmc_param.mu_f.set(param.mu_f());
    rhmc_param.cgMax_meas.set(param.cgMax());
    rhmc_param.residue_meas.set(param.residue());
    SimpleArray<double,NStacks> tr_arr = measure_condensate<PREC, true, HaloDepth, HaloDepthSpin, NStacks>(commBase, rhmc_param, mass, gauge, d_rand);
    rootLogger.info("Measured tr(D^-1) = ", tr_arr[0]);
#endif
#else

    const int NStacks = 8;
    TaylorMeasurement<PREC, true, HaloDepth, HaloDepthSpin, NStacks> taylor_measurement(gauge, param, mass, param.use_naik_epsilon(), d_rand);
    rootLogger.info("Class initialized");
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
#endif
    // output file name: "TaylorMeasurement_" + ensemble_id (like l328...) + "." + param.conf_nr
}
