/*
* gradientFlowParameters.h
*
* File for the parameters of a gradient flow measurement
*/

#pragma once
#include "../../base/latticeParameters.h"


template<class floatT>
struct gradientFlowParam : LatticeParameters {

    //! ---------------------------------basic options you probably care about------------------------------------------
    Parameter<std::string> measurements_dir; //! where the output gets stored
    Parameter<std::string> force; //! wilson or zeuthen flow
    Parameter<std::string> RK_method; //! RK_method = {fixed_stepsize, adaptive_stepsize, adaptive_stepsize_allgpu}
    Parameter<floatT> start_step_size;
    Parameter<floatT> accuracy; //! only used for adaptive stepsize. difference between 2nd and 3rd order RK
    DynamicParameter<floatT> necessary_flow_times; //! these flow times will not be skipped over in the integration
    DynamicParameter<floatT> measured_corr_flow_times; //! for these flow times, the correlator measurements will be done
    Parameter<floatT, 2> measurement_intervall; //! measurement_intervall[0]: start, [1]: stop
    Parameter<bool> useHDF5; //! whether to use HDF5 file output

    //! ---------------------------------which observables should be measured on the flowed configuration?--------------

    Parameter<bool> plaquette;
    Parameter<bool> clover;
    Parameter<bool> cloverTimeSlices;
    Parameter<bool> topCharge;
    Parameter<bool> topChargeTimeSlices;
    Parameter<bool> topCharge_imp;
    Parameter<bool> topChargeTimeSlices_imp;
    Parameter<bool> topCharge_imp_imp;
    Parameter<bool> topChargeTimeSlices_imp_imp;
    Parameter<bool> weinberg;
    Parameter<bool> weinbergTimeSlices;
    Parameter<bool> weinberg_imp;
    Parameter<bool> weinbergTimeSlices_imp;
    Parameter<bool> weinberg_imp_imp;
    Parameter<bool> weinbergTimeSlices_imp_imp;
    Parameter<bool> ColorElectricCorrTimeSlices_naive;
    Parameter<bool> ColorElectricCorrTimeSlices_clover;
    Parameter<bool> ColorMagneticCorrTimeSlices_naive;
    Parameter<bool> ColorMagneticCorrTimeSlices_clover;
    Parameter<bool> RenormPolyakovSusc;

    Parameter<bool> topCharge_imp_block;
    Parameter<bool> topCharge_imp_imp_block;
    Parameter<bool> shear_bulk_corr_block;
    Parameter<bool> energyMomentumTensorTraceless;
    Parameter<bool> energyMomentumTensorTracefull;
    Parameter<bool> energyMomentumTensorTracelessTimeSlices;
    Parameter<bool> energyMomentumTensorTracefullTimeSlices;
    Parameter<bool> energyMomentumTensorCorrFunctionsAveragedTau;
    Parameter<bool> energyMomentumTensorCorrFunctionsGeneralTau;
    Parameter<int> binsize; //! the binsize used in the blocking method

    Parameter<bool> PolyakovLoopCorrelator;
    Parameter<floatT> GaugeFixTol;
    Parameter<int> GaugeFixNMax;
    Parameter<int> GaugeFixNUnitarize;

    Parameter<std::string> normFileDir; //! Normalization file needed for correlator class calcuations.

    //! ---------------------------------advanced options---------------------------------------------------------------

    Parameter<bool> use_unit_conf; //! for testing (or benchmarking purposes using fixed stepsize)
    Parameter<bool> save_conf;
    //! ignore start_step_size and integrate to the necessary_flow_times without steps in between.
    //! only useful when using RK_method fixed_stepsize
    Parameter<bool> ignore_fixed_startstepsize;

    gradientFlowParam() {
        addDefault(force, "force", std::string("zeuthen"));

        add(start_step_size, "start_step_size");

        addDefault(RK_method, "RK_method", std::string("adaptive_stepsize"));
        addDefault(accuracy, "accuracy", floatT(1e-5));

        addDefault(binsize, "binsize", 8);

        add(measurements_dir, "measurements_dir");

        addOptional(necessary_flow_times, "necessary_flow_times");
        addOptional(measured_corr_flow_times, "measured_corr_flow_times");
        addDefault(ignore_fixed_startstepsize, "ignore_start_step_size", false);

        addDefault(save_conf, "save_configurations", false);  //! write gauge conf to disk at each flow time

        addDefault(use_unit_conf, "use_unit_conf", false);

        add(measurement_intervall, "measurement_intervall");

        addDefault(useHDF5, "useHDF5", false);

        addDefault(plaquette, "plaquette", true);
        addDefault(clover, "clover", false);
        addDefault(cloverTimeSlices, "cloverTimeSlices", false);
        addDefault(topCharge, "topCharge", false);
        addDefault(topChargeTimeSlices, "topChargeTimeSlices", false);
        addDefault(topCharge_imp, "topCharge_imp", false);
        addDefault(topChargeTimeSlices_imp, "topChargeTimeSlices_imp", false);
        addDefault(topCharge_imp_imp, "topCharge_imp_imp", false);
        addDefault(topChargeTimeSlices_imp_imp, "topChargeTimeSlices_imp_imp", false);
        addDefault(topCharge_imp_block, "topCharge_imp_block", false);
        addDefault(topCharge_imp_imp_block, "topCharge_imp_imp_block", false);
        addDefault(weinberg, "Weinberg", false);
        addDefault(weinbergTimeSlices, "WeinbergTimeSlices", false);
        addDefault(weinberg_imp, "Weinberg_imp", false);
        addDefault(weinbergTimeSlices_imp, "WeinbergTimeSlices_imp", false);
        addDefault(weinberg_imp_imp, "Weinberg_imp_imp", false);
        addDefault(weinbergTimeSlices_imp_imp, "WeinbergTimeSlices_imp_imp", false);
        addDefault(shear_bulk_corr_block, "shear_bulk_corr_block", false);
        addDefault(energyMomentumTensorTraceless, "energyMomentumTensorTraceless", false);
        addDefault(energyMomentumTensorTracefull, "energyMomentumTensorTracefull", false);
        addDefault(energyMomentumTensorTracelessTimeSlices, "energyMomentumTensorTracelessTimeSlices", false);
        addDefault(energyMomentumTensorTracefullTimeSlices, "energyMomentumTensorTracefullTimeSlices", false);
        addDefault(energyMomentumTensorCorrFunctionsAveragedTau, "energyMomentumTensorCorrFunctionsAveragedTau", false);
        addDefault(energyMomentumTensorCorrFunctionsGeneralTau, "energyMomentumTensorCorrFunctionsGeneralTau", false);
        addDefault(ColorElectricCorrTimeSlices_naive, "ColorElectricCorrTimeSlices_naive", false);
        addDefault(ColorElectricCorrTimeSlices_clover, "ColorElectricCorrTimeSlices_clover", false);
        addDefault(ColorMagneticCorrTimeSlices_naive, "ColorMagneticCorrTimeSlices_naive", false);
        addDefault(ColorMagneticCorrTimeSlices_clover, "ColorMagneticCorrTimeSlices_clover", false);
        addDefault(RenormPolyakovSusc, "RenormPolyakovSusc", false);

        addDefault(PolyakovLoopCorrelator, "PolyakovLoopCorrelator", false);
        addDefault(GaugeFixTol, "GaugeFixTol", floatT(1e-6));
        addDefault(GaugeFixNMax, "GaugeFixNMax", 9000);
        addDefault(GaugeFixNUnitarize, "GaugeFixNUnitarize", 20);

        addDefault(normFileDir, "normFileDir", std::string("./"));
    }
};
