/*
 * Minimal MDWF operator skeleton.
 *
 * This header combines the already-separated Stage 2 fifth-direction coupling
 * and Stage 3 slice-wise 4D Wilson application.  Coefficients are passed
 * explicitly.  There is no solver integration and no hidden boundary-condition
 * or clover behavior.
 */

#pragma once

#include "MDWFFifthDim.h"
#include "MDWFWilsonSlice.h"

#include <string>

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                       Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                       MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_part,
                       MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &fifth_part,
                       MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                       const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                       MDWFFifthDimCoefficients<floatT> fifth_coeff,
                       floatT mass,
                       floatT csw = 0.0,
                       bool update = false) {
    applyMDWFWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        wilson_part, gauge, wilson_tmp, spinor_in, mass, csw);

    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        fifth_part, spinor_in, fifth_coeff);

    spinor_out = wilson_part;
    spinor_out += fifth_part;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFCloverOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                             Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_part,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &fifth_part,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                             Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_upper,
                             Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_lower,
                             Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_upper,
                             Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_lower,
                             const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                             MDWFFifthDimCoefficients<floatT> fifth_coeff,
                             floatT mass,
                             floatT csw = 0.0,
                             bool update = false) {
    applyMDWFCloverWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        wilson_part, gauge, wilson_tmp, fmunu_upper, fmunu_lower, fmunu_inv_upper, fmunu_inv_lower,
        spinor_in, mass, csw);

    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        fifth_part, spinor_in, fifth_coeff);

    spinor_out = wilson_part;
    spinor_out += fifth_part;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using CloverField = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _wilson_part;
    Spinor _fifth_part;
    Spinor _wilson_tmp;
    CloverField _fmunu_upper;
    CloverField _fmunu_lower;
    CloverField _fmunu_inv_upper;
    CloverField _fmunu_inv_lower;

public:
    explicit MDWFOperatorWorkspace(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                   std::string name = "MDWF_operator_workspace")
        : _gauge(gauge),
          _wilson_part(gauge.getComm(), name + "_wilson_part"),
          _fifth_part(gauge.getComm(), name + "_fifth_part"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _fmunu_upper(gauge.getComm(), name + "_fmunu_upper"),
          _fmunu_lower(gauge.getComm(), name + "_fmunu_lower"),
          _fmunu_inv_upper(gauge.getComm(), name + "_fmunu_inv_upper"),
          _fmunu_inv_lower(gauge.getComm(), name + "_fmunu_inv_lower") {}

    void apply(Spinor &spinor_out,
               const Spinor &spinor_in,
               MDWFFifthDimCoefficients<floatT> fifth_coeff,
               floatT mass,
               bool update = false) {
        applyMDWFOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _wilson_part, _fifth_part, _wilson_tmp,
            spinor_in, fifth_coeff, mass, 0.0, update);
    }

    void applyClover(Spinor &spinor_out,
                     const Spinor &spinor_in,
                     MDWFFifthDimCoefficients<floatT> fifth_coeff,
                     floatT mass,
                     floatT csw = 0.0,
                     bool update = false) {
        applyMDWFCloverOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _wilson_part, _fifth_part, _wilson_tmp,
            _fmunu_upper, _fmunu_lower, _fmunu_inv_upper, _fmunu_inv_lower,
            spinor_in, fifth_coeff, mass, csw, update);
    }
};
