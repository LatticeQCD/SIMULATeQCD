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
