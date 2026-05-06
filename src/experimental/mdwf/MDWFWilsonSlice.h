/*
 * Slice-wise 4D Wilson application for the MDWF scaffold.
 *
 * This header applies the existing 4D Wilson kernel independently on every
 * MDWFSpinor stack.  It does not add fifth-direction coupling, clover, solver
 * integration, or MDWF boundary-condition behavior.
 */

#pragma once

#include "MDWFSpinor.h"
#include "../DWilson.h"

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFWilsonSlice(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                          Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_tmp,
                          const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                          floatT mass,
                          floatT csw = 0.0,
                          bool update = false) {
    spinor_tmp.template iterateOverBulk<BLOCKSIZE>(
        gamma5DiracWilson<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(gauge, spinor_in, mass, csw));

    spinor_out.template iterateOverBulk<BLOCKSIZE>(
        gamma5<floatT, All, HaloDepthSpin, Ls>(spinor_tmp));

    if (update) {
        spinor_out.updateAll();
    }
}
