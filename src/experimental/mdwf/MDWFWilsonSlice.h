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

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFCloverWilsonSlice(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                                Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_tmp,
                                Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_upper,
                                Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_lower,
                                Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_upper,
                                Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_lower,
                                const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                                floatT mass,
                                floatT csw = 0.0,
                                bool update = false) {
    typedef GIndexer<All, HaloDepthGauge> GInd;
    CalcGSite<All, HaloDepthSpin> calcGSite;
    const size_t elems = GInd::getLatData().vol4;

    iterateFunctorNoReturn<true, BLOCKSIZE>(
        preCalcFmunu<floatT, HaloDepthGauge>(
            gauge, fmunu_upper, fmunu_lower, fmunu_inv_upper, fmunu_inv_lower, mass, csw),
        calcGSite, elems);
    fmunu_upper.updateAll();
    fmunu_lower.updateAll();

    spinor_tmp.template iterateOverBulk<BLOCKSIZE>(
        DiracWilsonEvenOdd2<floatT, All, All, HaloDepthGauge, HaloDepthSpin, Ls, false>(
            gauge, spinor_in, 0.0, 0.0));
    spinor_out.template iterateOverBulk<BLOCKSIZE>(
        DiracWilsonEvenEven2<floatT, All, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_in, fmunu_upper, fmunu_lower));

    spinor_out += spinor_tmp;

    if (update) {
        spinor_out.updateAll();
    }
}
