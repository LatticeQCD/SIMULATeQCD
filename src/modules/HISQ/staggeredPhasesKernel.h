/*
 * staggeredPhasesKernel.h
 *
 * L. Mazur
 *
 */

#ifndef STAGGEREDPHASES_KERNEL_H
#define STAGGEREDPHASES_KERNEL_H

#include "../../gauge/gaugefield.h"
#include "staggeredPhases.h"

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct staggeredPhaseKernel {

    //Member variables to hold all information needed
    gaugeAccessor<floatT, comp> gAcc;

    calcStaggeredPhase staggPhase;
    calcStaggeredBoundary staggBound;
    imagMuphase<floatT> imaginaryPhase;
    double mu_f;

    //Constructor to initialize this member variable.
    explicit staggeredPhaseKernel(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeIn, floatT _mu_f=0.0)
        : gAcc(gaugeIn.getAccessor()),mu_f(_mu_f) {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu siteMu) {
        GCOMPLEX(floatT) phase =1.0;
        if (mu_f == 0 )
            phase = staggPhase(siteMu) * staggBound(siteMu);
        else
            phase = staggPhase(siteMu) * staggBound(siteMu) * imaginaryPhase(siteMu, mu_f);
        return phase * gAcc.getLink(siteMu);
    }
};

#endif //STAGGEREDPHASES_H
