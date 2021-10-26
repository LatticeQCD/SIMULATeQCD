//
// Created by Lukas Mazur on 04.01.19.
//
#ifndef STAGGEREDPHASES_H
#define STAGGEREDPHASES_H

#include "../../gauge/gaugefield.h"
#include "staggeredPhases.cuh"

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct staggeredPhaseKernel {

    //Member variables to hold all information needed
    gaugeAccessor<floatT, comp> gAcc;

    calcStaggeredPhase staggPhase;
    calcStaggeredBoundary staggBound;

    //Constructor to initialize this member variable.
    explicit staggeredPhaseKernel(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu siteMu) {
        floatT phase = staggPhase(siteMu) * staggBound(siteMu);
        return phase * gAcc.getLink(siteMu);
    }
};

#endif //STAGGEREDPHASES_H
