//
// Created by Lukas Mazur on 04.01.19.
//
#ifndef STAGGEREDPHASES_H
#define STAGGEREDPHASES_H

#include "../../gauge/gaugefield.h"
#include "staggeredPhases.cuh"

template <class floatT>
struct imagMuphase {
    inline __host__ __device__ GPUcomplex<floatT> operator()(const gSiteMu &siteMu, double chmp) const {

        typedef GIndexer<All> GInd;
        GPUcomplex<floatT> img_chmp;


        if (chmp>=0) {
            img_chmp.cREAL = cos(chmp);
            img_chmp.cIMAG = sin(chmp);
        }
        else {
            chmp=-chmp;
            img_chmp.cREAL = cos(chmp); // For dagger exp(-i*mu)
            img_chmp.cIMAG = -sin(chmp);
        }


        if ( siteMu.mu == 3) return img_chmp;

        return 1;
    }
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct staggeredPhaseKernel {

    //Member variables to hold all information needed
    gaugeAccessor<floatT, comp> gAcc;

    calcStaggeredPhase staggPhase;
    calcStaggeredBoundary staggBound;
    imagMuphase<floatT> imaginaryPhase;
    double mu_f;

    //Constructor to initialize this member variable.
    explicit staggeredPhaseKernel(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeIn, floatT _mu_f) :
            gAcc(gaugeIn.getAccessor()),mu_f(_mu_f) {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu siteMu) {
        GCOMPLEX(floatT) phase =1.0;
        if (mu_f != 0.0) {
            phase = staggPhase(siteMu) * staggBound(siteMu) * imaginaryPhase(siteMu, mu_f);
        } else {
            phase = staggPhase(siteMu) * staggBound(siteMu);
        }
        return phase * gAcc.getLink(siteMu);
    }
};

#endif //STAGGEREDPHASES_H
