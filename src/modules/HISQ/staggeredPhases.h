//
// Created by Lukas Mazur on 04.01.19.
//
#ifndef STAGGEREDPHASES_H
#define STAGGEREDPHASES_H
#include "../../define.h"
#include "../../base/indexer/BulkIndexer.h"



struct calcStaggeredPhase {
    inline __host__ __device__ int operator()(const gSiteMu &siteMu) const {

        typedef GIndexer<All> GInd;

        sitexyzt localCoord = siteMu.coord;
        /// I think we don't need to compute global coord here..
        sitexyzt globalCoord = GInd::getLatData().globalPos(localCoord);

        // printf("Is this even used?\n");

        int rest = globalCoord.x % 2;
        if (rest == 1 && siteMu.mu == 1) return -1;

        rest = (globalCoord.x + globalCoord.y) % 2;
        if (rest == 1 && siteMu.mu == 2) return -1;

        rest = (globalCoord.x + globalCoord.y + globalCoord.z) % 2;
        if (rest == 1 && siteMu.mu == 3) return -1;


        return 1;
    }
};

/*! For fermi statistics we want anti-periodic boundary conditions in the time-direction
 *
 */
struct calcStaggeredBoundary {
    inline __host__ __device__ int operator()(const gSiteMu &siteMu) const {

        typedef GIndexer<All> GInd;

        sitexyzt localCoord = siteMu.coord;
        sitexyzt globalCoord = GInd::getLatData().globalPos(localCoord);

        if ((globalCoord.t == (int) GInd::getLatData().globLT - 1) && siteMu.mu == 3) return -1;

        return 1;
    }
};

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
    explicit staggeredPhaseKernel(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeIn, floatT _mu_f=0.0) :
            gAcc(gaugeIn.getAccessor()),mu_f(_mu_f) {}

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
