/* 
 * staggeredPhases.h                                                               
 * 
 * L. Mazur 
 * 
 */

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

        int rest = globalCoord.x % 2;
        if (rest == 1 && siteMu.mu == 1) return -1;

        rest = (globalCoord.x + globalCoord.y) % 2;
        if (rest == 1 && siteMu.mu == 2) return -1;

        rest = (globalCoord.x + globalCoord.y + globalCoord.z) % 2;
        if (rest == 1 && siteMu.mu == 3) return -1;

        return 1;
    }
};

// For fermi statistics we want anti-periodic boundary conditions in the time-direction
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

#endif //STAGGEREDPHASES_H
