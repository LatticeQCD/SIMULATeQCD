/*
 * gfix.h
 *
 * D. Clarke
 *
 * Header file with function definitions for main_gfix.
 *
 */

#pragma once
#define I_FIX 3   /// 3=coulomb, 4=landau
#define D_FIX 3.0

#include "../../define.h"
#include "../../gauge/gaugefield.h"
//<<<<<<< HEAD
//#include "../../base/LatticeContainer.h"
//#include "../../base/math/gsu2.h"
//#include "../../base/math/matrix4x4_notSym.h"
//=======
#include "../../base/latticeContainer.h"
#include "../../base/math/su2.h"
#include "../../base/math/matrix4x4_notSym.h"
//>>>>>>> origin/main

/// Class for gauge fixing functions. For now this only includes simple functions that calculate the gauge fixing
/// action and theta, but will include everything else at some later point. Specify whether it is Coulomb or
/// Landau gauge fixing by the defines at the top.
template<class floatT, bool onDevice, size_t HaloDepth>
class GaugeFixing {
protected:
    LatticeContainer<onDevice,floatT> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t elems   = GInd::getLatData().vol4;
    const size_t ORelems = GInd::getLatData().sizehFull;

public:
    GaugeFixing(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield) :
        _redBase(gaugefield.getComm()),_gauge(gaugefield) {
            _redBase.adjustSize(GInd::getLatData().vol4);
        }

    floatT getAction();       /// Calculate gauge fixing functional
    floatT getTheta();        /// Calculate gauge fixing theta
    void   gaugefixOR();      /// One gauge fixing step for the lattice. Make sure you unitarize every so often...

    ///// R fixing
    floatT getR();        /// Calculate gauge fixing R
    void   gaugefixR();      /// One gauge fixing step for the lattice. Make sure you unitarize every so often...
    void projectZ(Gaugefield<floatT,onDevice,HaloDepth> &gauge2);

};

/// Even/odd read index
template<Layout LatLayout, size_t HaloDepth>
struct gfixReadIndexEvenOddFull {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        gSite site = GInd::getSiteFull(i);
        return site;
    }
};

