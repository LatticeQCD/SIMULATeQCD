/* 
 * main_ploop.cpp                                                               
 * 
 * v1.0: D. Clarke, 30 Oct 2018
 * 
 * Measure Polyakov loops using the multi-GPU framework. Initialization copied from main_plaquette.cpp. This is a good
 * example of a simple operator calculated on spatial sites only.
 * 
 */

#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../../base/microtimer.h"
//#include "../base/Reductionbase.h"
#include "../../base/LatticeContainer.h"
#include "../../gauge/GaugeAction.h"
#include "../../spinor/spinorfield.h"
#include "../../base/math/simpleArray.h"

#include "../../modules/rhmc/rhmcParameters.h"
#include "../../modules/gaugeFixing/gfix.h"

#include "../../base/math/correlators.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
using namespace std;

#include "WilsonLineCorrelatorMultiGPU.h"


template<class floatT, size_t HaloDepth,int stacks>
GCOMPLEX(floatT) WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gDotAlongXY( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,GCOMPLEX(floatT)> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
    const size_t elems = GInd::getLatData().vol3;
    redBase.adjustSize(elems);

    /// main call in this function
    redBase.template iterateOverSpatialBulk<All, HaloDepth>(DotAlongXYInterval<floatT,HaloDepth,All>(gauge,shiftx,shifty));

    /// Do the final reduction.
    GCOMPLEX(floatT) val;
    redBase.reduce(val, elems);

    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT vol=GInd::getLatData().globvol4;

    // normalize
    val /= (vol);

    return val;
};

template<class floatT, size_t HaloDepth,int stacks>
std::vector<floatT> WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gDotAlongXYStacked( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
    const size_t elems = GInd::getLatData().vol3;
    redBase.adjustSize(elems*stacks);

    /// main call in this function
    redBase.template iterateOverSpatialBulk<All, HaloDepth>(DotAlongXYIntervalStacked<floatT,HaloDepth,All,stacks>(redBase,gauge,shiftx,shifty));


    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT vol=GInd::getLatData().globvol4;


    std::vector<floatT> result;
    redBase.reduceStacked(result, stacks, elems,false);
    for (size_t i = 0; i < result.size(); i++){
        result[i] = result[i]/(vol);
    }
    return result;
};



template<class floatT, size_t HaloDepth,int stacks>
GCOMPLEX(floatT) WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gDotAlongXYFull( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,GCOMPLEX(floatT)> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
    const size_t elems = GInd::getLatData().vol4;
    redBase.adjustSize(elems);

    /// main call in this function
    redBase.template iterateOverBulk<All, HaloDepth>(DotAlongXYIntervalFull<floatT,HaloDepth,All>(gauge,shiftx,shifty));

    /// Do the final reduction.
    GCOMPLEX(floatT) val;
    redBase.reduce(val, elems);

    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT vol=GInd::getLatData().globvol4;

    // normalize
    val /= (vol);

    return val;
};


/// Function to compute the wilson line using the above struct CalcWilson.
template<class floatT, size_t HaloDepth,int stacks>
void WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gWilson(Gaugefield<floatT,true,HaloDepth> &gauge , size_t length){

    gauge.template iterateOverBulkAtMu<0,256>(CalcWilson<floatT,HaloDepth>(gauge,length));
//    rootLogger.info(spinor.dotProduct(spinor));
    return;

}


template<class floatT, size_t HaloDepth,int stacks>
void WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gMoveOne( Gaugefield<floatT,true,HaloDepth> &gauge , int direction, int up){

    /// move gauge field in mu=1 specified direction  and save it into mu=2
    if(direction == 0){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,0,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,0,false>(gauge));
        }            
    }

    if(direction == 1){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,1,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,1,false>(gauge));
        }
    }

    if(direction == 2){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,2,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,2,false>(gauge));
        }
    }

    // copy back from mu=2 to mu=1
    gauge.template iterateOverBulkAtMu<1,256>(CopyFromMu<floatT,HaloDepth,All,2>(gauge));


    return;

}

template<class floatT,size_t HaloDepth,Layout LatLayout, int sharedX>
__global__ void DotAlongXYIntervalStackedShared(MemoryAccessor _redBase, gaugeAccessor<floatT> _gaugeIn,int _shifty, const size_t size_x) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size_x) {
        return;
    }

    __shared__ GSU3<floatT> Links_shared[sharedX];

    typedef GIndexer<All,HaloDepth> GInd;


    int ix, iy, iz, it;
    it = 0;

    int  tmp;

    divmod(site, GInd::getLatData().vol2, iz, tmp);
    divmod(tmp,  GInd::getLatData().vol1, iy, ix);

//    ix = ix*2;
//    if(ix >= (int)GInd::getLatData().lx){
//        ix += 1-(int)GInd::getLatData().lx;
//    }

        typedef GIndexer<All,HaloDepth> GInd;

        int ix_shift[sharedX];
        floatT results[sharedX];

        /// Get coordinates.
        int iy_shift=iy+_shifty;


        for(int j =0; j < sharedX ; j++){
           ix_shift[j]=ix+j;
           results[j] = 0.0;

            if(ix_shift[j] >= (int)GInd::getLatData().lx){
                ix_shift[j] -=(int)GInd::getLatData().lx;
            }

            if(ix_shift[j] < 0){
                ix_shift[j] +=(int)GInd::getLatData().lx;
            }
        }

        if(iy_shift >= (int)GInd::getLatData().ly){
             iy_shift -=(int)GInd::getLatData().ly;
        }

        if(iy_shift < 0){
             iy_shift +=(int)GInd::getLatData().ly;
        }


        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += 1){


               GSU3<floatT> su3Temp = _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSite((size_t)ix,(size_t)iy, (size_t)iz, (size_t)(it+tt)),0));
               Links_shared[ix]     = _gaugeIn.getLink(GInd::getSiteMu(GInd::getSite((size_t)ix,(size_t)iy_shift, (size_t)iz, (size_t)(it+tt)),1));

               __syncthreads();

               for(int j =0; j < sharedX ; j++){
                   results[j] = results[j] +  tr_c(su3Temp*Links_shared[ix_shift[j]]).cREAL;
               }
               __syncthreads();
        }

        for(int j =0; j < sharedX ; j++){
            _redBase.setValue<floatT>(site+j*GInd::getLatData().vol3, results[j]/3.0);
        }


}

template<class floatT, size_t HaloDepth,int stacks>
std::vector<floatT> WilsonLineCorrelatorMultiGPU<floatT,HaloDepth,stacks>::gDotAlongXYStackedShared(
                        Gaugefield<floatT,true,HaloDepth> &gauge , int shifty,  LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    if(stacks == GInd::getLatData().lx){

        /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
        const size_t elems = GInd::getLatData().vol3;
        redBase.adjustSize(elems*stacks);

        /// main call in this function
        dim3 blockDim;

        blockDim.x = stacks;
        blockDim.y = 1;
        blockDim.z = 1;

        //Grid only in x direction!
        const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));

        DotAlongXYIntervalStackedShared<floatT,HaloDepth,All,stacks><<< gridDim, blockDim>>> (redBase.getAccessor(),gauge.getAccessor(),shifty, elems);


        /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
        floatT vol=GInd::getLatData().globvol4;

        std::vector<floatT> result;
        redBase.reduceStacked(result, stacks, elems,false);
        for (size_t i = 0; i < result.size(); i++){
            result[i] = result[i]/(vol);
        }
        return result;

    }
    else{
        rootLogger.info("ERROR, Wilson Line Shared only works with stacks equal to lx");
        std::vector<floatT> result;
        return result;
    }

};



#define CLASS_INIT2(floatT, HALO) \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,1>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,2>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,4>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,8>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,16>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,96>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,20>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,32>; \
template class WilsonLineCorrelatorMultiGPU<floatT,HALO,48>; \


INIT_PH(CLASS_INIT2)


