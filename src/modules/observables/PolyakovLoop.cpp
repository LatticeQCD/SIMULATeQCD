/*
 * PolyakovLoop.cpp
 *
 * L. Altenkort, 29 Jan 2019
 *
 */

#include "PolyakovLoop.h"

/// Kernel to calculate the PolyakovLoop (at each space point)
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct PolyakovLoopKernel{
    gaugeAccessor<floatT, comp> gaugeAccessor;

    PolyakovLoopKernel(Gaugefield<floatT,onDevice,HaloDepth, comp> &gauge) : gaugeAccessor(gauge.getAccessor()){
    }

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        const int Ntau = GInd::getLatData().lt;
        /// Start off at this site, pointing in N_tau direction.
        GSU3<floatT> tmp = gaugeAccessor.getLink(GInd::getSiteMu(site, 3));
        /// Loop over N_tau direction.
        for(int tau = 1; tau < Ntau; tau++){
            site = GInd::site_up(site,3);
            tmp *= gaugeAccessor.getLink(GInd::getSiteMu(site, 3));
        }
        return tr_c(tmp) / 3; /// tr(unity matrix)=3
    }
};

/// Kernel to compute thermal Wilson line and store in the array _ploop.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct CalcPloopKernel{
    gaugeAccessor<floatT, comp> gaugeAccessor;
    MemoryAccessor _ploop;
    CalcPloopKernel(Gaugefield<floatT,onDevice,HaloDepth, comp> &_gauge, MemoryAccessor ploop) :
            gaugeAccessor(_gauge.getAccessor()), _ploop(ploop) {}
    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        const int    Ntau = GInd::getLatData().lt;
        size_t       pind = site.isite;
        GSU3<floatT> temp = gaugeAccessor.getLink(GInd::getSiteMu(site, 3));
        for(int tau = 1; tau < Ntau; tau++){
            site = GInd::site_up(site,3);
            temp*= gaugeAccessor.getLink(GInd::getSiteMu(site, 3));
        }
        _ploop.setValue<GSU3<floatT>>(pind,temp);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
GCOMPLEX(floatT) PolyakovLoop<floatT, onDevice, HaloDepth, comp>::getPolyakovLoop() {
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }
    GCOMPLEX(floatT) result;
    _redBase.template iterateOverSpatialBulk<All, HaloDepth>(PolyakovLoopKernel<floatT, onDevice, HaloDepth, comp>(_gauge));
    _redBase.reduce(result, elems);
    return result / spatialvol; /// normalize to GLOBAL lattice
}

/// Function calculates Polyakov loop at each spatial site and stores the result at that site in _ploop array.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void PolyakovLoop<floatT, onDevice, HaloDepth, comp>::PloopInArray(MemoryAccessor _ploop){
    /// Exit if lattice is split in time
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }
    ReadIndexSpatial<HaloDepth> calcReadIndexSpatial;
    iterateFunctorNoReturn<onDevice>(CalcPloopKernel<floatT,onDevice,HaloDepth, comp>(_gauge,_ploop),
                                            calcReadIndexSpatial,elems);
}

/// Explicitly instantiate various instances of the class.
#define INIT_ONDEVICE_TRUE(floatT, HALO, comp) \
template class PolyakovLoop<floatT,true,HALO, comp>;
#define INIT_ONDEVICE_FALSE(floatT,HALO, comp) \
template class PolyakovLoop<floatT,false,HALO, comp>;

INIT_PHC(INIT_ONDEVICE_TRUE)
INIT_PHC(INIT_ONDEVICE_FALSE)
