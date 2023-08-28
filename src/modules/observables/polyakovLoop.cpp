/*
 * polyakovLoop.h
 *
 * L. Altenkort, 29 Jan 2019
 *
 */

#include "polyakovLoop.h"

/// Kernel to calculate the PolyakovLoop (at each space point)
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct PolyakovLoopKernel{
    SU3Accessor<floatT, comp> SU3Accessor;

    PolyakovLoopKernel(Gaugefield<floatT,onDevice,HaloDepth, comp> &gauge) : SU3Accessor(gauge.getAccessor()){
    }

    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        const int Ntau  = GInd::getLatData().lt;
        /// Start off at this site, pointing in N_tau direction.
        SU3<floatT> tmp = SU3Accessor.getLink(GInd::getSiteMu(site, 3));
        /// Loop over N_tau direction.
        for(int tau = 1; tau < Ntau; tau++){
            site = GInd::site_up(site,3);
            tmp *= SU3Accessor.getLink(GInd::getSiteMu(site, 3));
        }
        return tr_c(tmp) / 3; /// tr(unity matrix)=3
    }
};

/// Kernel to compute untraced, unnormalized Polyakov loop and store in the array _ploop.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct CalcPloopKernel{
    SU3Accessor<floatT, comp> SU3Accessor;
    MemoryAccessor _ploop;
    CalcPloopKernel(Gaugefield<floatT,onDevice,HaloDepth, comp> &_gauge, MemoryAccessor ploop) :
            SU3Accessor(_gauge.getAccessor()), _ploop(ploop) {}
    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        const int Ntau=GInd::getLatData().lt;
        size_t       pind = site.isite;
        SU3<floatT> temp = SU3Accessor.getLink(GInd::getSiteMu(site, 3));
        for(int tau = 1; tau < Ntau; tau++){
            site = GInd::site_up(site,3);
            temp*= SU3Accessor.getLink(GInd::getSiteMu(site, 3));
        }
        _ploop.setValue<SU3<floatT>>(pind,temp);
    }
};

/// Call this to get the PolyakovLoop. Don't forget to exchange halos before this!
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
COMPLEX(floatT) PolyakovLoop<floatT, onDevice, HaloDepth, comp>::getPolyakovLoop() {
    /// Exit if lattice is split in time
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }
    COMPLEX(floatT) poly_result;
    _redBase.template iterateOverSpatialBulk<All, HaloDepth>(PolyakovLoopKernel<floatT, onDevice, HaloDepth, comp>(_gauge));
    _redBase.reduce(poly_result, elems);
    return poly_result / spatialvol; /// normalize to GLOBAL lattice
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
