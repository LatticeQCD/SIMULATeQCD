/* 
 * PureGaugeUpdates.cpp                                                               
 * 
 * v1.0: D. Clarke, 1 Feb 2019
 * 
 * Some methods to update pure gauge systems.
 * 
 */

#include "PureGaugeUpdates.h"

/// Kernel to perform over-relaxation. Runs over sites, and loops over directions.
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct ORKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    uint8_t _mu;
    ORKernel(Gaugefield<floatT,true,HaloDepth> &gauge,uint8_t mu) : gaugeAccessor(gauge.getAccessor()), _mu(mu){}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        GSU3<floatT> U,Uprime,Ustaple;

        U = gaugeAccessor.getLink(GInd::getSiteMu(site,_mu));                  /// Original link
        Ustaple=SU3Staple<floatT,LatLayout,HaloDepth>(gaugeAccessor,site,_mu); /// Staple attached to it
        /// Perform the OR update.
        Uprime=OR_GPUSU3(U,Ustaple);
        Uprime.su3unitarize();
        /// Replace the link.
        gaugeAccessor.setLink(GInd::getSiteMu(site,_mu),Uprime);
    }
};

/// Kernel to perform heatbath. Runs over sites, and loops over directions.
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct HBKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    uint4*  _state;
    floatT  _beta;
    uint8_t _mu;
    bool    _ltest;
    HBKernel(Gaugefield<floatT,true,HaloDepth> &gauge,uint4* state,floatT beta, uint8_t mu, bool ltest) :
            gaugeAccessor(gauge.getAccessor()), _state(state), _beta(beta), _mu(mu), _ltest(ltest) {}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        GSU3<floatT> U,Uprime,Ustaple;

        /// The indexing for both even and odd runs from 0 to sizeh. This is a problem for the state array, which
        /// runs from 0 to size. This construction ensures we access the random numbers on each site correctly.
        gSite siteAll=GInd::convertToAll(site);
        uint4* stateElem=&_state[siteAll.isite];

        U = gaugeAccessor.getLink(GInd::getSiteMu(site,_mu));                  /// Original link
        Ustaple=SU3Staple<floatT,LatLayout,HaloDepth>(gaugeAccessor,site,_mu); /// Staple attached to it
        /// Perform the HB update.
        Uprime=HB_GPUSU3(U,Ustaple,stateElem,_beta,_ltest);
        Uprime.su3unitarize();
        /// Replace the link.
        gaugeAccessor.setLink(GInd::getSiteMu(site,_mu),Uprime);
    }
};

/// Checkerboard update using overrelaxation.
template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeUpdate<floatT,onDevice,HaloDepth>::updateOR(){
    ReadIndexEvenOdd<Even,HaloDepth> calcReadIndexEven;
    ReadIndexEvenOdd<Odd, HaloDepth> calcReadIndexOdd;

    for (uint8_t mu=0; mu<4; mu++) {
        /// OR update red sites.
        iterateFunctorNoReturn<onDevice>(ORKernel<floatT,Even,HaloDepth>(_gauge,mu),calcReadIndexEven,elems);
        _gauge.updateAll(Hyperplane | Plane | COMM_BOTH);
        /// OR update black sites.
        iterateFunctorNoReturn<onDevice>(ORKernel<floatT,Odd, HaloDepth>(_gauge,mu),calcReadIndexOdd, elems);
        _gauge.updateAll(Hyperplane | Plane | COMM_BOTH);
    }
}

/// Checkerboard update using heatbath.
template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeUpdate<floatT,onDevice,HaloDepth>::updateHB(uint4* state, floatT beta, bool ltest){
    ReadIndexEvenOdd<Even,HaloDepth> calcReadIndexEven;
    ReadIndexEvenOdd<Odd, HaloDepth> calcReadIndexOdd;
    for (uint8_t mu=0; mu<4; mu++) {
        /// HB update red sites.
        iterateFunctorNoReturn<onDevice>(HBKernel<floatT,Even,HaloDepth>(_gauge,state,beta,mu,ltest),
                                                calcReadIndexEven,elems);
        _gauge.updateAll(Hyperplane | Plane | COMM_BOTH);
        /// HB update black sites.
        iterateFunctorNoReturn<onDevice>(HBKernel<floatT,Odd, HaloDepth>(_gauge,state,beta,mu,ltest),
                                                calcReadIndexOdd, elems);
        _gauge.updateAll(Hyperplane | Plane | COMM_BOTH);
    }
}

template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeUpdate<floatT, onDevice, HaloDepth>::set_gauge_to_reference() {
    rootLogger.info("Calculating reference gaugefield: start from U=1, apply 100 HB updates with beta=8.0 using random "
                    "numbers generated with seed=0.");
    _gauge.one();
    grnd_state<false> host_state;
    grnd_state<true> dev_state;
    host_state.make_rng_state(0);
    dev_state = host_state;
    for (int i = 0; i < 100; ++i) {
        this->updateHB(dev_state.state, 8.0);
    }
}

///initialize various instances of the class
#define CLASS_INIT(floatT,HALO, comp) \
template class GaugeUpdate<floatT,true,HALO>;
INIT_PHC(CLASS_INIT)
