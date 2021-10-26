/* 
 * luscherweisz.cu                                                               
 * 
 * v1.0: Hai-Tao Shu, 10 May 2019
 * 
 * 
 */

#include "luscherweisz.h"
#include "PureGaugeUpdates.h"

template<class floatT,Layout LatLayout,size_t HaloDepth>
struct SubORKernel{
  gaugeAccessor<floatT> gaugeAccessor;
  uint8_t _mu;
  int _sub_lt;
  int _local_pos_t;
  SubORKernel(Gaugefield<floatT,true,HaloDepth> &gauge,uint8_t mu,int sub_lt, int local_pos_t) : gaugeAccessor(gauge.getAccessor()), _mu(mu), _sub_lt(sub_lt), _local_pos_t(local_pos_t){}

  __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<LatLayout,HaloDepth> GInd;
        int Nt = (int)GInd::getLatData().globLT;
 
        sitexyzt coord = site.coord;

        if ( coord[3]%_sub_lt == _local_pos_t && _mu != 3 ) { //not the spatial links on the (left) border. the right border won't be updated anyway
             return;
        }
        else {
             GSU3<floatT> U,Uprime,Ustaple;
             U = gaugeAccessor.getLink(GInd::getSiteMu(site,_mu));                  /// Original link
             Ustaple=SU3Staple<floatT,LatLayout,HaloDepth>(gaugeAccessor,site,_mu); /// Staple attached to it
             /// Perform the OR update.
             Uprime=OR_GPUSU3(U,Ustaple);
             Uprime.su3unitarize();
             /// Replace the link.
             gaugeAccessor.setLink(GInd::getSiteMu(site,_mu),Uprime);
        }
    }
};

/// Kernel to perform heatbath. Runs over sites, and loops over directions.
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct SubHBKernel{
  gaugeAccessor<floatT> gaugeAccessor;
  uint4*  _state;
  floatT  _beta;
  uint8_t _mu;
  int _sub_lt;
  int _local_pos_t;
  bool  _ltest;
  SubHBKernel(Gaugefield<floatT,true,HaloDepth> &gauge,uint4* state,floatT beta, uint8_t mu, int sub_lt, int local_pos_t, bool ltest) :
                                             gaugeAccessor(gauge.getAccessor()), _state(state), _beta(beta), _mu(mu), _sub_lt(sub_lt), _local_pos_t(local_pos_t), _ltest(ltest){}

  __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        int Nt = (int)GInd::getLatData().globLT;
        sitexyzt coord = site.coord;

        if ( coord[3]%_sub_lt == _local_pos_t && _mu != 3 ) { //not the spatial links on the (left) border. the right border won't be updated anyway
             return;
        }
        else {
             GSU3<floatT> U,Uprime,Ustaple;
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
    }
};

/// Checkerboard update using overrelaxation.
template<class floatT, bool onDevice, size_t HaloDepth>
void LuscherWeisz<floatT,onDevice,HaloDepth>::subUpdateOR(int sub_lt, int local_pos_t){
  ReadIndexEvenOdd<Even,HaloDepth> calcReadIndexEven;
  ReadIndexEvenOdd<Odd, HaloDepth> calcReadIndexOdd;

  for (uint8_t mu=0; mu<4; mu++) {
    /// OR update red sites.
    iterateFunctorNoReturn<onDevice>(SubORKernel<floatT,Even,HaloDepth>(_gauge,mu,sub_lt,local_pos_t),calcReadIndexEven,elems);
    _gauge.updateAll();
    /// OR update black sites.
    iterateFunctorNoReturn<onDevice>(SubORKernel<floatT,Odd, HaloDepth>(_gauge,mu,sub_lt,local_pos_t),calcReadIndexOdd, elems);
    _gauge.updateAll();
  }
}

/// Checkerboard update using heatbath.
template<class floatT, bool onDevice, size_t HaloDepth>
void LuscherWeisz<floatT,onDevice,HaloDepth>::subUpdateHB(uint4* state, floatT beta, int sub_lt, int local_pos_t, bool ltest){
  ReadIndexEvenOdd<Even,HaloDepth> calcReadIndexEven;
  ReadIndexEvenOdd<Odd, HaloDepth> calcReadIndexOdd;
  for (uint8_t mu=0; mu<4; mu++) {
    /// OR update red sites.
    iterateFunctorNoReturn<onDevice>(SubHBKernel<floatT,Even,HaloDepth>(_gauge,state,beta,mu,sub_lt,local_pos_t,ltest),
                                            calcReadIndexEven,elems);
    _gauge.updateAll();
    /// OR update black sites.
    iterateFunctorNoReturn<onDevice>(SubHBKernel<floatT,Odd, HaloDepth>(_gauge,state,beta,mu,sub_lt,local_pos_t,ltest),
                                            calcReadIndexOdd, elems);
    _gauge.updateAll();
  }
}

///initialize various instances of the class
#define CLASS_INIT(floatT,HALO, comp) \
template class LuscherWeisz<floatT,true,HALO>; 
INIT_PHC(CLASS_INIT)
