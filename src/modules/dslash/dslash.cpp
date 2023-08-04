#include "dslash.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif



//! HisqDslash
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);


#ifdef USE_CUDA
#pragma unroll
#endif
    for (int mu = 0; mu < 4; mu++) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorIn.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    }
    return Stmp;
}

//! HisqDslash - stack loop
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSite site, size_t loopidx) {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);


#ifdef USE_CUDA
#pragma unroll
#endif
    for (int mu = 0; mu < 4; mu++) {        

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorIn.getElement(GInd::site_up(GInd::getSiteStack(site,loopidx), mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn(GInd::getSiteStack(site,loopidx), mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_up_up_up(GInd::getSiteStack(site,loopidx), mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn_dn_dn(GInd::getSiteStack(site,loopidx), mu, mu, mu));
    
    }
    return Stmp;
}

template<bool onDevice, class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_cached>
__host__ __device__ void HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_cached>::operator()(gSiteStack site) {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;
    
    size_t stack_offset = GInd::getStack(site);
        SimpleArray<gVect3<floatT>, NStacks> Stmp(0.0);
        #ifdef USE_CUDA
        #pragma unroll
        #endif
        for (int mu = 0; mu < 4; mu++) {

     
            #pragma unroll
            for (auto [stack,i] = std::tuple{stack_offset, 0}; i < Nstacks; stack+=NStacks_cached, i++) {

                Stmp[i] += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                                    * _spinorIn.getElement(GInd::site_up(GInd::getSiteStack(site,stack), mu));

                Stmp[i] -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn(GInd::getSiteStack(site,stack), mu));

                Stmp[i] += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_up_up_up(GInd::getSiteStack(site,stack), mu, mu, mu));

                Stmp[i] -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn_dn_dn(GInd::getSiteStack(site,stack), mu, mu, mu));
            }
 
        
    }

    for (auto [stack,i] = std::tuple{stack_offset, 0}; i < NStacks; stack+=NStacks_cached, i++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,Stmp[i]);
 
    }
    
}


template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqMdaggMFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site){
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);
    for (int mu = 0; mu < 4; mu++) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorTmp.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorTmp.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorTmp.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorTmp.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    }
    return _spinorIn.getElement(site)*_mass2 - Stmp;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    // The getFunctor calls the DSlash functor. Presumably this is to clean up the DSlash functor call.
    lhs.template iterateOverBulk<BLOCKSIZE>(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash_stackloop(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    // The getFunctor calls the DSlash functor. Presumably this is to clean up the DSlash functor call.
    lhs.template iterateOverBulkLoopStack<BLOCKSIZE>(getFunctor(rhs),NStacks);
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<size_t NStacks_cache>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash_stacked(Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks*NStacks_cache> &lhs, const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks*NStacks_cache>& rhs, bool update){
   
    HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_cache> dslash_func(lhs, rhs,_gauge_smeared,_gauge_Naik,_c_3000);
   
    CalcGSiteStack<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> calcGSite;
   
    iterateFunctorNoReturn<onDevice,BLOCKSIZE>(dslash_func,calcGSite,lhs.getNumberLatticePoints(),NStacks);

    if(update) {
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

    Dslash(_tmpSpin, spinorIn, true);

    // Optimization: We might get a speed up if we put this in a custom operator
    if(_mass != 0.0) {
        spinorOut.template iterateOverBulk<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    } else {
        spinorOut.template iterateOverBulk<BLOCKSIZE>(getFunctor(_tmpSpin));
    }

    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<Layout LatLayout>
HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge_smeared, _gauge_Naik, _c_3000);
}


//! stdStagDslash
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto stdStagDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);
    floatT phase =1.0;
    floatT up_bound=1.0;
    floatT down_bound=1.0;

    for (int mu = 0; mu < 4; mu++) {

        int rest=site.coord.x%2;
        if (rest == 1 && mu == 1) phase= -1.0;

        rest = (site.coord.x + site.coord.y) % 2;
        if (rest == 1 && mu == 2) phase= -1.0;

        rest = (site.coord.x + site.coord.y + site.coord.z) % 2;
        if (rest == 1 && mu == 3) phase= -1.0;

        sitexyzt localCoord = site.coord;
        sitexyzt globalCoord = GInd::getLatData().globalPos(localCoord);

        if(mu==3 && (globalCoord.t == (int) GInd::getLatData().globLT - 1)) up_bound = -1.0;
        if(mu==3 && (globalCoord.t == 0)) down_bound = -1.0;

        Stmp += static_cast<floatT>(C_1000) * phase * up_bound *_gAcc.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
            _spinorIn.getElement(GInd::site_up(site, mu));
        Stmp -= static_cast<floatT>(C_1000) * phase * down_bound * _gAcc.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
            _spinorIn.getElement(GInd::site_dn(site, mu));
    }
    return Stmp;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash(
        SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    lhs.template iterateOverBulk(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

    Dslash(_tmpSpin, spinorIn, true);
    // This is not wrong! One Dslash is hiden in the line below in the getFunctor() call!
    // Optimization: We might get a speed up if we put this in a custom operator
    spinorOut.template iterateOverBulk(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<Layout LatLayout>
stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge);
}



//! explicit template instantiations
#define DSLASH_INIT(floatT, LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class HisqDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>;
INIT_PLHHSN(DSLASH_INIT)

#define DSLASH_INIT_HALF(floatT, LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template void HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>::Dslash_stacked<1>(Spinorfield<floatT, true, LayoutSwitcher<LO>(), HaloDepthSpin, NStacks>&, const Spinorfield<floatT, true, LO, HaloDepthSpin, NStacks>&, bool);\
  template void HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>::Dslash_stacked<2>(Spinorfield<floatT, true, LayoutSwitcher<LO>(), HaloDepthSpin, NStacks*2>&, const Spinorfield<floatT, true, LO, HaloDepthSpin, NStacks*2>&, bool);\
  template void HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>::Dslash_stacked<3>(Spinorfield<floatT, true, LayoutSwitcher<LO>(), HaloDepthSpin, NStacks*3>&, const Spinorfield<floatT, true, LO, HaloDepthSpin, NStacks*3>&, bool);\
  template void HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>::Dslash_stacked<4>(Spinorfield<floatT, true, LayoutSwitcher<LO>(), HaloDepthSpin, NStacks*4>&, const Spinorfield<floatT, true, LO, HaloDepthSpin, NStacks*4>&, bool);
INIT_PLHHSN_HALF(DSLASH_INIT_HALF)
