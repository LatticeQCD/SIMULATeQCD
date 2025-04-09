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

    Vect3<floatT> Stmp(0.0);

    #ifdef USE_HIP
    static_for<0,4>::apply([&](auto mu) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorIn.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    });
    #else
    #pragma unroll
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
    #endif
    return Stmp;
}


template<bool onDevice, class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
__host__ __device__ void HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::operator()(gSiteStack site) {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;
    size_t stack_offset = GInd::getStack(site);
        SimpleArray<Vect3<floatT>, NStacks> Stmp((floatT)0.0);
        
        #ifdef USE_HIP
        constexpr size_t Ntiles = NStacks/NStacks_blockdim;
        static_for<0,4>::apply([&](auto mu) {
            size_t stack = stack_offset;
            static_for<0,Ntiles>::apply([&](auto i) {

                Stmp[i] += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                                    * _spinorIn.getElement(GInd::site_up(GInd::getSiteStack(site,stack), mu));

                Stmp[i] -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn(GInd::getSiteStack(site,stack), mu));

                Stmp[i] += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_up_up_up(GInd::getSiteStack(site,stack), mu, mu, mu));

                Stmp[i] -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn_dn_dn(GInd::getSiteStack(site,stack), mu, mu, mu));

                stack += NStacks_blockdim;
            });
        });
        #else
 
        #pragma unroll
        for (int mu = 0; mu < 4; mu++) {
            size_t i = 0;
            #pragma unroll
            for (size_t stack = stack_offset; stack-stack_offset < NStacks; stack+=NStacks_blockdim, i++) {

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
        #endif
    

    // size_t i = 0;
    // #pragma unroll NStacks
    // for (size_t stack = stack_offset; i < Ntiles; stack+=NStacks_blockdim, i++) {
 
    #ifdef USE_HIP
    size_t stack = stack_offset;
    static_for<0,Ntiles>::apply([&](auto i) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,Stmp[i]);
        stack += NStacks_blockdim;
    });
    #else
    size_t i = 0;
    #pragma unroll NStacks
    for (size_t stack = stack_offset; stack-stack_offset < NStacks; stack+=NStacks_blockdim, i++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,Stmp[i]);
    }   
    #endif
}


template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqMdaggMFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site){
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);
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

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    if (NStacks * NStacks_blockdim == 1) {
        Dslash_nostack(lhs,rhs,update);
    }
    else {
        if (NStacks % NStacks_blockdim != 0) {
            throw std::runtime_error(stdLogger.fatal("Error in Dslash call: Nstacks not divisible by NStacks_blockdim!"));
        }
        Dslash_stacked(lhs,rhs,update);
    }
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_nostack(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    // The getFunctor calls the DSlash functor. This is to clean up the DSlash functor call.
    lhs.template iterateOverBulk<BLOCKSIZE>(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_stacked(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
   
    HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim> dslash_func(lhs, rhs,_gauge_smeared,_gauge_Naik,_c_3000);
   
    CalcGSiteStack<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> calcGSite;
   
    iterateFunctorNoReturn<onDevice,BLOCKSIZE>(dslash_func,calcGSite,lhs.getNumberLatticePoints(),NStacks_blockdim);

    if(update) {
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_center(SpinorLHS_t& lhs, const SpinorRHS_t& rhs) {
    lhs.template iterateOverCenter<BLOCKSIZE>(getFunctor(rhs));
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_halo(SpinorLHS_t& lhs, const SpinorRHS_t& rhs) {
    lhs.template iterateOverHalo<BLOCKSIZE>(getFunctor(rhs));
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_concurrent_comms(SpinorLHS_t &lhs, SpinorRHS_t &rhs) {
    rhs.updateAll(COMM_START | Hyperplane);
    Dslash_center(lhs,rhs);
    rhs.updateAll(COMM_FINISH | Hyperplane);
    Dslash_halo(lhs,rhs);
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM_concurrent_comms(SpinorRHS_t &spinorOut, SpinorRHS_t &spinorIn) {
    Dslash_concurrent_comms(_tmpSpin, spinorIn);

    _tmpSpin.updateAll(COMM_START | Hyperplane);
    if(_mass != 0.0) {
        spinorOut.template iterateOverCenter<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    } else {
        spinorOut.template iterateOverCenter<BLOCKSIZE>(getFunctor(_tmpSpin));
    }
    _tmpSpin.updateAll(COMM_FINISH | Hyperplane);

    if(_mass != 0.0) {
        spinorOut.template iterateOverHalo<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    } else {
        spinorOut.template iterateOverHalo<BLOCKSIZE>(getFunctor(_tmpSpin));
    }
    gpuDeviceSynchronize();
    
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM_nostack(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

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


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM_stacked(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){
    Dslash(_tmpSpin, spinorIn, true);

    HisqDslashStackedFunctor<onDevice, floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim> dslash_2nd(spinorOut,_tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000);
    CalcGSiteStack<LatLayoutRHS, HaloDepthSpin> calcGSite;

    iterateFunctorNoReturn<onDevice,BLOCKSIZE>(dslash_2nd,calcGSite,spinorOut.getNumberLatticePoints(),NStacks_blockdim);
    if (_mass != 0.0) {
        spinorOut = (spinorIn * _mass2) - spinorOut;
    }

    if(update) {
        spinorOut.updateAll();
    }

}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){
    if (NStacks * NStacks_blockdim == 1) {
        applyMdaggM_nostack(spinorOut,spinorIn,update);
    }
    else {
        if (NStacks % NStacks_blockdim != 0) {
            throw std::runtime_error(stdLogger.fatal("Error in Dslash call: Nstacks not divisible by NStacks_blockdim!"));
        }
        applyMdaggM_stacked(spinorOut, spinorIn, update);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
template<Layout LatLayout>
HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge_smeared, _gauge_Naik, _c_3000);
}


//! stdStagDslash
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto stdStagDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);
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

#define DSLASH_INIT2(floatT, LO, HaloDepth, HaloDepthSpin, NStacks, NStacks_blockdim) \
  template class HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks, NStacks_blockdim>;
INIT_PLHHSNNB_HALF(DSLASH_INIT2)

