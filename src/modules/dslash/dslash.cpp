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

    /*gVect3<floatT> Stmp(0.0);
#pragma unroll    
    for (int mu = 0; mu < 4; mu++) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
            _spinorIn.getElement(GInd::site_up(site, mu));
        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
            _spinorIn.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu))) *
            _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));
        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu))) *
            _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    }
    return Stmp;*/
    return  static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 0))) *
            _spinorIn.getElement(GInd::site_up(site, 0)) \
        -   static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 0), 0))) *
            _spinorIn.getElement(GInd::site_dn(site, 0)) \
        +   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, 0), 0))) *
            _spinorIn.getElement(GInd::site_up_up_up(site, 0, 0, 0)) \
        -   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, 0, 0), 0))) *
            _spinorIn.getElement(GInd::site_dn_dn_dn(site, 0, 0, 0))  \
    /*1*/+  static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 1))) *
            _spinorIn.getElement(GInd::site_up(site, 1)) \
        -   static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 1), 1))) *
            _spinorIn.getElement(GInd::site_dn(site, 1)) \
        +   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, 1), 1))) *
            _spinorIn.getElement(GInd::site_up_up_up(site, 1, 1, 1)) \
        -   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, 1, 1), 1))) *
            _spinorIn.getElement(GInd::site_dn_dn_dn(site, 1, 1, 1))  \            
    /*2*/+  static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 2))) *
            _spinorIn.getElement(GInd::site_up(site, 2)) \
        -   static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 2), 2))) *
            _spinorIn.getElement(GInd::site_dn(site, 2)) \
        +   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, 2), 2))) *
            _spinorIn.getElement(GInd::site_up_up_up(site, 2, 2, 2)) \
        -   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, 2, 2), 2))) *
            _spinorIn.getElement(GInd::site_dn_dn_dn(site, 2, 2, 2))  \  
    /*3*/+  static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 3))) *
            _spinorIn.getElement(GInd::site_up(site, 3)) \
        -   static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 3), 3))) *
            _spinorIn.getElement(GInd::site_dn(site, 3)) \
        +   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, 3), 3))) *
            _spinorIn.getElement(GInd::site_up_up_up(site, 3, 3, 3)) \
        -   static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, 3, 3), 3))) *
            _spinorIn.getElement(GInd::site_dn_dn_dn(site, 3, 3, 3));            
}

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqMdaggMFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site){
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);
    for (int mu = 0; mu < 4; mu++) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
            _spinorTmp.getElement(GInd::site_up(site, mu));
        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
            _spinorTmp.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu))) *
            _spinorTmp.getElement(GInd::site_up_up_up(site, mu, mu, mu));
        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu))) *
            _spinorTmp.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));

    }
    return _spinorIn.getElement(site)*_mass2 - Stmp;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash(
        SpinorLHS_t& lhs, SpinorRHS_t& rhs, bool update){
    lhs.template iterateOverBulk<BLOCKSIZE>(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, SpinorRHS_t& spinorIn, bool update){

    Dslash(_tmpSpin, spinorIn, true);
    // Optimization: We might get a speed up if we put this in a custom operator
    if(_mass != 0.0){
        spinorOut.template iterateOverBulk<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    }
    else
        spinorOut.template iterateOverBulk<BLOCKSIZE>(getFunctor(_tmpSpin));

    if(update)
        spinorOut.updateAll();
}



//Revised by ranluo
template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM_async(SpinorRHS_t& spinorOut, SpinorRHS_t& spinorIn, bool update)
{
    spinorIn.updateAll_async(COMM_START | Hyperplane);
    _tmpSpin.template iterateOverCenter<BLOCKSIZE>(getFunctor(spinorIn));
    spinorIn.updateAll_async(COMM_FINISH | Hyperplane);
    _tmpSpin.template iterateOverInnerHalo<BLOCKSIZE>(getFunctor(spinorIn));

    _tmpSpin.updateAll_async(COMM_START | Hyperplane);
    if(_mass != 0.0)
    {
        spinorOut.template iterateOverCenter<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
        _tmpSpin.updateAll_async(COMM_FINISH | Hyperplane);
        spinorOut.template iterateOverInnerHalo<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    }
    else
    {
        spinorOut.template iterateOverCenter<BLOCKSIZE>(getFunctor(_tmpSpin));
        _tmpSpin.updateAll_async(COMM_FINISH | Hyperplane);
        spinorOut.template iterateOverInnerHalo<BLOCKSIZE>(getFunctor(_tmpSpin));
    }

    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM_half(SpinorRHS_t& spinorOut, SpinorRHS_t& spinorIn, bool update)
{
    _tmpSpin.template iterateOverBulk_half<BLOCKSIZE>(getFunctor(spinorIn));

    _tmpSpin.updateAll(COMM_BOTH | Hyperplane);
    if(_mass != 0.0)spinorOut.template iterateOverBulk_half<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    else spinorOut.template iterateOverBulk_half<BLOCKSIZE>(getFunctor(_tmpSpin));

    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM_single(SpinorRHS_t& spinorOut, SpinorRHS_t& spinorIn, bool update)
{
    _tmpSpin.template iterateOverBulk_single<BLOCKSIZE>(getFunctor(spinorIn));

    _tmpSpin.updateAll(COMM_BOTH | Hyperplane);
    if(_mass != 0.0)spinorOut.template iterateOverBulk_single<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    else spinorOut.template iterateOverBulk_single<BLOCKSIZE>(getFunctor(_tmpSpin));

    if(update)
        spinorOut.updateAll();
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<Layout LatLayout>
HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::getFunctor(Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
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
        SpinorLHS_t& lhs, SpinorRHS_t& rhs, bool update){
    lhs.template iterateOverBulk(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, SpinorRHS_t& spinorIn, bool update){

    Dslash(_tmpSpin, spinorIn, true);
    // This is not wrong! One Dslash is hiden in the line below in the getFunctor() call!
    // Optimization: We might get a speed up if we put this in a custom operator
    spinorOut.template iterateOverBulk(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<Layout LatLayout>
stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::getFunctor(Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge);
}



//! explicit template instantiations

#define DSLASH_INIT(floatT, LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class HisqDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>;
INIT_PLHHSN(DSLASH_INIT)

#define DSLASH_INIT_HALF(floatT, LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>;
INIT_PLHHSN_HALF(DSLASH_INIT_HALF)
