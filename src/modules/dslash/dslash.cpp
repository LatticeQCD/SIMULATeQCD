#include "dslash.h"
#define BLOCKSIZE 64

//! HisqDslash

template<Layout LatLayoutRHS, size_t HaloDepthSpin>
__host__ __device__ size_t spinor_coord_to_ind(sitexyzt coord, int stack){

    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    return GInd::coordToIndex_Full_eo(coord) + stack * GInd::getLatData().sizehFull;

}

//! HisqDslash

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);

    for (int mu = 0; mu < 4; mu++) {


        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
                GIndexer<All, HaloDepthGauge>::coordMuToIndexMu_Full(site.coordFull.x, site.coordFull.y, site.coordFull.z, site.coordFull.t,  mu)) *
            _spinorIn.getElement(spinor_coord_to_ind<LatLayoutRHS, HaloDepthSpin>(GInd::template site_move<1>(site.coordFull, mu), site.stack));
        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(
                GIndexer<All, HaloDepthGauge>::coordMuToIndexMu_Full(GInd::template site_move<-1>(site.coordFull, mu).x, 
			GInd::template site_move<-1>(site.coordFull, mu).y, GInd::template site_move<-1>(site.coordFull, mu).z, 
			GInd::template site_move<-1>(site.coordFull, mu).t,  mu)) *
            _spinorIn.getElement(spinor_coord_to_ind<LatLayoutRHS, HaloDepthSpin>(GInd::template site_move<-1>(site.coordFull, mu), site.stack));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(
                GIndexer<All, HaloDepthGauge>::coordMuToIndexMu_Full(GInd::template site_move<1>(site.coordFull, mu).x, 
			GInd::template site_move<1>(site.coordFull, mu).y, GInd::template site_move<1>(site.coordFull, mu).z, 
			GInd::template site_move<1>(site.coordFull, mu).t,  mu)) *
            _spinorIn.getElement(spinor_coord_to_ind<LatLayoutRHS, HaloDepthSpin>(GInd::template site_move<3>(site.coordFull, mu), site.stack));


        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(
                GIndexer<All, HaloDepthGauge>::coordMuToIndexMu_Full(GInd::template site_move<-2>(site.coordFull, mu).x, 
			GInd::template site_move<-2>(site.coordFull, mu).y, GInd::template site_move<-2>(site.coordFull, mu).z, 
			GInd::template site_move<-2>(site.coordFull, mu).t,  mu)) *
            _spinorIn.getElement(spinor_coord_to_ind<LatLayoutRHS, HaloDepthSpin>(GInd::template site_move<-3>(site.coordFull, mu), site.stack));
    }
    return Stmp;
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
