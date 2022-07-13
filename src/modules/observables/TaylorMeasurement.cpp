//#include "TaylorMeasurement.h"
/*
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__device__ __host__ gVect3<floatT> dDdmuFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    gVect3<floatT> Stmp(0.0);

    const int mu = 3;
    Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
        _spinorIn.getElement(GInd::site_up(site, mu));
    Stmp += (static_cast<floatT>(C_1000) * _sign) * _gAcc_smeared.getLinkDagger(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
        _spinorIn.getElement(GInd::site_dn(site, mu));

    Stmp += (_c_3000 * _pow_3) * _gAcc_Naik.getLink(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu))) *
        _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));
    Stmp += (_c_3000 * _pow_3 * _sign) * _gAcc_Naik.getLinkDagger(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu))) *
        _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));

    return Stmp;
}

#define SPINOR_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class dDdmuFunctor<floatT,false,LO,HALOSPIN,STACKS>;\
template class dDdmuFunctor<floatT,true,LO,HALOSPIN,STACKS>;\
//template void dDdmuFunctor<floatT,false,LO,HALOSPIN,STACKS>::axpyThis(const GCOMPLEX(floatT)&, const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\

INIT_PLHSN(SPINOR_INIT_PLHSN)*/
