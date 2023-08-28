#include "dslashDerivative.h"

template<class floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
__device__ __host__ Vect3<floatT> dDdmuFunctor<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::operator()(gSiteStack site) const {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);

    const int mu = 3;
    Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
        _spinorIn.getElement(GInd::site_up(site, mu));
    Stmp -= (static_cast<floatT>(C_1000) * _sign) * _gAcc_smeared.getLinkDagger(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
        _spinorIn.getElement(GInd::site_dn(site, mu));

    Stmp += _pow_3 * _gAcc_Naik.getLink(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu))) *
        _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));
    Stmp -= (_pow_3 * _sign) * _gAcc_Naik.getLinkDagger(
            GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu))) *
        _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));

    return Stmp;
}

#define SPINOR_INIT_PLHHSN(floatT,LO,HALO,HALOSPIN,STACKS)\
template class dDdmuFunctor<floatT,false,LO,HALO,HALOSPIN,STACKS>;\
template class dDdmuFunctor<floatT,true,LO,HALO,HALOSPIN,STACKS>;\

INIT_PLHHSN(SPINOR_INIT_PLHHSN)
