/*
 * naikDerivativeConstructs.h
 *
 * J. Goswami
 *
 */

#pragma once
#include "../../base/indexer/bulkIndexer.h"
#include "../../base/math/su3Accessor.h"

template<class floatT,size_t HaloDepth>
__host__ __device__ SU3<floatT> inline naikLinkDerivative(SU3Accessor<floatT> gAcc, SU3Accessor<floatT> fAcc, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    SU3<floatT> temp;

    gSite origin = site;
    gSite up_mu  = GInd::site_up(origin, mu);
    gSite up_2mu = GInd::site_up(up_mu, mu);
    gSite dn_mu  = GInd::site_dn(origin, mu);
    gSite dn_2mu = GInd::site_dn(dn_mu, mu);

    temp  = gAcc.getLink(GInd::getSiteMu(up_mu , mu)) * gAcc.getLink(GInd::getSiteMu(up_2mu, mu)) * fAcc.getLink(GInd::getSiteMu(origin, mu));
    temp += gAcc.getLink(GInd::getSiteMu(up_mu , mu)) * fAcc.getLink(GInd::getSiteMu(dn_mu , mu)) * gAcc.getLink(GInd::getSiteMu(dn_mu , mu));
    temp += fAcc.getLink(GInd::getSiteMu(dn_2mu, mu)) * gAcc.getLink(GInd::getSiteMu(dn_2mu, mu)) * gAcc.getLink(GInd::getSiteMu(dn_mu , mu));

    return temp;
}
