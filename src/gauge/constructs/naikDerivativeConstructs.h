/*
 * naikDerivativeConstructs.h
 *
 * J. Goswami
 *
 */

#pragma once
#include "../../base/indexer/BulkIndexer.h"
#include "../../base/math/gaugeAccessor.h"

template<class floatT,size_t HaloDepth>
__host__ __device__ GSU3<floatT> inline naikLinkDerivative(gaugeAccessor<floatT> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    GSU3<floatT> temp;

    gSite origin = site;
    gSite up_mu  = GInd::site_up(origin, mu);
    gSite up_2mu = GInd::site_up(up_mu, mu);
    gSite dn_mu  = GInd::site_dn(origin, mu);
    gSite dn_2mu = GInd::site_dn(dn_mu, mu);

    temp = gAcc.getLink(GInd::getSiteMu(up_mu, mu)) * gAcc.getLink(GInd::getSiteMu(up_2mu, mu)) * finAccessor.getLink(GInd::getSiteMu(origin, mu));

    temp += gAcc.getLink(GInd::getSiteMu(up_mu, mu)) * finAccessor.getLink(GInd::getSiteMu(dn_mu, mu)) * gAcc.getLink(GInd::getSiteMu(dn_mu, mu));

    temp += finAccessor.getLink(GInd::getSiteMu(dn_2mu, mu)) * gAcc.getLink(GInd::getSiteMu(dn_2mu, mu)) * gAcc.getLink(GInd::getSiteMu(dn_mu, mu));

    return temp;
}
