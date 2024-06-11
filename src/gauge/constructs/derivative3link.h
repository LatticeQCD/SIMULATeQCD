/*
 * derivative3link.h
 *
 * D. Bollweg
 *
 */

#pragma once

#include "../../base/indexer/bulkIndexer.h"
#include "../../base/math/su3Accessor.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
__host__ __device__ SU3<floatT> linkDerivative3(
    SU3Accessor<floatT,compIn> gAcc, 
    SU3Accessor<floatT,compForce> fAcc, 
    gSite site, 
    int mu, 
    int nu) 
{
    typedef GIndexer<All,HaloDepth> GInd;
    SU3<floatT> temp;
    gSite origin = site;
    gSite up = GInd::site_up(site,nu);
    gSite right = GInd::site_up(site,mu);
    gSite down = GInd::site_dn(site,nu);
    gSite rightDn = GInd::site_dn(right,nu);

    temp=fAcc.getLinkDagger(GInd::getSiteMu(right,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up,mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin,nu));

    temp+=gAcc.getLink(GInd::getSiteMu(right,nu))
            *fAcc.getLink(GInd::getSiteMu(up,mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin,nu));

    temp+=gAcc.getLink(GInd::getSiteMu(right,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up,mu))
            *fAcc.getLink(GInd::getSiteMu(origin,nu));

    temp+=fAcc.getLink(GInd::getSiteMu(rightDn,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down,mu))
            *gAcc.getLink(GInd::getSiteMu(down,nu));

    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn,nu))
            *fAcc.getLink(GInd::getSiteMu(down,mu))
            *gAcc.getLink(GInd::getSiteMu(down,nu));

    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down,mu))
            *fAcc.getLinkDagger(GInd::getSiteMu(down,nu));

    return temp;
};

