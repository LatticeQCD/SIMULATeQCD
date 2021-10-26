//created by Dennis on 01.11.18

#ifndef DERIVATIVE_3LINK_H
#define DERIVATIVE_3LINK_H

#include "../../base/indexer/BulkIndexer.h"
#include "../../base/math/gaugeAccessor.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
__device__ GSU3<floatT> linkDerivative3(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp;
    gSite origin = site;
    gSite up = GInd::site_up(site,nu);
    gSite right = GInd::site_up(site,mu);
    gSite down = GInd::site_dn(site,nu);
    gSite rightDn = GInd::site_dn(right,nu);

    temp=finAccessor.getLinkDagger(GInd::getSiteMu(right,nu)) //fail ... now pass
            *gAcc.getLinkDagger(GInd::getSiteMu(up,mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin,nu));

    temp+=gAcc.getLink(GInd::getSiteMu(right,nu))
            *finAccessor.getLink(GInd::getSiteMu(up,mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin,nu)); //pass

    temp+=gAcc.getLink(GInd::getSiteMu(right,nu)) //pass
            *gAcc.getLinkDagger(GInd::getSiteMu(up,mu))
            *finAccessor.getLink(GInd::getSiteMu(origin,nu));

    temp+=finAccessor.getLink(GInd::getSiteMu(rightDn,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down,mu))
            *gAcc.getLink(GInd::getSiteMu(down,nu)); //pass

    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn,nu))
            *finAccessor.getLink(GInd::getSiteMu(down,mu))
            *gAcc.getLink(GInd::getSiteMu(down,nu)); //pass

    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn,nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down,mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(down,nu)); //fail ... now pass

    return temp;
};

#endif // DERIVATIVE_3LINK_H