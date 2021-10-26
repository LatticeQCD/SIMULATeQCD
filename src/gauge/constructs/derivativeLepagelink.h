//created by Dennis on 02.11.18

#ifndef DERIVATIVE_LEPAGE_H
#define DERIVATIVE_LEPAGE_H

#include "../../base/indexer/BulkIndexer.h"
#include "../../base/math/gaugeAccessor.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
__device__ GSU3<floatT> linkDerivativeLepage(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp;
    gSite origin = site;
    gSite up = GInd::site_up(site,nu);
    gSite down = GInd::site_dn(site,nu);
    gSite right = GInd::site_up(site,mu);
    gSite rightUp = GInd::site_up(right,nu);
    gSite rightDn = GInd::site_dn(right,nu);
    gSite rightDnDn = GInd::site_dn(rightDn,nu);
    gSite rightright = GInd::site_up(right,mu);
    gSite rightrightDn = GInd::site_dn(rightright,nu);
    gSite left = GInd::site_dn(site,mu);
    gSite leftUp = GInd::site_up(left,nu);
    gSite leftDn = GInd::site_dn(left,nu);
    gSite UpUp = GInd::site_up(up,nu);
    gSite DnDn = GInd::site_dn(down,nu);

    //term 6
    temp=gAcc.getLink(GInd::getSiteMu(right, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
            *finAccessor.getLink(GInd::getSiteMu(left, nu))
            *gAcc.getLink(GInd::getSiteMu(left, mu)); //matches mw2b

    //term 5
    temp+=gAcc.getLink(GInd::getSiteMu(right, mu))
            *gAcc.getLink(GInd::getSiteMu(rightright, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up, mu))
            *finAccessor.getLink(GInd::getSiteMu(origin, nu)); //matches mw1b

    //term 9
    temp+=finAccessor.getLinkDagger(GInd::getSiteMu(right, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(left, nu))
            *gAcc.getLink(GInd::getSiteMu(left, mu)); //matches mw5b

    //term 10
    temp+=gAcc.getLink(GInd::getSiteMu(right, mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(rightright, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin, nu)); //matches mw4a


    //term 4
    temp+=gAcc.getLink(GInd::getSiteMu(right,nu))
            *gAcc.getLink(GInd::getSiteMu(rightUp,nu))
            *finAccessor.getLink(GInd::getSiteMu(UpUp,mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(up, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(origin,nu)); //matches mw3b //dagger was missing

    //term 2
    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(leftDn, mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(leftDn, nu))
            *gAcc.getLink(GInd::getSiteMu(left, mu)); //matches mw2a

    //term 1
    temp+=gAcc.getLink(GInd::getSiteMu(right, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightrightDn, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightDn, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down, mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(down, nu)); //matches mw1a

    //term 7
    temp+=finAccessor.getLink(GInd::getSiteMu(rightDn, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(leftDn, mu))
            *gAcc.getLink(GInd::getSiteMu(leftDn, nu))
            *gAcc.getLink(GInd::getSiteMu(left, mu)); //matches mw5a

    //term 8
    temp+=gAcc.getLink(GInd::getSiteMu(right, mu))
            *finAccessor.getLink(GInd::getSiteMu(rightrightDn, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightDn, mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(down, mu))
            *gAcc.getLink(GInd::getSiteMu(down, nu)); //matches mw4b

    //term 3
    temp+=gAcc.getLinkDagger(GInd::getSiteMu(rightDn, nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(rightDnDn, nu))
            *finAccessor.getLink(GInd::getSiteMu(DnDn, mu))
            *gAcc.getLink(GInd::getSiteMu(DnDn, nu))
            *gAcc.getLink(GInd::getSiteMu(down, nu)); //matches m3a

    return temp;
}

#endif // DERIVATIVE_LEPAGE_H