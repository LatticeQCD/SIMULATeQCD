/*
 * linkLepageConstructs.h
 *
 * J. Goswami
 *
 */

#pragma once
#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"


template<class floatT,size_t HaloDepth,CompressionType comp>
__device__ GSU3<floatT> inline linkLpUp(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    gSite origin = site;
    gSite up = GInd::site_up(site, nu);
    gSite twoUp = GInd::site_up(up, nu);
    gSite right = GInd::site_up(site, mu);
    gSite rightUp = GInd::site_up(right, nu);

    GSU3<floatT> temp;

    /*
     *   nu
     *   ^
     *   |
     *   |
     *    --> mu
     *   ^---->
     *   |    |
     *   ^    v
     *   |    |
     *   *
     *
     */

    temp =  gAcc.getLink(GInd::getSiteMu(origin, nu))
            * gAcc.getLink(GInd::getSiteMu(up, nu))
            * gAcc.getLink(GInd::getSiteMu(twoUp, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(rightUp, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(right, nu));

    return temp;
}


template<class floatT,size_t HaloDepth,CompressionType comp>
__device__ GSU3<floatT> inline linkLpDn(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;
    gSite dn = GInd::site_dn(site, nu);
    gSite twoDn = GInd::site_dn(dn, nu);
    gSite right = GInd::site_up(site, mu);
    gSite rightDn = GInd::site_dn(right, nu);
    gSite right2Dn = GInd::site_dn(rightDn, nu);

    GSU3<floatT> temp;

    /*
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *     *   ^
     *     |   |
     *     v   ^
     *     |   |
     *     v-- >
     *
     *
     */
    temp = gAcc.getLinkDagger(GInd::getSiteMu(dn, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
            * gAcc.getLink(GInd::getSiteMu(twoDn, mu))
            * gAcc.getLink(GInd::getSiteMu(right2Dn, nu))
            * gAcc.getLink(GInd::getSiteMu(rightDn, nu)) ;

    return temp;
}
