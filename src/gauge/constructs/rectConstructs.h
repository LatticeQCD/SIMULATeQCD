/*
 * rectConstructs.h
 *
 * L. Mazur
 *
 */

#pragma once

#include "../../define.h"
#include "../../base/math/complex.h"
#include "../../base/gutils.h"
#include "../../base/math/su3array.h"
#include "../../base/math/su3.h"
#include "../gaugefield.h"


template<class floatT,size_t HaloDepth>
__device__ SU3<floatT> inline Rect_P(SU3Accessor<floatT> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    gSite origin = site;
    gSite up = GInd::site_up(site, nu);
    gSite twoUp = GInd::site_up(up, nu);

    gSite right = GInd::site_up(site, mu);
    gSite twoRight = GInd::site_up(right, mu);
    gSite rightUp = GInd::site_up(right, nu);

    SU3<floatT> temp;

    /*
     *  Rect_(mu,nu)(x) = U_(mu)(x)*U_(mu)(x+mu)*U_(nu)(x+2mu)*U+_(mu)(x+nu+mu)*U+_(mu)(x+nu)*U+_(nu)(x)
     *  			+ U_(mu)(x)*U_(nu)(x+mu)*U_(nu)(x+mu+nu)*U+_(mu)(x+2nu)*U+_(nu)(x+nu)*U+_(nu)(x)
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *
     *
     *     <--<--^
     *     |  	 |
     *    *v-->-->
     *
     *
     */
    temp = gAcc.getLink(GInd::getSiteMu(origin, mu))
           * gAcc.getLink(GInd::getSiteMu(right, mu))
           * gAcc.getLink(GInd::getSiteMu(twoRight, nu))
           * gAcc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
           * gAcc.getLinkDagger(GInd::getSiteMu(up, mu))
           * gAcc.getLinkDagger(GInd::getSiteMu(origin, nu));
    /* nu  ^
     *     |
     *     |
     *  mu  --->
     *
     *
     *     <--^
     *     |  |
     *     v  ^
     *     |  |
     *    *v-->
     *
     *
     */
    temp += gAcc.getLink(GInd::getSiteMu(origin, mu))
            * gAcc.getLink(GInd::getSiteMu(right, nu))
            * gAcc.getLink(GInd::getSiteMu(rightUp, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoUp, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(up, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(origin, nu));

    return temp;
}


template<class floatT,size_t HaloDepth>
__device__ SU3<floatT> inline Rect_Q(SU3Accessor<floatT> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    gSite origin = site;
    gSite dn = GInd::site_dn(site, nu);
    gSite twoDn = GInd::site_dn(dn, nu);

    gSite right = GInd::site_up(site, mu);
    gSite twoRight = GInd::site_up(right, mu);
    gSite twoRightDn = GInd::site_dn(twoRight, nu);
    gSite rightDn = GInd::site_dn(right, nu);
    gSite right2Dn = GInd::site_dn(rightDn, nu);

    SU3<floatT> temp;

    /*
     *  Rect_(mu,nu)(x) = U+_(nu)(x-nu)*U_(mu)(x-nu)*U_(mu)(x+mu-nu)*U_(nu)(x+2mu-nu)*U+_(mu)(x+mu)*U+_(mu)(x)
     *  			+ U+_(nu)(x-nu)*U+_(nu)(x-2nu)*U_(mu)(x-2nu)*U_(nu)(x+mu-2nu)*U_(nu)(x+mu-nu)*U+_(mu)(x)
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *
     *
     *     *--<--^
     *     |  	 |
     *     v-->-->
     *
     *
     */
    temp += gAcc.getLinkDagger(GInd::getSiteMu(dn, nu))
            * gAcc.getLink(GInd::getSiteMu(dn, mu))
            * gAcc.getLink(GInd::getSiteMu(rightDn, mu))
            * gAcc.getLink(GInd::getSiteMu(twoRightDn, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(right, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(origin, mu));
    /*
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *     *-- ^
     *     |   |
     *     v   ^
     *     |   |
     *     v-- >
     *
     *
     */
    temp += gAcc.getLinkDagger(GInd::getSiteMu(dn, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
            * gAcc.getLink(GInd::getSiteMu(twoDn, mu))
            * gAcc.getLink(GInd::getSiteMu(right2Dn, nu))
            * gAcc.getLink(GInd::getSiteMu(rightDn, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(origin, mu));

    return temp;
}

template<class floatT,size_t HaloDepth>
__device__ SU3<floatT> inline Rect_S(SU3Accessor<floatT> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    gSite origin = site;
    gSite up = GInd::site_up(site, nu);

    gSite left = GInd::site_dn(site, mu);
    gSite twoLeft = GInd::site_dn(left, mu);
    gSite twoLeftUp = GInd::site_up(twoLeft, nu);
    gSite leftUp = GInd::site_up(left, nu);
    gSite left2Up = GInd::site_up(leftUp, nu);

    SU3<floatT> temp;
    /*  Rect_(mu,nu)(x) = U_(nu)(x)*U+_(mu)(x-mu+nu)*U+_(mu)(x-2mu+nu)*U+_(nu)(x-2mu)*U_(mu)(x-2mu)*U_(nu)(x-mu)
     *      + U_(nu)(x)*U_(nu)(x+nu)*U+_(mu)(x-mu+2nu)*U+_(nu)(x+2nu)*U+_(nu)(x+nu)*U+_(nu)(x)
     *  nu ^
     *     |
     *     |
     *      ---> mu
     *
     *
     *
     *     <--<--^
     *     |  	 |
     *     v-->--v*
     *
     *
     */
    temp += gAcc.getLink(GInd::getSiteMu(origin, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoLeftUp, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoLeft, nu))
            * gAcc.getLink(GInd::getSiteMu(twoLeft, mu))
            * gAcc.getLink(GInd::getSiteMu(left, mu));

    /*
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *
     * <-- ^
     * |   |
     * v   ^
     * |   |
     * v-->*
     *
     */
    temp += gAcc.getLink(GInd::getSiteMu(origin, nu))
            * gAcc.getLink(GInd::getSiteMu(up, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(left2Up, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(leftUp, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(left, nu))
            * gAcc.getLink(GInd::getSiteMu(left, mu));

    return temp;
}

template<class floatT,size_t HaloDepth>
__device__ SU3<floatT> inline Rect_R(SU3Accessor<floatT> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    gSite dn = GInd::site_dn(site, nu);
    gSite twoDn = GInd::site_dn(dn, nu);

    gSite left = GInd::site_dn(site, mu);
    gSite twoLeft = GInd::site_dn(left, mu);
    gSite twoLeftDn = GInd::site_dn(twoLeft, nu);
    gSite leftDn = GInd::site_dn(left, nu);
    gSite left2Dn = GInd::site_dn(leftDn, nu);

    SU3<floatT> temp;

    /* Rect_(mu,nu)(x) = {U+_(mu)(x-mu)*U+_(mu)(x-2mu)*U+_(mu)(x-2mu+nu)*U_(mu)(x-2mu-nu)*U_(nu)(x-mu-nu)
     *      +U_(nu)(x-nu)}
     *     + {U+_(mu)(x-mu)*U+_(nu)(x-mu-nu)*U+_(nu)(x-mu-2nu)*U_(mu)(x-mu-2nu)*U_(nu)(x-2nu)*U_(nu)(x-nu)}
     *  nu ^
     *     |
     *     |
     *      ---> mu
     *
     *
     *     <--<--^*
     *     |  	 |
     *     v-->-->
     *
     */

    temp += gAcc.getLinkDagger(GInd::getSiteMu(left, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(twoLeftDn, nu))
            * gAcc.getLink(GInd::getSiteMu(twoLeftDn, mu))
            * gAcc.getLink(GInd::getSiteMu(leftDn, mu))
            * gAcc.getLink(GInd::getSiteMu(dn, nu));
    /*
     *  nu ^
     *     |
     *     |
     *  mu  --->
     *
     *     <--*
     *     |   |
     *     v   ^
     *     |   |
     *     v -->
     *
     */

    temp += gAcc.getLinkDagger(GInd::getSiteMu(left, mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(leftDn, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(left2Dn, nu))
            * gAcc.getLink(GInd::getSiteMu(left2Dn, mu))
            * gAcc.getLink(GInd::getSiteMu(twoDn, nu))
            * gAcc.getLink(GInd::getSiteMu(dn, nu));

    return temp;
}

