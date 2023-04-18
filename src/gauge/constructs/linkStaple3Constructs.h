/*
 * linkStaple3Constructs.h
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


template<class floatT,size_t HaloDepth, CompressionType comp>
__device__ GSU3<floatT> inline linkStaple3Up(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;
    gSite up = GInd::site_up(site, nu);
    gSite right = GInd::site_up(site, mu);

   /*
    *    nu
    *    ^
    *    |
    *    |
    *     --> mu
    *
    *    ^ --->
    *    |    |
    *    *    v
    *
    */

    temp = gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
    return temp;
}


template<class floatT,size_t HaloDepth,CompressionType comp>
__device__ GSU3<floatT> inline linkStaple3Dn(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;
    gSite dn = GInd::site_dn(site, nu);
    gSite right = GInd::site_up(site, mu);
    gSite rightDn = GInd::site_dn(right, nu);

    /*
     *
     *
     *    *    ^
     *    |    |
     *    v---->
     *
    */

    temp = gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), mu, nu);
    return temp;
}
