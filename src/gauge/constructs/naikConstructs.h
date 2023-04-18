/*
 * naikConstructs.h
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

template<class floatT,size_t HaloDepth>
__device__ GSU3<floatT> inline naik3LinkUp(gaugeAccessor<floatT> gAcc, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    GSU3<floatT> temp;

    gSite origin = site;
    gSite down_mu = GInd::site_dn(origin,mu);
    gSite up_mu = GInd::site_up(origin, mu);

    temp = gAcc.getLink(GInd::getSiteMu(down_mu, mu))
           * gAcc.getLink(GInd::getSiteMu(origin, mu))
           * gAcc.getLink(GInd::getSiteMu(up_mu, mu));

    return temp;
}

