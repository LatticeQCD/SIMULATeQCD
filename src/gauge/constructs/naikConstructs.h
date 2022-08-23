//
// Created by Jishnu on 21/08/18.
//

#ifndef NAIKCONSTRUCTS_H
#define NAIKCONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"

template<class floatT,size_t HaloDepth>
DEVICE GSU3<floatT> inline naik3LinkUp(gaugeAccessor<floatT> gAcc, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    GSU3<floatT> temp;

    gSite origin = site;
    gSite down_mu = GInd::site_dn(origin,mu);
    gSite up_mu = GInd::site_up(origin, mu);
    //gSite up_2mu = GInd::site_up(up_mu, mu);
    

    temp = gAcc.getLink(GInd::getSiteMu(down_mu, mu))
           * gAcc.getLink(GInd::getSiteMu(origin, mu))
           * gAcc.getLink(GInd::getSiteMu(up_mu, mu));


    return temp;
}
/*
template<class floatT,size_t HaloDepth>
DEVICE GSU3<floatT> inline naik3LinkDn(gaugeAccessor<floatT> gAcc, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    GSU3<floatT> temp;

    gSite origin = site;
    gSite dn_mu = GInd::site_dn(origin, mu);
    gSite dn_2mu = GInd::site_dn(dn_mu, mu);
    gSite dn_3mu = GInd::site_dn(dn_2mu, mu);



    temp = gAcc.getLinkDagger(GInd::getSiteMu(dn_mu, mu))
           * gAcc.getLinkDagger(GInd::getSiteMu(dn_2mu, mu))
           * gAcc.getLinkDagger(GInd::getSiteMu(dn_3mu, mu));


    return temp;


    }*/


#endif //NAIKCONSTRUCTS_H
