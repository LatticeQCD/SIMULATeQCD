/* 
 * linkStaple7Constructs.h                                                               
 * 
 * J. Goswami 
 * 
 */

#ifndef LINKSTAPLE7CONSTRUCTS_H
#define LINKSTAPLE7CONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"
#include "linkStaple3Constructs.h"
#include "linkStaple5Constructs.h"

template<class floatT,size_t HaloDepth, CompressionType comp>
__device__ GSU3<floatT> inline linkStaple7Up(gaugeAccessor<floatT,comp> gAcc,gSite site, int mu, int nu, int rho, int gamma){
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> staple5=gsu3_zero<floatT>();
    GSU3<floatT> temp = gsu3_zero<floatT>();

    gSite origin = site;
    gSite up_nu = GInd::site_up(origin, nu);
    gSite up_nu_up_nu = GInd::site_up(up_nu, nu);
    gSite dn_nu = GInd::site_dn(origin, nu);
    gSite up_mu = GInd::site_up(origin, mu);
    gSite dn_nu_up_mu=GInd::site_up(dn_nu,mu);

    /*
     *    nu ^
     *       |
     *       |_____ > mu
     *
     *
     *    sum of staple5
     *     ^+++>
     *     |   |
     *     |   |
     *    *    v
     *
     *    nu ^
     *       |
     *       |_____ > mu
     *
     *
     *    *    ^
     *     |   |
     *     |   |
     *     v+++>
     *   sum of staple5
     *
     */

    staple5 = linkStaple5Up<floatT,HaloDepth,comp>(gAcc,up_nu,mu,nu,rho,gamma);

    temp =  gAcc.getLink(GInd::getSiteMu(origin, nu)) * staple5
            * gAcc.getLinkDagger(GInd::getSiteMu(up_mu, nu)) ;

    staple5 = linkStaple5Up<floatT,HaloDepth,comp>(gAcc,dn_nu,mu,nu,rho,gamma);

    temp += gAcc.getLinkDagger(GInd::getSiteMu(dn_nu, nu)) *staple5
             * gAcc.getLink(GInd::getSiteMu(dn_nu_up_mu, nu));

    return temp;
}
#endif //LINKSTAPLE7CONSTRUCTS_H
