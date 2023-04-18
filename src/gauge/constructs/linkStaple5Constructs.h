/*
 * linkStaple5Constructs.h
 *
 * J. Goswami
 *
 */

#ifndef LINKSTAPLE5CONSTRUCTS_H
#define LINKSTAPLE5CONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"
#include "linkStaple3Constructs.h"

template<class floatT,size_t HaloDepth,CompressionType comp>
__device__ GSU3<floatT> inline linkStaple5Up(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu, int rho, int gamma) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> staple3;
    GSU3<floatT> temp = gsu3_zero<floatT>() ;

    gSite origin = site;
    gSite site_up_rho = GInd::site_up(site,rho);
    gSite site_dn_rho = GInd::site_dn(site,rho);
    gSite site_up_gamma = GInd::site_up(site,gamma);
    gSite site_dn_gamma = GInd::site_dn(site,gamma);
    gSite site_up_mu = GInd::site_up(site, mu);
    gSite site_dn_rho_up_mu= GInd::site_up(site_dn_rho,mu);
    gSite site_dn_gamma_up_mu= GInd::site_up(site_dn_gamma,mu);

    /*
     *
     *
     *
     *    nu ^   ^ rho
     *       |  /
     *       | /
     *       |/
     *       -------> mu
     *
     *
     *    sum of staple3
     *      ^+++>
     *     /   /
     *    /   /
     *   *   v
     *
     *
     *        *   ^
     *       /   /
     *      /   /
     *     v+++>
     *    sum of staple3
     *
     *
     */

    staple3=linkStaple3Up<floatT,HaloDepth,comp>(gAcc,site_up_rho,mu,gamma)
            + linkStaple3Dn<floatT,HaloDepth,comp>(gAcc,site_up_rho,mu,gamma);

    temp = gAcc.getLink(GInd::getSiteMu(origin, rho)) * staple3
                  * gAcc.getLinkDagger(GInd::getSiteMu(site_up_mu, rho));


    staple3=linkStaple3Up<floatT,HaloDepth,comp>(gAcc,site_dn_rho,mu,gamma)
            + linkStaple3Dn<floatT,HaloDepth,comp>(gAcc,site_dn_rho,mu,gamma);

    temp += gAcc.getLinkDagger(GInd::getSiteMu(site_dn_rho, rho)) * staple3
           * gAcc.getLink(GInd::getSiteMu(site_dn_rho_up_mu, rho));


    /*
     *    nu ^   ^ gamma
     *       |  /
     *       | /
     *       |/
     *       -------> mu
     *
     *
     *    sum of staple3
     *      ^+++>
     *     /   /
     *    /   /
     *   *   v
     *
     *
     *        *   ^
     *       /   /
     *      /   /
     *     v+++>
     *    sum of staple3
     *
     */

    staple3=linkStaple3Up<floatT,HaloDepth,comp>(gAcc,site_up_gamma,mu,rho)
            + linkStaple3Dn<floatT,HaloDepth,comp>(gAcc,site_up_gamma,mu,rho);

    temp += gAcc.getLink(GInd::getSiteMu(origin, gamma)) * staple3
           * gAcc.getLinkDagger(GInd::getSiteMu(site_up_mu, gamma));

    staple3=linkStaple3Up<floatT,HaloDepth,comp>(gAcc,site_dn_gamma,mu,rho)
            + linkStaple3Dn<floatT,HaloDepth,comp>(gAcc,site_dn_gamma,mu,rho);

    temp += gAcc.getLinkDagger(GInd::getSiteMu(site_dn_gamma, gamma))*staple3
                   * gAcc.getLink(GInd::getSiteMu(site_dn_gamma_up_mu, gamma));

    return temp;
}
#endif //5LINKSTAPLECONSTRUCTS_H
