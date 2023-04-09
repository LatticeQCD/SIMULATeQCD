/*
 * PlaqConstructs.h
 *
 * L. Mazur
 *
 */

#ifndef LINKCONSTRUCTS_H
#define LINKCONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"


template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline Plaq_P(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;

    /*
     *   compute  U_(mu,nu)(x) = U_(mu)(x)*U_(nu)(x+mu)*U+_(mu)(x+nu)U+_(nu)(x)
     *   P_{\mu\nu}(x)=U_{\mu}(x)U_{\nu}(x+\mu)U_{\mu}(x+\nu)^{\dagger}U_{\nu}(x)^{\dagger}
     *
     *   mu --->
     *   nu
     *   |    *^--->
     *   v     |   |
     *         <---v
     *
     */
    temp = gAcc.getLink(GInd::getSiteMu(site, mu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_up(site, mu), nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site, nu), mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(site, nu));
    return temp;
}


template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline Plaq_Q(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;
    /*
     *   compute U_(mu,nu)(x) = U+_(nu)(x-nu)*U_(mu)(x-nu)*U_(nu)(x-nu+mu)*U+_(mu)(x)
     *   Q_{\mu\nu}(x)=U_{\nu}(x-\nu)^{\dagger}U_{\mu}(x-\nu)U_{\nu}(x+\mu-\nu)U_{\mu}(x)^{\dagger}
     *
     *   mu --->
     *   nu
     *   |     ^--->
     *   v     |   |
     *        *<---v
     *
     */

    temp = gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, nu), nu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site, nu), mu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site, mu, nu), nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(site, mu));
    return temp;
}


template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline Plaq_R(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;

    /*
     *  compute U_(-mu,-nu)(x) = U+_(mu)(x-mu)*U+_(nu)(x-nu-mu)*U_(mu)(x-nu-mu)*U_(nu)(x-nu)
     *  R_{\mu\nu}(x)=U_{\mu}(x-\mu)^{\dagger}U_{\nu}(x-\mu-\nu)^{\dagger}U_{\mu}(x-\mu-\nu)U_{\nu}(x-\nu)
     *
     *   mu --->
     *   nu
     *   |     ^--->
     *   v     |   |
     *         <---v*
     *
     */
    temp = gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, mu), mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), nu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), mu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site, nu), nu));
    return temp;
}


template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline Plaq_S(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;

    /*
     *  compute U_(-mu,nu)(x) = U_(nu)(x)*U+_(mu)(x+nu-mu)*U+_(nu)(x-mu)*U_(mu)(x-mu)
     *  S_{\mu\nu}(x)=U_{\nu}(x)U_{\mu}(x-\mu+\nu)^{\dagger}U_{\nu}(x-\mu)^{\dagger}U_{\mu}(x-\mu)
     *
     *   mu --->
     *   nu
     *   |     ^--->*
     *   v     |   |
     *         <---v
     *
     */

    temp = gAcc.getLink(GInd::getSiteMu(site, nu))
            * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site, nu, mu), mu))
            * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, mu), nu))
            * gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu));
    return temp;
}


#endif //LINKCONSTRUCTS_H
