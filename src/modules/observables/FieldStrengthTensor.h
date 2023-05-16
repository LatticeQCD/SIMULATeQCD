//
//created by Marcel Rodekamp 30.05.2018
//


#pragma once
#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../../gauge/constructs/PlaqConstructs.h"


/*Qmunu(gaugeAccessor<floatT>, GSU3<floatT> &,  gSite,  int ,  int )
	Determine:
	Q_(mu,nu)(x) = U_(mu,nu)(x) + U_(-mu,nu)(x) + U_(-mu,-nu)(x) + U_(mu,nu)(x)
	where U_(mu,nu)(x) denotes the plaquette at site x.
*/
template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct plaqClover {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;

    plaqClover(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__  inline GSU3<floatT> operator()(gSite site, int mu, int nu) {

        return Plaq_P<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_Q<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_R<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_S<floatT, HaloDepth>(acc, site, mu, nu);
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct rectClover {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    typedef GIndexer<All, HaloDepth> GInd;

    rectClover(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        gSite origin = site;
        gSite up = GInd::site_up(site, nu);
        gSite twoUp = GInd::site_up(up, nu);
        gSite dn = GInd::site_dn(site, nu);
        gSite twoDn = GInd::site_dn(dn, nu);

        gSite left = GInd::site_dn(site, mu);
        gSite twoLeft = GInd::site_dn(left, mu);
        gSite twoLeftUp = GInd::site_up(twoLeft, nu);
        gSite twoLeftDn = GInd::site_dn(twoLeft, nu);
        gSite leftUp = GInd::site_up(left, nu);
        gSite leftDn = GInd::site_dn(left, nu);
        gSite left2Up = GInd::site_up(leftUp, nu);
        gSite left2Dn = GInd::site_dn(leftDn, nu);

        gSite right = GInd::site_up(site, mu);
        gSite twoRight = GInd::site_up(right, mu);
        gSite twoRightDn = GInd::site_dn(twoRight, nu);
        gSite rightUp = GInd::site_up(right, nu);
        gSite rightDn = GInd::site_dn(right, nu);
        gSite right2Dn = GInd::site_dn(rightDn, nu);


        GSU3<floatT> temp;

        // top right
        temp = acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, mu))
               * acc.getLink(GInd::getSiteMu(twoRight, nu))
               * acc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));

        temp += acc.getLink(GInd::getSiteMu(origin, mu))
                * acc.getLink(GInd::getSiteMu(right, nu))
                * acc.getLink(GInd::getSiteMu(rightUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, nu));

        //top left
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, nu))
                * acc.getLink(GInd::getSiteMu(twoLeft, mu))
                * acc.getLink(GInd::getSiteMu(left, mu));

        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLink(GInd::getSiteMu(up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left, nu))
                * acc.getLink(GInd::getSiteMu(left, mu));

        // bottom left
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftDn, nu))
                * acc.getLink(GInd::getSiteMu(twoLeftDn, mu))
                * acc.getLink(GInd::getSiteMu(leftDn, mu))
                * acc.getLink(GInd::getSiteMu(dn, nu));


        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(leftDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Dn, nu))
                * acc.getLink(GInd::getSiteMu(left2Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(dn, nu));

        // bottom right

        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLink(GInd::getSiteMu(dn, mu))
                * acc.getLink(GInd::getSiteMu(rightDn, mu))
                * acc.getLink(GInd::getSiteMu(twoRightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(right, mu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));


        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(twoDn, mu))
                * acc.getLink(GInd::getSiteMu(right2Dn, nu))
                * acc.getLink(GInd::getSiteMu(rightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));

        return temp;
    }
};

/*FmunuKernel(gaugeAccessor<floatT> , GSU3<floatT> & ,  gSite ,  int ,  int )
	computes the tracless clover given by
		a^2 * F_{mu,nu} = -i * 1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
	with
		Q_{mu,nu} = U_(mu,nu)(x) + U_(nu,-mu)(x) + U_(-mu,-nu)(x) + U_(-nu,mu)(x)
	where U denotes the Link variables.

*/
template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct FieldStrengthTensor {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    plaqClover<floatT,HaloDepth,onDevice,comp> plClov;
    typedef GIndexer<All, HaloDepth> GInd;

    FieldStrengthTensor(gaugeAccessor<floatT,comp> acc) : acc(acc),
    plClov(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        //define a unitary matrix for the addition in the end
        GSU3<floatT> unityGSU3 = gsu3_one<floatT>();

        //define a temporary GSU3 for the Fmunu computation
        GSU3<floatT> Fmunu;

        //define a temporary GSU3 for the Qmunu computations
        GSU3<floatT> Qmunu;

        Qmunu = plClov(site, mu, nu);

        // compute F_{mu,nu} = -i*1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
        Fmunu = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu)); // this is faster...

        // return tracless F_{mu,nu}
        return Fmunu - 1. / 3. * tr_c(Fmunu) * unityGSU3;

    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct FieldStrengthTensor_imp {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> acc;
    plaqClover<floatT,HaloDepth,onDevice,comp> plClov;
    rectClover<floatT,HaloDepth,onDevice,comp> rcClov;
    typedef GIndexer<All, HaloDepth> GInd;

    FieldStrengthTensor_imp(gaugeAccessor<floatT,comp> acc) : acc(acc),
    plClov(acc), rcClov(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        //define a unitary matrix for the addition in the end
        GSU3<floatT> unityGSU3 = gsu3_one<floatT>();

        //define a temporary GSU3 for the Fmunu computation
        GSU3<floatT> Fmunu;
        GSU3<floatT> Fmunu_plaq;
        GSU3<floatT> Fmunu_rect;

        //define a temporary GSU3 for the Qmunu computations
        GSU3<floatT> Qmunu;

        Qmunu = plClov(site, mu, nu);
        // compute F_{mu,nu} = -i*1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
        Fmunu_plaq = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu)); // this is faster...

        Qmunu = rcClov(site, mu, nu);
        Fmunu_rect = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 16) * (Qmunu - dagger(Qmunu));


        Fmunu = floatT(5./3.) * Fmunu_plaq - floatT(1./3.) * Fmunu_rect;

        // return tracless F_{mu,nu}
        return Fmunu - floatT(1./3.) * tr_c(Fmunu) * unityGSU3;

    }
};

