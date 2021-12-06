/* 
 * gfix.cpp
 *
 * v2.4: D. Clarke, 11 Feb 2019
 *
 * Gpu file with kernel definitions for main_gfix, as well as the functions calling these kernels.
 *
 */

#include "gfix.h"

/// Kernel to compute local contribution to GF functional.
template<class floatT,size_t HaloDepth>
struct GFActionKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    GFActionKernel(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){
    }
    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        floatT gfa=0.0;
        /// For Coulomb and Landau gauge fixing, the functional to be maximized is ~sum_{x,mu} Re tr U.
        for ( int mu = 0; mu < I_FIX; mu++) {
            gfa+=tr_d(gaugeAccessor.getLink(GInd::getSiteMu(site, mu)));
        }
        return gfa;
    }
};

/// Kernel to compute local contribution to GF theta.
template<class floatT,size_t HaloDepth>
struct GFThetaKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    GFThetaKernel(Gaugefield<floatT,true,HaloDepth>&gauge):gaugeAccessor(gauge.getAccessor()){
    }
    __device__ __host__ floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;
        floatT theta=0.0;
        GSU3<floatT> delta,temp;
        delta=gsu3_zero<floatT>();
        for(int mu=0;mu<I_FIX;mu++){
            temp=        gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu))
                 -gaugeAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, mu), mu))
                 -       gaugeAccessor.getLink(GInd::getSiteMu(site, mu))
                 +gaugeAccessor.getLinkDagger(GInd::getSiteMu(site, mu));
            temp=temp-1./3.*tr_c(temp)*gsu3_one<floatT>();
            delta+=temp;
        }
        theta=tr_d(delta,dagger(delta));
        return theta;
    }
};

/// Kernel to gauge fix via over-relaxation.
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct GFORKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    GFORKernel(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        GSU3<floatT> v,g;
        GSU2<floatT> z1,z2,z3;
        floatT a0,a1,a2,a3,asq,a0sq,x,r,xdr;
        const floatT relax=1.3;
        GCOMPLEX(floatT) x00,x01;

        v=gsu3_one<floatT>();
        for( int mu = 0; mu < I_FIX; mu++){
            v+=gaugeAccessor.getLinkDagger(GInd::getSiteMu(site, mu));              /// w += U_{mu}(site)^{dagger}
            v+=gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu)); /// w += U_{mu}(site-hat{mu})
        }

        /// FIRST SU(2) SUBGROUP: COMPUTE LOCAL MAX
        a0 =  real(v.getLink00()) + real(v.getLink11());
        a1 =  imag(v.getLink10()) + imag(v.getLink01());
        a2 = -real(v.getLink10()) + real(v.getLink01());
        a3 =  imag(v.getLink00()) - imag(v.getLink11());

        asq  = a1*a1 + a2*a2 + a3*a3;
        a0sq = a0*a0;

        x = (relax*a0sq+asq)/(a0sq+asq);
        r = sqrt(a0sq+x*x*asq);
        xdr = x/r;

        a0/=r;
        a1*=xdr;
        a2*=xdr;
        a3*=xdr;

        /// Eventually we will recover an SU(3) matrix via left-multiplication of SU(2) matrices embedded in SU(3).
        /// Let us write our SU(2) matrix as
        ///     a   b
        ///     c   d,
        /// with a,b,c,d complex. In the fundamental representation, d=conj(a) and c=-conj(b); therefore an SU(2) matrix
        /// can be specified by 2 complex numbers.
        x00=GCOMPLEX(floatT)(a0,a3);
        x01=GCOMPLEX(floatT)(a2,a1);
        z1 =GSU2<floatT>(x00,x01);

        /// SECOND SU(2) SUBGROUP: COMPUTE LOCAL MAX
        a0 =  real(v.getLink00()) + real(v.getLink22());
        a1 =  imag(v.getLink20()) + imag(v.getLink02());
        a2 = -real(v.getLink20()) + real(v.getLink02());
        a3 =  imag(v.getLink00()) - imag(v.getLink22());

        asq  = a1*a1 + a2*a2 + a3*a3;
        a0sq = a0*a0;

        x = (relax*a0sq+asq)/(a0sq+asq);
        r = sqrt(a0sq+x*x*asq);
        xdr = x/r;

        a0/=r;
        a1*=xdr;
        a2*=xdr;
        a3*=xdr;

        x00=GCOMPLEX(floatT)(a0,a3);
        x01=GCOMPLEX(floatT)(a2,a1);
        z2 =GSU2<floatT>(x00,x01);

        /// THIRD SU(2) SUBGROUP: COMPUTE LOCAL MAX
        a0 =  real(v.getLink11()) + real(v.getLink22());
        a1 =  imag(v.getLink21()) + imag(v.getLink12());
        a2 = -real(v.getLink21()) + real(v.getLink12());
        a3 =  imag(v.getLink11()) - imag(v.getLink22());

        asq  = a1*a1 + a2*a2 + a3*a3;
        a0sq = a0*a0;

        x = (relax*a0sq+asq)/(a0sq+asq);
        r = sqrt(a0sq+x*x*asq);
        xdr = x/r;

        a0/=r;
        a1*=xdr;
        a2*=xdr;
        a3*=xdr;

        x00=GCOMPLEX(floatT)(a0,a3);
        x01=GCOMPLEX(floatT)(a2,a1);
        z3=GSU2<floatT>(x00,x01);

        /// Recover the OR SU(3) matrix
        g=gsu3_one<floatT>();
        g=sub12(z1,g);
        g=sub13(z2,g);
        g=sub23(z3,g);

        /// OR update: Apply g to U_{mu}(site) and U_{mu}(site-hat{mu})
        for( int mu=0; mu<4; mu++){
            gaugeAccessor.setLink(GInd::getSiteMu(site, mu),
                                  g*gaugeAccessor.getLink(GInd::getSiteMu(site, mu)));
            gaugeAccessor.setLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu),
                                gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu))*dagger(g));
        }
    }
};

/// Compute GF functional. This is what will be maximized for the OR
template<class floatT, bool onDevice, size_t HaloDepth>
floatT GaugeFixing<floatT,onDevice,HaloDepth>::getAction() {
    _redBase.template iterateOverBulk<All,HaloDepth>(GFActionKernel<floatT,HaloDepth>(_gauge));
    floatT gfa;
    _redBase.reduce(gfa, elems);
    floatT vol=GInd::getLatData().globvol4;
    gfa /= (D_FIX*3.*vol);
    return gfa;
}

/// Compute GF theta. This is what determines whether we are sufficiently gauge fixed.
template<class floatT, bool onDevice, size_t HaloDepth>
floatT GaugeFixing<floatT,onDevice,HaloDepth>::getTheta() {
    _redBase.template iterateOverBulk<All,HaloDepth>(GFThetaKernel<floatT,HaloDepth>(_gauge));
    floatT gft;
    _redBase.reduce(gft, elems);
    floatT vol=GInd::getLatData().globvol4;
    gft /= (D_FIX*3.*vol);
    return gft;
}

/// Checkerboard gauge fixing step using over-relaxation.
template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeFixing<floatT,onDevice,HaloDepth>::gaugefixOR() {
    gfixReadIndexEvenOddFull<Even,HaloDepth> calcReadIndexEven;
    gfixReadIndexEvenOddFull<Odd, HaloDepth> calcReadIndexOdd;
    /// OR update red sites.
    iterateFunctorNoReturn<onDevice>(GFORKernel<floatT,Even,HaloDepth>(_gauge),calcReadIndexEven,ORelems);
    _gauge.updateAll();
    /// OR update black sites.
    iterateFunctorNoReturn<onDevice>(GFORKernel<floatT,Odd, HaloDepth>(_gauge),calcReadIndexOdd, ORelems);
    _gauge.updateAll();
}



/// Initialize various possibilities of template parameter combinations for the class GaugeFixing, as well as for most
/// of the above kernels. It is crucial that you do this for templated objects inside of *.cpp files.
#define CLASS_INIT(floatT,HALO) \
template class GaugeFixing<floatT,true,HALO>;
INIT_PH(CLASS_INIT)
