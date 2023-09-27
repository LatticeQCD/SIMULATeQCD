/*
 * gfix.cpp
 *
 * D. Clarke
 *
 * Gpu file with kernel definitions for main_gfix, as well as the functions calling these kernels.
 *
 */

#include "gfix.h"

/// Kernel to compute local contribution to GF functional.
template<class floatT,size_t HaloDepth>
struct GFActionKernel{
    SU3Accessor<floatT> SU3Accessor;
    GFActionKernel(Gaugefield<floatT,true,HaloDepth> &gauge) : SU3Accessor(gauge.getAccessor()){
    }
    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        floatT gfa=0.0;
        /// For Coulomb and Landau gauge fixing, the functional to be maximized is ~sum_{x,mu} Re tr U.
        for ( int mu = 0; mu < I_FIX; mu++) {
            gfa+=tr_d(SU3Accessor.getLink(GInd::getSiteMu(site, mu)));
        }
        return gfa;
    }
};

/// Kernel to compute local contribution to GF theta.
template<class floatT,size_t HaloDepth>
struct GFThetaKernel{
    SU3Accessor<floatT> SU3Accessor;
    GFThetaKernel(Gaugefield<floatT,true,HaloDepth>&gauge):SU3Accessor(gauge.getAccessor()){
    }
    __device__ __host__ floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;
        floatT theta=0.0;
        SU3<floatT> delta,temp;
        delta=su3_zero<floatT>();
        for(int mu=0;mu<I_FIX;mu++){
            temp=       SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu))
                 -SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, mu), mu))
                 -      SU3Accessor.getLink(GInd::getSiteMu(site, mu))
                 +SU3Accessor.getLinkDagger(GInd::getSiteMu(site, mu));
            temp=temp-1./3.*tr_c(temp)*su3_one<floatT>();
            delta+=temp;
        }
        theta=tr_d(delta,dagger(delta));
        return theta;
    }
};

/// Kernel to gauge fix via over-relaxation.
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct GFORKernel{
    SU3Accessor<floatT> SU3Accessor;
    GFORKernel(Gaugefield<floatT,true,HaloDepth> &gauge) : SU3Accessor(gauge.getAccessor()){}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        SU3<floatT> v,g;
        SU2<floatT> z1,z2,z3;
        floatT a0,a1,a2,a3,asq,a0sq,x,r,xdr;
        const floatT relax=1.3;
        COMPLEX(floatT) x00,x01;

        v=su3_one<floatT>();
        for( int mu = 0; mu < I_FIX; mu++){
            v+=SU3Accessor.getLinkDagger(GInd::getSiteMu(site, mu));              /// w += U_{mu}(site)^{dagger}
            v+=SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu)); /// w += U_{mu}(site-hat{mu})
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
        x00=COMPLEX(floatT)(a0,a3);
        x01=COMPLEX(floatT)(a2,a1);
        z1 =SU2<floatT>(x00,x01);

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

        x00=COMPLEX(floatT)(a0,a3);
        x01=COMPLEX(floatT)(a2,a1);
        z2 =SU2<floatT>(x00,x01);

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

        x00=COMPLEX(floatT)(a0,a3);
        x01=COMPLEX(floatT)(a2,a1);
        z3=SU2<floatT>(x00,x01);

        /// Recover the OR SU(3) matrix
        g=su3_one<floatT>();
        g=sub12(z1,g);
        g=sub13(z2,g);
        g=sub23(z3,g);

        /// OR update: Apply g to U_{mu}(site) and U_{mu}(site-hat{mu})
        for( int mu=0; mu<4; mu++){
            SU3Accessor.setLink(GInd::getSiteMu(site, mu),
                                  g*SU3Accessor.getLink(GInd::getSiteMu(site, mu)));
            SU3Accessor.setLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu),
                                SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu))*dagger(g));
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


////////////////////////
// Rasmus Larsen 10 2020
//Maximal center group gauge implemented using 
//https://arxiv.org/pdf/hep-lat/9906010v1.pdf
///////

/// Kernel to compute local contribution to GF R.
template<class floatT,size_t HaloDepth>
struct GFRKernel{
    SU3Accessor<floatT> gaugeAccessor;
    GFRKernel(Gaugefield<floatT,true,HaloDepth>&gauge):gaugeAccessor(gauge.getAccessor()){
    }
    __device__ __host__ floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;
        floatT RVal=0.0;
        SU3<floatT> temp;
        for(int mu=0;mu<4;mu++){
            temp=gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
            floatT val = abs(tr_c(temp));
            RVal=RVal+val*val;
        }
        return RVal;
    }
};

// Project to center
template<class floatT,Layout LatLayout,size_t HaloDepth>
struct ProjectZ{

   SU3Accessor<floatT> gAcc;
   SU3Accessor<floatT> gAcc2;

   ProjectZ(Gaugefield<floatT,true,HaloDepth> &gauge,Gaugefield<floatT,true,HaloDepth> &gauge2) : gAcc(gauge.getAccessor()), gAcc2(gauge2.getAccessor()){}

   __device__ __host__ void operator()(gSite site) {
      typedef GIndexer<LatLayout,HaloDepth> GInd;


      SU3<floatT> link = su3_one<floatT>();


      for( int mu=0; mu<4; mu++){
         COMPLEX(floatT) val = tr_c(gAcc.getLink(GInd::getSiteMu(site, mu)))/3.0;
         floatT angle = asin(val.cIMAG/sqrt(val.cIMAG*val.cIMAG+val.cREAL*val.cREAL));
         //printf("%f \n", angle);
         if(val.cREAL < 0.0 && val.cIMAG > 0.0){
             angle = 3.14159-angle;
         }
         else if(val.cREAL < 0.0 && val.cIMAG < 0.0){
             angle = -3.14159-angle;
         }
         //printf("2 %f \n", angle);

         if(angle < - 1.0472){
            val = COMPLEX(floatT)(-0.5, -0.866025);
         }
         else if(angle >  1.0472){
            val = COMPLEX(floatT)(-0.5, 0.866025);
         }
         else{
            val = COMPLEX(floatT)(1.0, 0.0);
         }

         gAcc2.setLink(GInd::getSiteMu(site, mu), val*link);
         gAcc.setLink(GInd::getSiteMu(site, mu), gAcc.getLink(GInd::getSiteMu(site, mu))*dagger(gAcc2.getLink(GInd::getSiteMu(site, mu))));
      }

   }
};


template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeFixing<floatT,onDevice,HaloDepth>::projectZ(Gaugefield<floatT,onDevice,HaloDepth> &gauge2) {
    gfixReadIndexEvenOddFull<All,HaloDepth> calcReadIndex;

    iterateFunctorNoReturn<onDevice>(ProjectZ<floatT,All,HaloDepth>(_gauge, gauge2),calcReadIndex,elems);
    _gauge.updateAll();
    gauge2.updateAll();
};


/// Kernel to gauge fix R
template<class floatT,Layout LatLayout,size_t HaloDepth, size_t nu>
struct GFRORKernel{
    SU3Accessor<floatT> gaugeAccessor;
    GFRORKernel(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        SU3<floatT> v,g, g0,g1,g2,g3;
        SU2<floatT> z1,z2,z3, z0;
        COMPLEX(floatT) d0[4*8], e0[8];
        COMPLEX(floatT) x00,x01;
        COMPLEX(floatT) im1(0.0,1.0);

        x00=COMPLEX(floatT)(1.0,-0.0);
        x01=COMPLEX(floatT)(-0.0,-0.0);
        z0 =SU2<floatT>(x00,x01);
        x00=COMPLEX(floatT)(0.0,-0.0);
        x01=COMPLEX(floatT)(-0.0,-1.0);
        z1 =SU2<floatT>(x00,x01);
        x00=COMPLEX(floatT)(0.0,-0.0);
        x01=COMPLEX(floatT)(-1.0,-0.0);
        z2 =SU2<floatT>(x00,x01);
        x00=COMPLEX(floatT)(0.0,-1.0);
        x01=COMPLEX(floatT)(-0.0,-0.0);
        z3 =SU2<floatT>(x00,x01);



        g=su3_one<floatT>();

        if(nu==0){
            g.setLink22(0.0);
            g0=sub12(z0,g);
            g1=sub12(z1,g);
            g2=sub12(z2,g);
            g3=sub12(z3,g);

        }
        if(nu==1){
            g.setLink11(0.0);
            g0=sub13(z0,g);
            g1=sub13(z1,g);
            g2=sub13(z2,g);
            g3=sub13(z3,g);
        }
        if(nu==2){
            g.setLink00(0.0);
            g0=sub23(z0,g);
            g1=sub23(z1,g);
            g2=sub23(z2,g);
            g3=sub23(z3,g);
        }



    if(nu==0){
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
/*
            d0[mu+8*0] = (v.getLink00() + v.getLink11())/2.0;
            d0[mu+8*1] = -im1*(v.getLink10() + v.getLink01())/2.0;
            d0[mu+8*2] = (-v.getLink10() + v.getLink01())/2.0;
            d0[mu+8*3] = -im1*(v.getLink00() - v.getLink11())/2.0;
*/

            d0[mu+8*0] = tr_c(g0*v)/2.0;
            d0[mu+8*1] = tr_c(g1*v)/2.0;
            d0[mu+8*2] = tr_c(g2*v)/2.0;
            d0[mu+8*3] = tr_c(g3*v)/2.0;

            e0[mu] =  v.getLink22();
        }
/*
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu));
            d0[mu+8*0+4] = (v.getLink00() + v.getLink11())/2.0;
            d0[mu+8*1+4] = -im1*(-v.getLink10() - v.getLink01())/2.0;
            d0[mu+8*2+4] = (v.getLink10() - v.getLink01())/2.0;
            d0[mu+8*3+4] = -im1*(-v.getLink00() + v.getLink11())/2.0;

            e0[mu+4] =  v.getLink22();
        }
*/

        for( int mu = 0; mu < 4; mu++){
            v=dagger(gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
/*
            d0[mu+8*0+4] = (v.getLink00() + v.getLink11())/2.0;
            d0[mu+8*1+4] = -im1*(v.getLink10() + v.getLink01())/2.0;
            d0[mu+8*2+4] = (-v.getLink10() + v.getLink01())/2.0;
            d0[mu+8*3+4] = -im1*(v.getLink00() - v.getLink11())/2.0;
*/

            d0[mu+8*0+4] = tr_c(g0*v)/2.0;
            d0[mu+8*1+4] = tr_c(g1*v)/2.0;
            d0[mu+8*2+4] = tr_c(g2*v)/2.0;
            d0[mu+8*3+4] = tr_c(g3*v)/2.0;


            e0[mu+4] =  v.getLink22();
        }

    }


    if(nu==1){
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
/*            d0[mu+8*0] = (v.getLink00() + v.getLink22())/2.0;
            d0[mu+8*1] = -im1*(v.getLink20() + v.getLink02())/2.0;
            d0[mu+8*2] = (-v.getLink20() + v.getLink02())/2.0;
            d0[mu+8*3] = -im1*(v.getLink00() - v.getLink22())/2.0;
*/
            d0[mu+8*0] = tr_c(g0*v)/2.0;
            d0[mu+8*1] = tr_c(g1*v)/2.0;
            d0[mu+8*2] = tr_c(g2*v)/2.0;
            d0[mu+8*3] = tr_c(g3*v)/2.0;

            e0[mu] =  v.getLink11();
        }

/*
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu));
            d0[mu+8*0+4] = (v.getLink00() + v.getLink22())/2.0;
            d0[mu+8*1+4] = -im1*(-v.getLink20() - v.getLink02())/2.0;
            d0[mu+8*2+4] = (v.getLink20() - v.getLink02())/2.0;
            d0[mu+8*3+4] = -im1*(-v.getLink00() + v.getLink22())/2.0;

            e0[mu+4] =  v.getLink11();
        }
*/
        for( int mu = 0; mu < 4; mu++){
            v=dagger(gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
/*            d0[mu+8*0+4] = (v.getLink00() + v.getLink22())/2.0;
            d0[mu+8*1+4] = -im1*(v.getLink20() + v.getLink02())/2.0;
            d0[mu+8*2+4] = (-v.getLink20() + v.getLink02())/2.0;
            d0[mu+8*3+4] = -im1*(v.getLink00() - v.getLink22())/2.0;
*/
            d0[mu+8*0+4] = tr_c(g0*v)/2.0;
            d0[mu+8*1+4] = tr_c(g1*v)/2.0;
            d0[mu+8*2+4] = tr_c(g2*v)/2.0;
            d0[mu+8*3+4] = tr_c(g3*v)/2.0;

            e0[mu+4] =  v.getLink11();
        }


    }



    if(nu==2){
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
/*            d0[mu+8*0] = (v.getLink11() + v.getLink22())/2.0;
            d0[mu+8*1] = -im1*(v.getLink21() + v.getLink12())/2.0;
            d0[mu+8*2] = (-v.getLink21() + v.getLink12())/2.0;
            d0[mu+8*3] = -im1*(v.getLink11() - v.getLink22())/2.0;
*/
            d0[mu+8*0] = tr_c(g0*v)/2.0;
            d0[mu+8*1] = tr_c(g1*v)/2.0;
            d0[mu+8*2] = tr_c(g2*v)/2.0;
            d0[mu+8*3] = tr_c(g3*v)/2.0;

            e0[mu] =  v.getLink00();
        }
/*
        for( int mu = 0; mu < 4; mu++){
            v=gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu));
            d0[mu+8*0+4] = (v.getLink11() + v.getLink22())/2.0;
            d0[mu+8*1+4] = -im1*(-v.getLink21() - v.getLink12())/2.0;
            d0[mu+8*2+4] = (v.getLink21() - v.getLink12())/2.0;
            d0[mu+8*3+4] = -im1*(-v.getLink11() + v.getLink22())/2.0;

            e0[mu+4] =  v.getLink00();
        }
*/

        for( int mu = 0; mu < 4; mu++){
            v=dagger(gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
/*            d0[mu+8*0+4] = (v.getLink11() + v.getLink22())/2.0;
            d0[mu+8*1+4] = -im1*(v.getLink21() + v.getLink12())/2.0;
            d0[mu+8*2+4] = (-v.getLink21() + v.getLink12())/2.0;
            d0[mu+8*3+4] = -im1*(v.getLink11() - v.getLink22())/2.0;
*/
            d0[mu+8*0+4] = tr_c(g0*v)/2.0;
            d0[mu+8*1+4] = tr_c(g1*v)/2.0;
            d0[mu+8*2+4] = tr_c(g2*v)/2.0;
            d0[mu+8*3+4] = tr_c(g3*v)/2.0;

            e0[mu+4] =  v.getLink00();
        }


    }


//////////////////

        floatT  bV[4], cS, eigenVal[4];
        Matrix4x4<floatT> aM, eigenVec;
        for( int i = 0; i < 4; i++){
           for( int j = 0; j < 4; j++){
              for( int l = 0; l < 8; l++){
                  aM.a[i+4*j] = aM.a[i+4*j] + real(d0[l+8*i]*conj(d0[l+8*j])+ d0[l+8*j]*conj(d0[l+8*i]));
              }
           }
           bV[i]=0.0;
           for( int l = 0; l < 8; l++){
                  bV[i] = bV[i] - 0.5*real(e0[l]*conj(d0[l+8*i])+ conj(e0[l])*d0[l+8*i]);
           }
        }
        cS= 0.0;
        for( int l = 0; l < 8; l++){
            cS = cS + 0.25*real(e0[l]*conj(e0[l]));
        }

        QR(eigenVec,eigenVal,aM);

  //      floatT vecb[4];
        floatT vecg[4];
        getSU2Rotation(vecg,eigenVal, bV, eigenVec);


        /// Eventually we will recover an SU(3) matrix via left-multiplication of SU(2) matrices embedded in SU(3).
        /// Let us write our SU(2) matrix as
        ///     a   b
        ///     c   d,
        /// with a,b,c,d complex. In the fundamental representation, d=conj(a) and c=-conj(b); therefore an SU(2) matrix
        /// can be specified by 2 complex numbers.
        //printf("vecg %lu  %f %f %f %f \n",site.isiteFull ,vecg[0], vecg[1], vecg[2], vecg[3]);


        x00=COMPLEX(floatT)(vecg[0],-vecg[3]);
        x01=COMPLEX(floatT)(-vecg[2],-vecg[1]);
        z1 =SU2<floatT>(x00,x01);

        /// Recover the OR SU(3) matrix
        g=su3_one<floatT>();
        if(nu==0){
            g=sub12(z1,g);
        }
        if(nu==1){
            g=sub13(z1,g);
        }
        if(nu==2){
            g=sub23(z1,g);
        }
        /// OR update: Apply g to U_{mu}(site) and U_{mu}(site-hat{mu})
        for( int mu=0; mu<4; mu++){
            gaugeAccessor.setLink(GInd::getSiteMu(site, mu),
                                  g*gaugeAccessor.getLink(GInd::getSiteMu(site, mu)));
            gaugeAccessor.setLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu),
                                gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site, mu), mu))*dagger(g));
        }

  }

};


template<class floatT, bool onDevice, size_t HaloDepth>
floatT GaugeFixing<floatT,onDevice,HaloDepth>::getR() {
    _redBase.template iterateOverBulk<All,HaloDepth>(GFRKernel<floatT,HaloDepth>(_gauge));
    floatT gft;
    _redBase.reduce(gft, elems);
    floatT vol=GInd::getLatData().globvol4;
    gft /= (4.0*9.*vol);
    return gft;
}


template<class floatT, bool onDevice, size_t HaloDepth>
void GaugeFixing<floatT,onDevice,HaloDepth>::gaugefixR() {
    gfixReadIndexEvenOddFull<Even,HaloDepth> calcReadIndexEven;
    gfixReadIndexEvenOddFull<Odd, HaloDepth> calcReadIndexOdd;
    /// OR update red sites.
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Even,HaloDepth,0>(_gauge),calcReadIndexEven,ORelems);
    _gauge.updateAll();
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Even,HaloDepth,1>(_gauge),calcReadIndexEven,ORelems);
    _gauge.updateAll();
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Even,HaloDepth,2>(_gauge),calcReadIndexEven,ORelems);
    _gauge.updateAll();
    /// OR update black sites.
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Odd, HaloDepth,0>(_gauge),calcReadIndexOdd, ORelems);
    _gauge.updateAll();
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Odd, HaloDepth,1>(_gauge),calcReadIndexOdd, ORelems);
    _gauge.updateAll();
    iterateFunctorNoReturn<onDevice>(GFRORKernel<floatT,Odd, HaloDepth,2>(_gauge),calcReadIndexOdd, ORelems);
    _gauge.updateAll();
}




/// Initialize various possibilities of template parameter combinations for the class GaugeFixing, as well as for most
/// of the above kernels. It is crucial that you do this for templated objects inside of *.cpp files.
#define CLASS_INIT(floatT,HALO) \
template class GaugeFixing<floatT,true,HALO>;
INIT_PH(CLASS_INIT)
