#pragma once

#include "../gauge/gaugefield.h"
#include "../spinor/spinorfield.h"
#include "../base/math/simpleArray.h"
#include "../modules/inverter/inverter.h"

template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct WilsonDslashKernel {

    //! The functor has to know about all the elements that it needs for computation.
    //! However, it does not need the Spinor, where the result should go (SpinorOut).
    SU3Accessor<floatT> gAcc;
    SpinorColorAcc<floatT> spinorIn;
    floatT _kappa;
    floatT _c_sw;
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;

    //! Use the constructor to initialize the members
    WilsonDslashKernel(
            Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12> &spinorIn,
            Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
            floatT kappa, floatT c_sw
        ) :
      gAcc(gauge.getAccessor()),
      spinorIn(spinorIn.getAccessor()),
      _kappa(kappa), _c_sw(c_sw), 
      FT(gauge.getAccessor())
  {}

    /*! This is the operator() overload that is called to perform the Dslash. This has to have the following design: It
     * takes a gSite, and it returns the object that we want to write. In this case, we want to return a Vect3<floatT>
     * to store it in another spinor.
     */
    __device__ __host__ inline auto operator()(gSite site) const
    {
      //! We need an indexer to access elements. As the indexer knows about the lattice layout, we do not have to
      //! care about even/odd here explicitly. All that is done by the indexer.
      typedef GIndexer<LatLayout, HaloDepthSpin > GInd;

      /// Define temporary spinor that's 0 everywhere
      ColorVect<floatT> Dirac_psi;
      FourMatrix<floatT> I=FourMatrix<floatT>::identity();
      FourMatrix<floatT> G[4];
      for(int mu=0;mu<4;mu++){
        G[mu]=FourMatrix<floatT>::gamma(mu);
      }
      /// loop through all 4 directions and add result to current site
      for (int mu = 0; mu < 4; mu++) {
        FourMatrix<floatT> P_plus = (I+G[mu]);   
        FourMatrix<floatT> P_minus = (I-G[mu]); 
        //! transport spinor psi(x+mu) to psi(x) with link
        Dirac_psi = Dirac_psi - 0.5 * (gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
          * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) )
          //! transport spinor psi(x-mu) to psi(x) with link dagger
          + gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
          * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) ) );
      }
      //mass term
      floatT M = 1.0/(2.0*_kappa);
      Dirac_psi = Dirac_psi + M * spinorIn.getColorVect(site);

      ColorVect<floatT> Clover;
      for(int mu = 0 ; mu < 4 ; mu++){
        for(int nu = 0 ; nu < 4 ; nu++){
          if(mu==nu) continue;
            SU3<floatT> Fmunu = FT(site,mu,nu);
            Clover = Clover + (_c_sw/4.0) * (COMPLEX(floatT)(0, -1)) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
        }
      }
      Dirac_psi = Dirac_psi + Clover;
      return convertColorVectToVect12(Dirac_psi);
    }
};

//! Abstract base class for all kinds of Dslash operators that shall enter the inversion
template<typename floatT, bool onDevice, int HaloDepth, typename SpinorLHS_t, typename SpinorRHS_t>
class WilsonDslash {
  private:
    Gaugefield<floatT, true, HaloDepth>& _gauge;
    double _kappa, _c_sw;
  public:
    //! This shall be a simple call of the Dslash without involving a constant
    WilsonDslash(Gaugefield<floatT, true, HaloDepth>& gauge, floatT kappa, floatT c_sw) : _gauge(gauge),_kappa(kappa),_c_sw(c_sw) {}

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    void apply(SpinorRHS_t & lhs, const SpinorRHS_t & rhs, bool update = true){
         lhs.template iterateOverBulk(WilsonDslashKernel<floatT, All, All, HaloDepth, HaloDepth>(rhs, _gauge, _kappa, _c_sw));
 
    };
};

template<class floatT, bool onDevice, int HaloDepth, typename Spinor_t>
class BiCGStabInverter{
public:

    void invert(WilsonDslash<floatT, onDevice, HaloDepth, Spinor_t, Spinor_t>& dslash, Spinor_t& x, Spinor_t& rhs, int max_iter, double precision);

};

template<class floatT, bool onDevice, int HaloDepth, typename Spinor_t>
void BiCGStabInverter<floatT,onDevice,HaloDepth,Spinor_t>::invert(WilsonDslash<floatT, onDevice, HaloDepth, Spinor_t, Spinor_t>& dslash, Spinor_t& x, Spinor_t& rhs, int max_iter, double precision)
{
    Spinor_t r0(x.getComm());
    Spinor_t r(x.getComm());
    Spinor_t p(x.getComm());
    
    //should go to member
    Spinor_t Ap(x.getComm());
    Spinor_t As(x.getComm());
    Spinor_t s(x.getComm());
    
    x*=0.0;
    r0 = rhs;
    p=rhs;
    r=rhs;

    floatT rhsinvnorm=1.0/rhs.realdotProduct(rhs);
    COMPLEX(floatT) rr0 = GPUcomplex<floatT>(r.realdotProduct(r),0.0);
    floatT resnorm=rr0.cREAL*rhsinvnorm;

    for (int i = 0; i < max_iter && resnorm > precision; i++) {
        dslash.apply(Ap, p); // Ap:output, p:input; Dslash p = Ap
        COMPLEX(floatT) beta=Ap.dotProduct(r0);
        COMPLEX(floatT) alpha=rr0*beta;

        floatT eps = abs(beta)/sqrt(Ap.realdotProduct(Ap) * r0.realdotProduct(r0));
        if(eps < 1e-8) {
          rootLogger.trace("restarting BICGSTAB. eps = " ,  eps);
          // r = r0 = p = b-Ax
          dslash.apply(r0, x);
          r0=-1.0*r0+rhs;
          r=r0;
          p=r0;
          rr0=r.realdotProduct(r);
          resnorm = rr0.cREAL * rhsinvnorm;
          continue;
        }

        //s = r-alpha*Ap
        s = r - alpha*Ap;
        const floatT snorm = s.realdotProduct(s);
        if (snorm < precision * precision){
            x = x + alpha*p;
            r = s;
            resnorm = snorm * rhsinvnorm;
            continue;
        }
        dslash.apply(As, s);
        //omega = (As,s)/(As,As)
        COMPLEX(floatT) omega = As.dotProduct(s) / As.realdotProduct(As);

        //x=x+alpha*p+pmega*s
        x = x + (alpha * p) + (omega * s);

        //r = s - omega*As
        r = s - omega * As;
        resnorm = r.realdotProduct(r) * rhsinvnorm;

        if (std::isnan(resnorm)){
          rootLogger.fatal("Nan");
        }

        if (resnorm > precision || i == max_iter-1 ){// the last steps are not needed if this is the last iteration
          //beta = alpha/omega * rr'/rr
          beta = 1.0/rr0; //reuse temporary
          rr0=r.dotProduct(r0);
          beta = (beta * (alpha/omega) ) * rr0;
          
          p = r + beta*(p - omega*Ap);
        }
        rootLogger.trace("iteration ",i);

    }
    rootLogger.info("residue " ,  resnorm);
}

