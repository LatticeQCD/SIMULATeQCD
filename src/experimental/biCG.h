#pragma once

#include "../gauge/gaugefield.h"
#include "../spinor/spinorfield.h"
#include "../base/math/simpleArray.h"
#include "../modules/inverter/inverter.h"
#include "../base/latticeContainer.h"

template<class floatT, size_t HaloDepth>
struct ScalarProductKernel{

    //Gauge accessor to access the gauge field
    //SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor_lhs;
    SpinorColorAcc<floatT> _SpinorColorAccessor_rhs;
    using SpinorLHS=Spinorfield<floatT, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<floatT, true, All, HaloDepth, 12>;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    ScalarProductKernel(SpinorLHS &spinor_LHS, SpinorRHS &spinor_RHS) 
                : _SpinorColorAccessor_lhs(spinor_LHS.getAccessor()),
                  _SpinorColorAccessor_rhs(spinor_RHS.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
      
      ColorVect<floatT> spinCol;
      COMPLEX(floatT) res(0.0,0.0);
      ColorVect<floatT> spinCol_lhs = _SpinorColorAccessor_lhs.getColorVect(site);
      ColorVect<floatT> spinCol_rhs = _SpinorColorAccessor_rhs.getColorVect(site);
      for(int i = 0 ; i < 4 ; i++){
          res+= re_dot_prod(spinCol_lhs[i], spinCol_rhs[i]);
      }
        return res;
    }
};
template<class floatT, size_t HaloDepth>
COMPLEX(floatT) ScalarProduct(Spinorfield<floatT, true, All, HaloDepth, 12>& spinor_lhs,Spinorfield<floatT, true, All, HaloDepth, 12>& spinor_rhs){
  LatticeContainer<true,COMPLEX(double)> _redBase(spinor_rhs.getComm());

  _redBase.adjustSize(spinor_lhs.getNumberElements());

  _redBase.template iterateOverBulk<All, HaloDepth>(
      ScalarProductKernel<floatT, HaloDepth>(spinor_lhs, spinor_rhs));

  COMPLEX(double) result = 0;
  _redBase.reduce(result, spinor_lhs.getNumberElements());   
  
    return result; 
}

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

        SU3<floatT> first_term=(gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))));
        SU3<floatT> second_term=gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
        /*
        COMPLEX(floatT) result(0.0,0.0);
        for(int i = 0 ; i < 3 ; i++){
          for(int j = 0 ; j < 3 ; j++){
            result+=first_term(i,j);

            printf("%d %d %lf %lf\n",i,j,result.cREAL,result.cIMAG);
          }
          }*/
  
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
            //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,P_plus.A[i][j].cREAL,P_minus.A[i][j].cREAL);
//          }
//          printf("\n");
//        }

        //floatT bla = 0;
        /* for(int i = 0 ; i < 4 ; i++){ */
        /*   for(int j = 0 ; j < 4 ; j++){ */
        /*      // bla += FourMatrix<floatT>::gamma(0).A[i][j].cIMAG; */
        /*        printf("(%.1f,%.1f) ", FourMatrix<floatT>::gamma(1).A[i][j].cREAL, FourMatrix<floatT>::gamma(1).A[i][j].cIMAG ); */
        /*   } */
        /*  // printf("\n"); */
        /* } */


        //! transport spinor psi(x+mu) to psi(x) with link
//        COMPLEX(floatT) result(0.0,0.0);
//        for(int i = 0 ; i < 4 ; i++){
//          result+=re_dot_prod( (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)))[i],spinorIn.getColorVect(GInd::site_up(site, mu))[i]);
//
//          printf("%d %lf %lf\n",i,result.cREAL,result.cIMAG);
//        }
        Dirac_psi = Dirac_psi - 0.5 * ( first_term * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) )
          //! transport spinor psi(x-mu) to psi(x) with link dagger
          + second_term * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) ) );
      }
      //mass term
      floatT M = 1.0/(2.0*_kappa);
      Dirac_psi = Dirac_psi + M * spinorIn.getColorVect(site);

      ColorVect<floatT> Clover;
      for(int mu = 0 ; mu < 4 ; mu++){
        for(int nu = 0 ; nu < 4 ; nu++){
          if(mu==nu) continue;
             SU3<floatT> Fmunu = FT(site,mu,nu);
          /*
             COMPLEX(floatT) result(0.0,0.0);
             for(int i = 0 ; i < 3 ; i++){
             for(int j = 0 ; j < 3 ; j++){
             result+=Fmunu(i,j);
             }
             }
             printf("%d %d %d %d %lf %lf\n",mu,nu,result.cREAL,result.cIMAG);
             */
          Clover = Clover + (_c_sw/4.0) * (COMPLEX(floatT)(0, -1)) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
          //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,(G[mu]*G[nu]).A[i][j].cREAL,(G[mu]*G[nu]).A[i][j].cIMAG);
//          }
//          printf("\n");
//        }
/*          COMPLEX(floatT) result(0.0,0.0);
          for(int i = 0 ; i < 4 ; i++){
            result+=re_dot_prod( ((G[mu]*G[nu])[i],  ((G[mu]*G[nu])[i]);

            printf("%d %lf %lf\n",i, result.cREAL,result.cIMAG);
          }
          */
        }
      }
      Dirac_psi = Dirac_psi + Clover;
      COMPLEX(floatT) res(0.0,0.0);
      for(int i = 0 ; i < 4 ; i++){
        res+= re_dot_prod(Dirac_psi[i], Dirac_psi[i]);
      }
 //     printf("Dirac %lf %lf\n",res.cREAL,res.cIMAG);
      return convertColorVectToVect12(Dirac_psi);
    }
};

//! Abstract base class for all kinds of Dslash operators that shall enter the inversion
template<typename floatT, bool onDevice, size_t HaloDepth, typename SpinorLHS_t, typename SpinorRHS_t>
class WilsonDslash {
  private:
    Gaugefield<floatT, true, HaloDepth>& _gauge;
    floatT _kappa, _c_sw;
  public:
    //! This shall be a simple call of the Dslash without involving a constant
    WilsonDslash(Gaugefield<floatT, true, HaloDepth>& gauge, floatT kappa, floatT c_sw) : _gauge(gauge),_kappa(kappa),_c_sw(c_sw) {}

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    void apply(SpinorRHS_t & lhs, SpinorRHS_t & rhs, bool update = true){
        auto kernel = WilsonDslashKernel<floatT, All, All, HaloDepth, HaloDepth>(rhs, _gauge, _kappa, _c_sw);
         lhs.template iterateOverBulk(kernel);
 
    };
};

template<class floatT, bool onDevice, int HaloDepth, typename Spinor_t>
class BiCGStabInverter{
public:

    void invert(WilsonDslash<floatT, onDevice, HaloDepth, Spinor_t, Spinor_t>& dslash, Spinor_t& x, Spinor_t& rhs, int max_iter, double precision)
    {
        Spinor_t r0(x.getComm());
        Spinor_t p(x.getComm());
        Spinor_t r(x.getComm());
        
        //should go to member
        Spinor_t Ap(x.getComm());
        Spinor_t As(x.getComm());
        Spinor_t s(x.getComm());
        r0 = rhs;
        p=rhs;
        r=rhs;
        
        x*=0.0;
        Ap=x;
        As=x;
        s=x;

        rootLogger.info("invert start");
        floatT rhsinvnorm=1.0/ScalarProduct(rhs,rhs).cREAL;

        rootLogger.info("rhsinvnorm: ", rhsinvnorm);
        COMPLEX(floatT) rr0 = GPUcomplex<floatT>(ScalarProduct(r,r).cREAL,0.0);
        floatT resnorm=rr0.cREAL*rhsinvnorm;

        for (int i = 0; i < max_iter && resnorm > precision; i++) {
          rootLogger.info("Iteration ", i);
          
          dslash.apply(Ap, p); // Ap:output, p:input; Dslash p = Ap
          COMPLEX(floatT) beta=ScalarProduct(Ap,r0);
          rootLogger.info("beta ", beta.cREAL, beta.cIMAG);
          COMPLEX(floatT) alpha=rr0/beta;
          rootLogger.info("alpha ", alpha.cREAL, alpha.cIMAG);

          floatT eps = abs(beta)/sqrt(ScalarProduct(Ap,Ap).cREAL * ScalarProduct(r0,r0).cREAL);
          rootLogger.info("eps ", eps);
          if(eps < 1e-8) {
                rootLogger.trace("restarting BICGSTAB. eps = " ,  eps);
                // r = r0 = p = b-Ax
                dslash.apply(r0, x);
                r0=-1.0*r0+rhs;
                r=r0;
                p=r0;
                rr0=ScalarProduct(r,r).cREAL;
                resnorm = rr0.cREAL * rhsinvnorm;
                rootLogger.info("resnorm ", resnorm, i);
                continue;
            }

            //s = r-alpha*Ap
            s = r - alpha*Ap;
            const floatT snorm = ScalarProduct(s,s).cREAL;
            rootLogger.info("snorm ", snorm);
            if (snorm < precision * precision){
                x = x + alpha*p;
                r = s;
                resnorm = snorm * rhsinvnorm;
                continue;
            }
            dslash.apply(As, s);

            //omega = (As,s)/(As,As)
            COMPLEX(floatT) omega = ScalarProduct(As,s) / ScalarProduct(As,As).cREAL;

            //x=x+alpha*p+pmega*s
            x = x + (alpha * p) + (omega * s);

            //r = s - omega*As
            r = s - omega * As;
            
            resnorm = ScalarProduct(r,r).cREAL * rhsinvnorm;

            rootLogger.info("ScalarProduct(r,r).cREAL: ", ScalarProduct(r,r).cREAL);
            if (std::isnan(resnorm)){
              rootLogger.fatal("Nan");
            }

            if (resnorm > precision || i == max_iter-1 ){// the last steps are not needed if this is the last iteration
              //beta = alpha/omega * rr'/rr
              beta = 1.0/rr0; //reuse temporary
              rr0=ScalarProduct(r,r0);
              beta = beta * (alpha/omega) * rr0;
              
              p = r + beta*(p - omega*Ap);
            }

        }
        rootLogger.info("residue " ,  resnorm);
    }
};
