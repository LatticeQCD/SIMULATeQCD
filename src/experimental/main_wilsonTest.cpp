#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "fullSpinorfield.h"
#include "gammaMatrix.h"
#include "../modules/observables/fieldStrengthTensor.h"

struct WilsonParameters : LatticeParameters {
    Parameter<double> kappa;
    Parameter<double> c_sw;

    // constructor
    WilsonParameters() {
        addDefault(kappa, "kappa", 0.125);
        addDefault(c_sw, "c_sw", 1.0 );
    }
};


template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct WilsonQuickDslash {

    //! The functor has to know about all the elements that it needs for computation.
    //! However, it does not need the Spinor, where the result should go (SpinorOut).
    SU3Accessor<floatT> gAcc;
    SpinorColorAcc<floatT> spinorIn;
    floatT _kappa;
    floatT _c_sw;
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;

    //! Use the constructor to initialize the members
    WilsonQuickDslash(
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
        Dirac_psi = Dirac_psi + gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
          * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) )
          //! transport spinor psi(x-mu) to psi(x) with link dagger
          + gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
          * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) );
      }

      //mass term
      floatT M = 1.0/(2.0*_kappa);
      Dirac_psi = Dirac_psi + M * spinorIn.getColorVect(site);

      ColorVect<floatT> Clover;
          

      for(int mu = 0 ; mu < 4 ; mu++){
        for(int nu = 0 ; nu < 4 ; nu++){
          if(mu==nu) continue;
            SU3<floatT> Fmunu = FT(site,mu,nu);
          Clover = Clover + (_c_sw/2.0) * (COMPLEX(floatT)(0, -1)) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
        }
      }
      Dirac_psi = Dirac_psi + Clover;
      return convertColorVectToVect12(Dirac_psi);
    }
};

template<class floatT, size_t HaloDepth>
struct TestKernel{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    TestKernel(Gaugefield<floatT,true,HaloDepth> &gauge, FullSpinorfield<floatT,true,HaloDepth> &spinorIn) 
                : _SU3Accessor(gauge.getAccessor()), 
                  _SpinorColorAccessor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSite site) {

        SU3<floatT> link;
        //Dslash
        FourMatrix<floatT> I=FourMatrix<floatT>::identity();
        FourMatrix<floatT> G[4];
        for(int mu=0;mu<4;mu++){
          G[mu]=FourMatrix<floatT>::gamma(mu);
        }
       
        ColorVect<floatT> spinCol;
        for(int mu = 0 ; mu < 4 ; mu++){
          gSite site_mu_up = GInd::site_up(site, mu);
          gSite site_mu_dn = GInd::site_dn(site, mu);
          ColorVect<floatT> spinCol_mu_up = _SpinorColorAccessor.getColorVect(site_mu_up);
          ColorVect<floatT> spinCol_mu_dn = _SpinorColorAccessor.getColorVect(site_mu_dn);
          SU3<floatT> U_mu = _SU3Accessor.getLink(GInd::getSiteMu(site, mu));
          SU3<floatT> U_mu_dag = dagger(_SU3Accessor.getLink(GInd::getSiteMu(site_mu_dn, mu)));
          FourMatrix<floatT> P_plus = (I+G[mu]);   
          FourMatrix<floatT> P_minus = (I-G[mu]); 
          spinCol = spinCol + U_mu * (P_minus * spinCol_mu_up) + U_mu_dag * (P_plus * spinCol_mu_dn); 
        }
        //for (int nu = 1; nu < 4; nu++) {
        //    for (int mu = 0; mu < nu; mu++) {
        //        link += _SU3Accessor.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu));
        //    }
        //}

        
        /* for (auto& s : spinCol){ */
        /*     s = link*s; */
        /* } */
        //spinCol = link * spinCol;

        return convertColorVectToVect12(spinCol);
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    CommunicationBase commBase(&argc, &argv);
    WilsonParameters param;
    param.readfile(commBase, "../parameter/tests/wilsonTest.param", argc, argv);
    commBase.init(param.nodeDim());
    //const int LatDim[] = {20, 20, 20, 20};
    //const int NodeDim[] = {1, 1, 1, 1};
    //param.latDim.set(LatDim);
    //param.nodeDim.set(NodeDim);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;

    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    
    using PREC = double;
    
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    //FullSpinorfield<PREC, true,HaloDepth> spinor_res(commBase);
    //FullSpinorfield<PREC, true,HaloDepth> spinor_in(commBase);
    Spinorfield<PREC, true, Even, HaloDepth, 12> spinor_res(commBase);
    Spinorfield<PREC, true, Odd, HaloDepth, 12> spinor_in(commBase);

    grnd_state<true> d_rand;
    initialize_rng(1337, d_rand);
    
//    gauge.gauss(d_rand.state);
//    spinor_res.gauss(d_rand.state);
    spinor_in.gauss(d_rand.state);



    StopWatch<true> timer;
    timer.start();
    //spinor_res.template iterateOverBulk(TestKernel<PREC, HaloDepth>(gauge, spinor_in));
    //spinor_res.template iterateOverBulk(WilsonQuickDslash<PREC, All, All, HaloDepth, HaloDepth>(spinor_in, gauge));
    spinor_res.template iterateOverBulk(WilsonQuickDslash<PREC, Even, Odd, HaloDepth, HaloDepth>(spinor_in, gauge, param.kappa(), param.c_sw()));
    timer.stop();
    timer.print("Test Kernel runtime");
    
    return 0;
}
