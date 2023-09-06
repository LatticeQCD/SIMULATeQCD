#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "fullSpinorfield.h"
#include "gammaMatrix.h"
#include "../modules/observables/fieldStrengthTensor.h"
#include "biCG.h"

struct WilsonParameters : LatticeParameters {
    Parameter<double> kappa;
    Parameter<double> c_sw;

    // constructor
    WilsonParameters() {
        addDefault(kappa, "kappa", 0.125);
        addDefault(c_sw, "c_sw", 1.0 );
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
    using SpinorLHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    SpinorLHS spinor_lhs(commBase);
    SpinorRHS spinor_rhs(commBase);

    
    
    const std::string& format = param.format();
    std::string Gaugefile = param.GaugefileName();
    //Our gaugefield
    if (format == "nersc") {
        gauge.readconf_nersc(Gaugefile);
    } else if (format == "ildg") {
        gauge.readconf_ildg(Gaugefile);
    } else if (format == "milc") {
        gauge.readconf_milc(Gaugefile);
    } else if (format == "openqcd") {
        gauge.readconf_openqcd(Gaugefile);
    } else {
        throw (std::runtime_error(rootLogger.fatal("Invalid specification for format ", format)));
    }

    /*if(param.format=="ildg"){
      gauge.readconf_ildg(param.GaugefileName());
    }
    else if(param.format=="openqcd"){
      gauge.readconf_openqcd(param.GaugefileName());
    }*/
    gauge.updateAll();

    WilsonDslash<PREC, true, HaloDepth, SpinorLHS, SpinorRHS > wDslash(gauge, param.kappa(), param.c_sw());

    BiCGStabInverter<PREC, true, HaloDepth, SpinorLHS> bicg;

    grnd_state<true> d_rand;
    initialize_rng(1337, d_rand);
    
//    gauge.gauss(d_rand.state);
//    spinor_res.gauss(d_rand.state);
    spinor_rhs.gauss(d_rand.state);



    StopWatch<true> timer;
    timer.start();
    //spinor_res.template iterateOverBulk(TestKernel<PREC, HaloDepth>(gauge, spinor_in));
    //spinor_res.template iterateOverBulk(WilsonDslashKernel<PREC, All, All, HaloDepth, HaloDepth>(spinor_in, gauge));
    //spinor_res.template iterateOverBulk(WilsonDslashKernel<PREC, Even, Odd, HaloDepth, HaloDepth>(spinor_in, gauge, param.kappa(), param.c_sw()));
    bicg.invert(wDslash, spinor_lhs, spinor_rhs, 1000, 1e-8); 
    timer.stop();
    timer.print("Test Kernel runtime");
    
    return 0;
}
