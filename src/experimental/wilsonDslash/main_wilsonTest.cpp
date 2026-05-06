#include "../../simulateqcd.h"
#include "fullSpinor.h"
#include "fullSpinorfield.h"
#include "gammaMatrix.h"
#include "wilsonPropagator.h"
#include "../../modules/observables/fieldStrengthTensor.h"
#include "biCG.h"
#include "../../base/IO/fileWriter.h"

struct WilsonParameters : LatticeParameters {
    Parameter<double> kappa;
    Parameter<double> c_sw;

    // constructor
    WilsonParameters() {
        addDefault(kappa, "kappa", 0.125);
        addDefault(c_sw, "c_sw", 1.0 );
    }
};

//template<class floatT, size_t HaloDepth>


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    CommunicationBase commBase(&argc, &argv);
    WilsonParameters param;
    param.readfile(commBase, "../parameter/tests/wilsonTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;

    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    
    using PREC = double;
    
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    //FullSpinorfield<PREC, true,HaloDepth> spinor_res(commBase);
    //FullSpinorfield<PREC, true,HaloDepth> spinor_in(commBase);
    using SpinorLHS_cpu=Spinorfield<PREC, false, All, HaloDepth, 12>;
    using SpinorLHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    using SpinorEVEN=Spinorfield<PREC, true, Even, HaloDepth, 12>;
    using SpinorODD=Spinorfield<PREC, true, Odd, HaloDepth, 12>;
    SpinorLHS spinor_lhs(commBase);
    SpinorRHS spinor_rhs(commBase);
    SpinorEVEN spinor_src(commBase);
    SpinorEVEN spinor_out(commBase);
    SpinorEVEN spinor_odd(commBase);
    
    gauge.readconf(param.GaugefileName(), param.format());
    gauge.updateAll();

//    WilsonPropagator<PREC> point_source;
//    sitexyzt origin(0,0,0,0);
//    point_source.point_source(origin);

//    SpinorRHS<PREC, true, All, HaloDepthSpin,12> src(commBase); //! even source

    SpinorLHS propagator1(commBase);
    SpinorLHS propagator2(commBase);
    SpinorLHS propagator3(commBase);
    SpinorLHS propagator4(commBase);
    SpinorLHS propagator5(commBase);
    SpinorLHS propagator6(commBase);
    SpinorLHS propagator7(commBase);
    SpinorLHS propagator8(commBase);
    SpinorLHS propagator9(commBase);
    SpinorLHS propagator10(commBase);
    SpinorLHS propagator11(commBase);
    SpinorLHS propagator12(commBase);
    //print spinor_rhs
    sitexyzt origin(0,0,0,0);
    SpinorRHS point_src1(commBase);
    point_src1.setPointSource(origin,0,1.0);
    SpinorRHS point_src2(commBase);
    point_src2.setPointSource(origin,1,1.0);
    SpinorRHS point_src3(commBase);
    point_src3.setPointSource(origin,2,1.0);
    SpinorRHS point_src4(commBase);
    point_src4.setPointSource(origin,3,1.0);
    SpinorRHS point_src5(commBase);
    point_src5.setPointSource(origin,4,1.0);
    SpinorRHS point_src6(commBase);
    point_src6.setPointSource(origin,5,1.0);
    SpinorRHS point_src7(commBase);
    point_src7.setPointSource(origin,6,1.0);
    SpinorRHS point_src8(commBase);
    point_src8.setPointSource(origin,7,1.0);
    SpinorRHS point_src9(commBase);
    point_src9.setPointSource(origin,8,1.0);
    SpinorRHS point_src10(commBase);
    point_src10.setPointSource(origin,9,1.0);
    SpinorRHS point_src11(commBase);
    point_src11.setPointSource(origin,10,1.0);
    SpinorRHS point_src12(commBase);
    point_src12.setPointSource(origin,11,1.0);

//    PointSource<PREC,HaloDepth> pt_source(origin, spinor_rhs,spin,color);
//    printf("%d %d %lf %lf\n",spinorACC.getColorVect(origin));
    WilsonDslash<PREC, true, HaloDepth, SpinorLHS, SpinorRHS > wDslash(gauge, param.kappa(), param.c_sw());
    WilsonDslashEven<PREC, true, HaloDepth, SpinorEVEN, SpinorEVEN > wDslashEven(gauge, param.kappa(), param.c_sw());

    BiCGStabInverter<PREC, true, HaloDepth, SpinorLHS> bicg;
 //   BiCGStabInverterEven<PREC, true, HaloDepth, SpinorEVEN> bicgE;

    grnd_state<true> d_rand;
    initialize_rng(1337, d_rand);
    
//    gauge.gauss(d_rand.state);
//    spinor_res.gauss(d_rand.state);
//    spinor_rhs.gauss(d_rand.state);

//    spinor_rhs.setPointSource(


    StopWatch<true> timer;
    timer.start();
    //spinor_lhs.template iterateOverBulk(TestKernel<PREC, HaloDepth>(spinor_lhs, spinor_rhs));
    //spinor_res.template iterateOverBulk(WilsonDslashKernel<PREC, All, All, HaloDepth, HaloDepth>(spinor_in, gauge));
    //spinor_res.template iterateOverBulk(WilsonDslashKernel<PREC, Even, Odd, HaloDepth, HaloDepth>(spinor_in, gauge, param.kappa(), param.c_sw()));
    bicg.invert(wDslash, propagator1, point_src1, 1000, 1e-8); 
    bicg.invert(wDslash, propagator2, point_src2, 1000, 1e-8); 
    bicg.invert(wDslash, propagator3, point_src3, 1000, 1e-8); 
    bicg.invert(wDslash, propagator4, point_src4, 1000, 1e-8); 
    bicg.invert(wDslash, propagator5, point_src5, 1000, 1e-8); 
    bicg.invert(wDslash, propagator6, point_src6, 1000, 1e-8); 
    bicg.invert(wDslash, propagator7, point_src7, 1000, 1e-8); 
    bicg.invert(wDslash, propagator8, point_src8, 1000, 1e-8); 
    bicg.invert(wDslash, propagator9, point_src9, 1000, 1e-8); 
    bicg.invert(wDslash, propagator10, point_src10, 1000, 1e-8); 
    bicg.invert(wDslash, propagator11, point_src11, 1000, 1e-8); 
    bicg.invert(wDslash, propagator12, point_src12, 1000, 1e-8); 
//    bicg.invert(wDslash, spinor_lhs, spinor_rhs, 1000, 1e-8); 
//    bicgE.invert(wDslashEven, spinor_src, spinor_out, 1000, 1e-8); 
    
    SpinorLHS_cpu spinor_lhs_cpu(commBase);
    typedef GIndexer<All, HaloDepth> Gind;
    typedef Vect<PREC, 12> Vect12;
    spinor_lhs_cpu = propagator1;
    VectArrayAcc<PREC,12> spinorAcc = spinor_lhs_cpu.getAccessor();
    for(size_t t = 0 ; t < Gind::getLatData().lt ; t++){
      for(size_t z = 0 ; z < Gind::getLatData().lz ; z++){
        for(size_t y = 0 ; y < Gind::getLatData().ly ; y++){
          for(size_t x = 0 ; x < Gind::getLatData().lx ; x++){
            gSiteStack site = Gind::getSiteStack(x,y,z,t,0);
            Vect12 vec = spinorAcc.template getElement<PREC>(site);

            rootLogger.info(x," ",y," ",z," ",t," : " ,vec);
          }
        }
      }
    }

    timer.stop();
    timer.print("Test Kernel runtime");
    
    //propagator file write
    std::stringstream filename;
    filename << "propagator";
//    FileWriter file_prop(commBase, param, filename);
        



    return 0;
}
