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
    using SpinorLHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<PREC, true, All, HaloDepth, 12>;
    SpinorLHS spinor_lhs(commBase);
    SpinorRHS spinor_rhs(commBase);
    
    spinor_lhs.zero();
    spinor_rhs.one();

    gauge.readconf(param.GaugefileName(), param.format());
    gauge.updateAll();

    WilsonDslash<PREC, true, HaloDepth, SpinorLHS, SpinorRHS > wDslash(gauge, param.kappa(), param.c_sw());

    wDslash.apply(spinor_lhs, spinor_rhs); // lhs = D*rhs
    
    
    using SpinorLHS_cpu=Spinorfield<PREC, false, All, HaloDepth, 12>;
    SpinorLHS_cpu spinor_lhs_cpu(commBase);
    typedef GIndexer<All, HaloDepth> Gind;
    typedef Vect<PREC, 12> Vect12;
    spinor_lhs_cpu = spinor_lhs;
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

    return 0;
}
