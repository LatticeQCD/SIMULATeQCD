#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "DWilson.h"
#include "source.h"

template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter<double,1>  mass; 

    wilsonParam() {
        addOptional(gauge_file, "gauge_file");
        add(mass, "mass");
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    using PREC = double;

    wilsonParam<PREC> param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/test.param", argc, argv);

    commBase.init(param.nodeDim());

    const size_t HaloDepth = 2;

    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
        

    PREC mass = param.mass();

    // timer
    StopWatch<true> timer;
    timer.start();

    // set up containers
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);

    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();


    // make source
    Source source;
    source.makePointSource(spinor_in,0,0,0,0);


    // containers for inversion
    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_out(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_in(commBase);

    //last int used if spinorAll has less stacks
    source.copyHalfFromAll(spinorAll_in,spinor_in,0);

    DWilsonInverseShurComplement<PREC,true,2,2,12> _dslashinverseSC12(gauge,mass,1.0);
    timer.reset();
    timer.start();
    _dslashinverseSC12.DslashInverseShurComplementClover(spinorAll_out,spinorAll_in,10000,1e-10);
    timer.stop();
    timer.print("Shur test 12");

    // copy back again
    source.copyAllFromHalf(spinor_out,spinorAll_out,0);

    // contract, make this part of the same class
    DWilsonInverse<PREC,true,2,2,12> dslashinverse12(gauge,mass,0.0);

    COMPLEX(PREC) CC[20];
    for (int t=0; t<20; t++){
        CC[t] =  dslashinverse12.Correlator(t,spinor_out);
    }

    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }



    // stack 1 version
    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_out1(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_in1(commBase);

    source.makePointSource(spinor_in,0,0,0,0);

    DWilsonInverseShurComplement<PREC,true,2,2,1> _dslashinverseSC1(gauge,mass,1.0);
    timer.reset();
    timer.start();
    for (int j=0; j<12; j+=1){
        source.copyHalfFromAll(spinorAll_in1,spinor_in,j);
        _dslashinverseSC1.DslashInverseShurComplementClover(spinorAll_out1,spinorAll_in1,10000,1e-10);
        source.copyAllFromHalf(spinor_out,spinorAll_out1,j);
    }
    timer.stop();
    timer.print("Shur test 1");

    for (int t=0; t<20; t++){
        CC[t] =  dslashinverse12.Correlator(t,spinor_out);
    }

    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }


    // stack 4 version
    SpinorfieldAll<PREC,true, HaloDepth, 12, 4> spinorAll_out4(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 4> spinorAll_in4(commBase);

    source.makePointSource(spinor_in,0,0,0,0);

    DWilsonInverseShurComplement<PREC,true,2,2,4> _dslashinverseSC4(gauge,mass,1.0);
    timer.reset();
    timer.start();
    for (int j=0; j<12; j+=4){
        source.copyHalfFromAll(spinorAll_in4,spinor_in,j);
        _dslashinverseSC4.DslashInverseShurComplementClover(spinorAll_out4,spinorAll_in4,10000,1e-10);
        source.copyAllFromHalf(spinor_out,spinorAll_out4,j);
    }
    timer.stop();
    timer.print("Shur test 4");

    for (int t=0; t<20; t++){
        CC[t] =  dslashinverse12.Correlator(t,spinor_out);
    }

    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }


    return 0;
}