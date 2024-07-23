#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "DWilson.h"
#include "source.h"

template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;
    Parameter<double,1>  mass; 
    Parameter<double,1>  csw;

    wilsonParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
        add(mass, "mass");
        add(csw, "csw");
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
    
    typedef GIndexer<All,HaloDepth> GInd;    

    PREC mass = param.mass();
    PREC csw = param.csw();

    // file write
    std::string Name = "pion_m";
    Name.append(std::to_string(param.mass()));
    Name.append("_c");
    Name.append(std::to_string(param.csw()));
    Name.append("_"); 
    Name.append(param.gauge_file());
    Name.append(".txt");
    FileWriter fileOut(commBase, param, Name);

    // timer
    StopWatch<true> timer;
    timer.start();

    // set up containers
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);

    std::string file_path = param.gauge_file_folder();
    file_path.append(param.gauge_file());

    gauge.readconf_nersc(file_path);
    gauge.updateAll();


    // make source
    Source source;
    source.makePointSource(spinor_in,1,0,0,0);


    // containers for inversion
    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_out(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_in(commBase);

    //last int used if spinorAll has less stacks
    source.copyHalfFromAll(spinorAll_in,spinor_in,0);

    // stack 12 version
    DWilsonInverseShurComplement<PREC,true,2,2,12> _dslashinverseSC12(gauge,mass,csw);
    timer.reset();
    timer.start();
    _dslashinverseSC12.DslashInverseShurComplementClover(spinorAll_out,spinorAll_in,10000,1e-10);
    timer.stop();
    timer.print("Shur test 12");

    // copy back again
    source.copyAllFromHalf(spinor_out,spinorAll_out,0);

    // contract, make this part of the same class
    DWilsonInverse<PREC,true,2,2,12> dslashinverse12(gauge,mass,0.0);

    COMPLEX(PREC) CC[GInd::getLatData().globLT];
    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC[t] =  _dslashinverseSC12.sumXYZ_TrMdaggerM(t,spinor_out,spinor_out);
    }

    for (int t=0; t<GInd::getLatData().globLT; t++){
         rootLogger.info( CC[t]);
    }



    // stack 1 version
    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_out1(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_in1(commBase);

    source.makePointSource(spinor_in,1,0,0,0);

    DWilsonInverseShurComplement<PREC,true,2,2,1> _dslashinverseSC1(gauge,mass,csw);
    timer.reset();
    timer.start();
    for (int j=0; j<12; j+=1){
        source.copyHalfFromAll(spinorAll_in1,spinor_in,j);
        _dslashinverseSC1.DslashInverseShurComplementClover(spinorAll_out1,spinorAll_in1,10000,1e-10);
        source.copyAllFromHalf(spinor_out,spinorAll_out1,j);
    }
    timer.stop();
    timer.print("Shur test 1");

    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC[t] =  _dslashinverseSC1.sumXYZ_TrMdaggerM(t,spinor_out,spinor_out);
    }

    for (int t=0; t<GInd::getLatData().globLT; t++){
        rootLogger.info( CC[t]);
    }


    // stack 4 version
    SpinorfieldAll<PREC,true, HaloDepth, 12, 4> spinorAll_out4(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 4> spinorAll_in4(commBase);

    source.makePointSource(spinor_in,1,0,0,0);

    DWilsonInverseShurComplement<PREC,true,2,2,4> _dslashinverseSC4(gauge,mass,csw);
    timer.reset();
    timer.start();
    for (int j=0; j<12; j+=4){
        source.copyHalfFromAll(spinorAll_in4,spinor_in,j);
        _dslashinverseSC4.DslashInverseShurComplementClover(spinorAll_out4,spinorAll_in4,10000,1e-10);
        source.copyAllFromHalf(spinor_out,spinorAll_out4,j);
    }
    timer.stop();
    timer.print("Shur test 4");

    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC[t] =  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_out,spinor_out);
    }

    for (int t=0; t<GInd::getLatData().globLT; t++){
        rootLogger.info( CC[t]);
        fileOut << t << " " << real(CC[t]) << " " << imag(CC[t]) << "\n";
    }

    //vector test
    spinor_in = spinor_out;
    source.gammaMuRight<PREC,All,2,5>(spinor_out);
    source.gammaMu<PREC,All,2,12,5>(spinor_out);
    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC[t] =  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out);
    }

    for (int t=0; t<GInd::getLatData().globLT; t++){
        rootLogger.info( CC[t]);
    }

    // multi sources
    timer.reset();
    timer.start();

    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC[t] = 0.0;
    }
    for (int px=0; px<GInd::getLatData().globLX; px+= GInd::getLatData().globLX/2){
        for (int py=0; py<GInd::getLatData().globLY; py+= GInd::getLatData().globLY/2){
            for (int pz=0; pz<GInd::getLatData().globLZ; pz+= GInd::getLatData().globLZ/2){
                for (int pt=0; pt<GInd::getLatData().globLT; pt+= GInd::getLatData().globLT/2){
                     source.makePointSource(spinor_in,px,py,pz,pt);

                     for (int j=0; j<12; j+=4){
                         source.copyHalfFromAll(spinorAll_in4,spinor_in,j);
                         _dslashinverseSC4.DslashInverseShurComplementClover(spinorAll_out4,spinorAll_in4,10000,1e-10);
                         source.copyAllFromHalf(spinor_out,spinorAll_out4,j);       
                     }

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC[t] = CC[t] +  _dslashinverseSC4.sumXYZ_TrMdaggerM(((t+pt)%(GInd::getLatData().globLT)),spinor_out,spinor_out);
                     }

                }
            }
        }
    }
    timer.stop();
    timer.print("Shur test 4 multi");
    for (int t=0; t<GInd::getLatData().globLT; t++){
        rootLogger.info( CC[t]/(2.*2.*2.*2.));
    }

    


    return 0;
}
