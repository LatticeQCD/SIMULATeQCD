/* 
 * main_milc_to_nersc.cu                                                               
 * 
 * Rasmus Larsen
 * 
 */

#include "../SIMULATeQCD.h"

#include <iostream>
using namespace std;

#define PREC double
#define MY_BLOCKSIZE 256


template<class floatT>
struct LoadParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> directory;

    LoadParam() {
        add(gauge_file, "gauge_file");
        add(directory, "directory");
    }
};

template<class floatT>
struct milcInfo : ParameterList {
    Parameter<floatT>  ssplaq;
    Parameter<floatT>  stplaq;
    Parameter<floatT>  linktr;

    milcInfo() {
        addDefault(ssplaq, "gauge.ssplaq",0.0);
        addDefault(stplaq, "gauge.stplaq",0.0);
        addDefault(linktr, "gauge.nersc_linktr",0.0);
    }
};


int main(int argc, char *argv[]) {

    /// Controls whether DEBUG statements are shown as it runs; could also set to INFO, which is less verbose.
    stdLogger.setVerbosity(INFO);

    /// Initialize parameter class.
    LoadParam<PREC> param;

    /// Initialize the CommunicationBase.
    CommunicationBase commBase(&argc, &argv);

    param.readfile(commBase, "../parameter/test.param", argc, argv);


    commBase.init(param.nodeDim());


    /// Set the HaloDepth.
    const size_t HaloDepth = 2;

    rootLogger.info() << "Initialize Lattice";

    /// Initialize the Lattice class.
    initIndexer(HaloDepth,param,commBase);

    /// Initialize the Gaugefield.
    rootLogger.info() << "Initialize Gaugefield";
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);

    /// Initialize gaugefield with unit-matrices.
    gauge.one();

    std::string gauge_file;

        std::string file_path = param.directory();
        file_path.append(param.gauge_file()); 
        rootLogger.info() << "Starting from configuration: " << file_path;

            gauge.readconf_milc(file_path);

            gauge.updateAll();         
            GaugeAction<PREC,true,HaloDepth> enDensity(gauge);
            PREC SpatialPlaq  = enDensity.plaquetteSS();
            PREC TemporalPlaq = enDensity.plaquette()*2.0-SpatialPlaq;
//            rootLogger.info() << "plaquetteST: "   << TemporalPlaq;
//            rootLogger.info() << "plaquetteSS: " << SpatialPlaq;



            std::string info_path = file_path;
            info_path.append(".info");
            milcInfo<PREC> paramMilc;
//            paramMilc.readfile(commBase,info_path);
            rootLogger.info() << "plaquette SS: " << SpatialPlaq << "  and info file: " << (paramMilc.ssplaq())/3.0;
            rootLogger.info() << "plaquette ST: "  << TemporalPlaq << "  and info file: "<<  (paramMilc.stplaq())/3.0;
            rootLogger.info() << "linktr info file: " << paramMilc.linktr();
            if(!(abs((paramMilc.ssplaq())/3.0-SpatialPlaq) < 1e-5)){
//                throw PGCError("Error ssplaq!");
            }
            else{
                rootLogger.info() << "Passed ssplaq check";
            }
            if(!(abs((paramMilc.stplaq())/3.0-TemporalPlaq) < 1e-5)){
//                throw PGCError("Error stplaq!");
            }
            else{
                rootLogger.info() << "Passed stplaq check";
            }


        

    /// Exchange Halos
    gauge.updateAll();
/*
   

    /// Initialize ReductionBase.
    LatticeContainer<true,PREC> redBase(commBase);

    /// We need to tell the Reductionbase how large our array will be. Again it runs on the spacelike volume only,
    /// so make sure you adjust this parameter accordingly, so that you don't waste memory.
    typedef GIndexer<All,HaloDepth> GInd;
    redBase.adjustSize(GInd::getLatData().vol4);
    rootLogger.info() << "volume size " << GInd::getLatData().globvol4;
*/

    file_path.append("_nersc");
    gauge.writeconf_nersc(file_path);


    return 0;
}
