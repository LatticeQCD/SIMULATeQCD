/* 
 * configConverter.cu                                                               
 * 
 * R. Larsen, S. Ali, D. Clarke
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double


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

    stdLogger.setVerbosity(INFO);

    LoadParam<PREC> param;

    CommunicationBase commBase(&argc, &argv);

    param.readfile(commBase, "../parameter/test.param", argc, argv);

    commBase.init(param.nodeDim());

    const size_t HaloDepth = 2;

    rootLogger.info("Initialize Lattice");
    initIndexer(HaloDepth,param,commBase);

    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);

    /// Initialize gaugefield with unit-matrices.
    gauge.one();

    std::string gauge_file;

    std::string file_path = param.directory();
    file_path.append(param.gauge_file()); 
    rootLogger.info("Starting from configuration: " ,  file_path);

    gauge.readconf_milc(file_path);

    gauge.updateAll();         
    GaugeAction<PREC,true,HaloDepth> enDensity(gauge);
    PREC SpatialPlaq  = enDensity.plaquetteSS();
    PREC TemporalPlaq = enDensity.plaquette()*2.0-SpatialPlaq;

    std::string info_path = file_path;
    info_path.append(".info");
    milcInfo<PREC> paramMilc;
    rootLogger.info("plaquette SS: " ,  SpatialPlaq ,  "  and info file: " ,  (paramMilc.ssplaq())/3.0);
    rootLogger.info("plaquette ST: "  ,  TemporalPlaq ,  "  and info file: ",   (paramMilc.stplaq())/3.0);
    rootLogger.info("linktr info file: " ,  paramMilc.linktr());
    if(!(abs((paramMilc.ssplaq())/3.0-SpatialPlaq) < 1e-5)){
        throw std::runtime_error(stdLogger.fatal("Error ssplaq!"));
    } else {
        rootLogger.info("Passed ssplaq check");
    }
    if(!(abs((paramMilc.stplaq())/3.0-TemporalPlaq) < 1e-5)){
        throw std::runtime_error(stdLogger.fatal("Error stplaq!"));
    } else {
        rootLogger.info("Passed stplaq check");
    }

    gauge.updateAll();

    file_path.append("_nersc");
    gauge.writeconf_nersc(file_path);

    return 0;
}


//#define PREC double
//#define USE_GPU true
//
//#include "../SIMULATeQCD.h"
//#include <string>
//using namespace std;
//
//int main(int argc, char *argv[]) {
//
//    stdLogger.setVerbosity(DEBUG);
//
//	LatticeParameters param;
//    const size_t HaloDepth = 1;
//    std::stringstream datNameConf;
//    CommunicationBase commBase(&argc, &argv);
//    param.readfile(commBase, "../parameter/NERSC_ILDG_Converter.param", argc, argv);
//	commBase.init(param.nodeDim());
//
//	typedef GIndexer<All,HaloDepth> GInd;
//    initIndexer(HaloDepth,param,commBase);
//    Gaugefield<PREC, USE_GPU, HaloDepth> gauge_field_in(commBase);
//
//    if(param.use_unit_conf())
//        gauge_field_in.one();
//    else{
//        if(param.format()=="nersc")
//            gauge_field_in.readconf_nersc(param.GaugefileName());
//        else if(param.format()=="ildg")
//            gauge_field_in.readconf_ildg(param.GaugefileName());
//        else
//            rootLogger.error("Input configuration format is not supported");
//    }
//    gauge_field_in.updateAll();
//
//    if(param.format_out()=="nersc") {
//        datNameConf << param.measurements_dir()<< "conf_nersc" << param.fileExt();
//        gauge_field_in.writeconf_nersc(datNameConf.str(), 2, param.prec_out());
//    }
//    else if (param.format_out()=="ildg") {
//        datNameConf << param.measurements_dir() << "conf_ildg" << param.fileExt();
//        gauge_field_in.writeconf_ildg(  datNameConf.str(),3,param.prec_out());
//    }
//    else
//        rootLogger.error("Output configurations format can only be nersc or ildg");
//
//    return 0;
//
//}
