//
// Created by Sajid Ali on 12/21/21.
//
// converts NERSC file format to ILDG and vice versa.

#define PREC double
#define USE_GPU true

#include "../simulateqcd.h"
#include <string>
using namespace std;

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;
    const size_t HaloDepth = 1;
    std::stringstream datNameConf;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../../parameter/NERSC_ILDG_Converter.param", argc, argv);
    commBase.init(param.nodeDim());

    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);
    Gaugefield<PREC, USE_GPU, HaloDepth> gauge_field_in(commBase);

    if(param.use_unit_conf())
        gauge_field_in.one();
    else{
        if(param.format()=="nersc")
            gauge_field_in.readconf_nersc(param.GaugefileName());
        else if(param.format()=="ildg")
            gauge_field_in.readconf_ildg(param.GaugefileName());
        else
            rootLogger.error("Input configuration format is not supported");
    }
    gauge_field_in.updateAll();

    if(param.format_out()=="nersc") {
        datNameConf << param.measurements_dir()<< "conf_nersc" << param.fileExt();
        gauge_field_in.writeconf_nersc(datNameConf.str(), 2, param.prec_out());
    }
    else if (param.format_out()=="ildg") {
        datNameConf << param.measurements_dir() << "conf_ildg" << param.fileExt();
        gauge_field_in.writeconf_ildg(datNameConf.str(),3,param.prec_out());
    }
    else
        rootLogger.error("Output configurations format can only be nersc or ildg");

    return 0;

}