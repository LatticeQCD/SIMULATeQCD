/*
 * configConverter.cu
 *
 * R. Larsen, S. Ali, D. Clarke
 *
 */

#include "../simulateqcd.h"

#define PREC double


struct ConvertParameters : LatticeParameters {
    Parameter<bool> compress_out;
    Parameter<std::string> format_out;

    ConvertParameters() {
        addDefault(compress_out, "compress_out", true);
        addDefault(format_out, "format_out", std::string("nersc"));
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 2;

    ConvertParameters param;
    CommunicationBase commBase(&argc, &argv);

    param.readfile(commBase, "../parameter/applications/configConverter.param", argc, argv);
    if( param.compress_out()==true && param.format_out()=="ildg" ) {
        throw(rootLogger.fatal("ILDG format does not support compression."));
    }

    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);

    Gaugefield<PREC,true,HaloDepth> gauge(commBase);

    /// Read in:
    if(param.format()=="nersc") {
        gauge.readconf_nersc(param.GaugefileName());
    } else if(param.format()=="ildg") {
        gauge.readconf_ildg(param.GaugefileName());
    } else if(param.format()=="milc"){
        gauge.readconf_milc(param.GaugefileName());
    } else if(param.format()=="openqcd"){
        gauge.readconf_openqcd(param.GaugefileName());
    } else {
        throw(rootLogger.fatal("Invalid specification for format ",param.format()));
    }

    /// Print out:
    if(param.format_out()=="nersc") {
        if(param.compress_out()) {
            gauge.writeconf_nersc(param.GaugefileName_out(), 2, param.prec_out());
        } else {
            gauge.writeconf_nersc(param.GaugefileName_out(), 3, param.prec_out());
        }
    } else if(param.format_out()=="ildg") {
        gauge.writeconf_ildg(param.GaugefileName_out(), param);
    } else {
        throw(rootLogger.fatal("Invalid specification for format_out ",param.format_out()));
    }

    return 0;
}
