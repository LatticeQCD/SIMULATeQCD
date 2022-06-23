#include "../SIMULATeQCD.h"

struct CheckParams : LatticeParameters {
    Parameter<std::string> prec;

    CheckParams() {
        addDefault(prec, "prec", std::string("single"));
    }
};

template<typename floatT, size_t HaloDepth>
void CheckConf(CommunicationBase &commBase, std::string format, std::string Gaugefile){
    Gaugefield<floatT, false, HaloDepth> gauge(commBase);
    /// Read in:
    if (format == "nersc") {
        gauge.readconf_nersc(Gaugefile);
    } else if (format == "ildg") {
        gauge.readconf_ildg(Gaugefile);
    } else if (format == "milc") {
        gauge.readconf_milc(Gaugefile);
    } else {
        throw (std::runtime_error(rootLogger.fatal("Invalid specification for format ", format)));
    }
    GaugeAction<floatT, false, HaloDepth> gaugeAction(gauge);
    rootLogger.info("Plaquette = ", gaugeAction.plaquette());
}


int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(INFO);
        const size_t HaloDepth = 0;

        CheckParams param;
        param.nodeDim.set({1,1,1,1});
        CommunicationBase commBase(&argc, &argv);

        param.readfile(commBase, "../parameter/applications/ConfCheck.param", argc, argv);

        commBase.init(param.nodeDim());
        initIndexer(HaloDepth, param, commBase);

        if (param.prec() == "single"){
            CheckConf<float, HaloDepth>(commBase, param.format(), param.GaugefileName());
        } else if (param.prec() == "double") {
            CheckConf<double, HaloDepth>(commBase, param.format(), param.GaugefileName());
        } else {
            throw (std::runtime_error(rootLogger.fatal("Invalid specification for prec ", param.prec())));
        }

    }
    catch (const std::runtime_error &error) {
        return 1;
    }
    return 0;
}
