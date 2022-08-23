#include "../SIMULATeQCD.h"

struct CheckParams : LatticeParameters {
    Parameter<std::string> prec;

    CheckParams() {
        addDefault(prec, "prec", std::string("double"));
    }
};

template<class floatT, size_t HaloDepth, CompressionType comp=R18>
struct do_check_unitarity
{
    explicit do_check_unitarity(Gaugefield<floatT,false,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()) {};
    gaugeAccessor<floatT, comp> gAcc;
    HOST_DEVICE floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;
        floatT ret=0.0;
        for (size_t mu = 0; mu < 4; ++mu)
        {
            gSiteMu siteM = GInd::getSiteMu(site, mu);
            ret += tr_d(gAcc.getLinkDagger(siteM)*gAcc.getLink(siteM));
        }
        return ret/4.0;
    }
};

template <class floatT, size_t HaloDepth>
void check_unitarity(Gaugefield<floatT,false,HaloDepth> &gauge)
{
    LatticeContainer<false,floatT> unitarity(gauge.getComm());
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    unitarity.adjustSize(elems);
    unitarity.template iterateOverBulk<All, HaloDepth>(do_check_unitarity<floatT, HaloDepth>(gauge));
    floatT unit_norm;
    unitarity.reduce(unit_norm, elems);
    unit_norm /= static_cast<floatT>(GIndexer<All,HaloDepth>::getLatData().globvol4);
    rootLogger.info("Average unitarity norm <Tr(U^+U)>=" , std::fixed, std::setprecision(std::numeric_limits<floatT>::digits10 + 1),  unit_norm);
    if (!isApproximatelyEqual(unit_norm, static_cast<floatT>(3.0))){
        throw std::runtime_error(rootLogger.fatal("<Tr(U^+U)> != 3"));
    }
}

template<typename floatT, size_t HaloDepth>
void CheckConf(CommunicationBase &commBase, const std::string& format, std::string Gaugefile){
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
    check_unitarity<floatT,HaloDepth>(gauge);

    GaugeAction<floatT, false, HaloDepth> gaugeAction(gauge);
    floatT plaq = gaugeAction.plaquette();
    rootLogger.info("Plaquette = ", plaq);
    if ( (plaq > 1.0) || (plaq < 0.0) ) {
        throw std::runtime_error(rootLogger.fatal("Plaquette should not be negative or larger than 1."));
    }


}


int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(RESULT);
        rootLogger.setVerbosity(RESULT);
        const size_t HaloDepth = 0;

        CheckParams param;
        param.nodeDim.set({1,1,1,1});
        CommunicationBase commBase(&argc, &argv);

        param.readfile(commBase, "../parameter/applications/CheckConf.param", argc, argv);

        commBase.init(param.nodeDim());
        initIndexer(HaloDepth, param, commBase);
        rootLogger.setVerbosity(INFO);
        rootLogger.info("Checking Gaugefile ", param.GaugefileName());
        rootLogger.setVerbosity(RESULT);

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
    rootLogger.result("Gaugefile OK! (readin, plaquette, unitarity)");
    return 0;
}
