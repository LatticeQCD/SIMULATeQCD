#include "../simulateqcd.h"

struct CheckParams : LatticeParameters {
    Parameter<std::string> prec;

    CheckParams() {
        addDefault(prec, "prec", std::string("double"));
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp=R18>
struct do_check_unitarity
{
    explicit do_check_unitarity(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()) {};
    SU3Accessor<floatT, comp> gAcc;
    __device__ __host__ floatT operator()(gSite site){
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

template <class floatT, bool onDevice, size_t HaloDepth>
void check_unitarity(Gaugefield<floatT,onDevice,HaloDepth> &gauge)
{
    LatticeContainer<onDevice,floatT> unitarity(gauge.getComm());
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    unitarity.adjustSize(elems);
    unitarity.template iterateOverBulk<All, HaloDepth>(do_check_unitarity<floatT, onDevice, HaloDepth>(gauge));
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
    } else if (format == "openqcd") {
        gauge.readconf_openqcd(Gaugefile);
    } else {
        throw (std::runtime_error(rootLogger.fatal("Invalid specification for format ", format)));
    }
    check_unitarity<floatT,false,HaloDepth>(gauge);

    GaugeAction<floatT, false, HaloDepth> gaugeAction(gauge);
    floatT plaq = gaugeAction.plaquette();
    rootLogger.info("Plaquette = ", plaq);
    if ( (plaq > 1.0) || (plaq < 0.0) ) {
        throw std::runtime_error(rootLogger.fatal("Plaquette should not be negative or larger than 1."));
    }
}


int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(INFO);
        const size_t HaloDepth = 0;

        CheckParams param;
        param.nodeDim.set({1,1,1,1});
        CommunicationBase commBase(&argc, &argv);

        param.readfile(commBase, "../parameter/applications/checkConf.param", argc, argv);

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
    rootLogger.info("Gaugefile seems to be fine.");
    return 0;
}
