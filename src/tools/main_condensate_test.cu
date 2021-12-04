#include "../define.h"
#include "../gauge/gaugefield.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../spinor/spinorfield.h"
#include "../modules/dslash/dslash.h"
#include "../modules/HISQ/hisqSmearing.h"

#include <stdlib.h> 
#include <string>

#include <time.h> 

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, size_t NStacks>
void measure_condensate(CommunicationBase &commBase, RhmcParameters param, std::string conf) {

    typedef GIndexer<All,HaloDepth> GInd;

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge(commBase);
    
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    int seed;

    srand(time(NULL));

    seed = rand();

    rootLogger.info("seed from rand and time = " ,  seed);

    h_rand.make_rng_state(seed);

    d_rand = h_rand;
    rootLogger.info("Reading configuration");
    gauge.readconf_nersc(conf);
    gauge.updateAll();

    rootLogger.info("Smearing gauge fields");

    Gaugefield<floatT, onDevice, HaloDepth, U3R14> smeared_X(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> smeared_W(commBase);
    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> smearing(gauge, smeared_W, smeared_X);
    smearing.SmearAll();
    
    double mass = param.m_ud();
    
    ConjugateGradient<floatT, NStacks> cg;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e(smeared_W, smeared_X, 0.0);
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e_inv(smeared_W, smeared_X, mass);
    HisqDSlash<floatT, onDevice, Odd, HaloDepth, HaloDepthSpin, NStacks> dslash_o(smeared_W, smeared_X, 0.0);

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> eta_e(commBase);
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> eta_o(commBase);
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> x_e(commBase);
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> w_e(commBase);  
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> w_o(commBase);

    eta_o.gauss(d_rand.state);
    eta_e.gauss(d_rand.state);

    SimpleArray<double, NStacks> dot_e(0.0);
    SimpleArray<double, NStacks> dot_o(0.0);

    dslash_o.Dslash(x_e, eta_o);
    x_e = eta_e * floatT(mass) - x_e;

    rootLogger.info("Starting inversion");

    cg.invert_new(dslash_e_inv, w_e, x_e, param.cgMax(), param.residue());

    dslash_e.Dslash(w_o, w_e);
    w_o = eta_o - w_o;


    dot_e = eta_e.realdotProductStacked(w_e);
    dot_o = eta_o.realdotProductStacked(w_o);

    dot_o = (1.0/mass)*dot_o;

    for (size_t i = 0; i < NStacks; ++i) {
         rootLogger.info("CHIRAL CONDENSATE = " ,   (dot_o[i]+dot_e[i])/floatT(GInd::getLatData().globvol4));
    }
};


int main(int argc, char *argv[])
{
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    RhmcParameters param;

    param.readfile(commBase, "../parameter/run.param", 0, NULL);

    std::string conf;

    if (argc != 2) {
        throw std::runtime_error(stdLogger.fatal("Wrong number of arguments!");
    } else {
        conf = argv[1];
    }
    
    commBase.init(param.nodeDim());

    initIndexer(4,param, commBase);

    const size_t numVec=10;

    // rootLogger.info("START MEASURING CHIRAL CONDENSATE IN DOUBLE");
    // measure_condensate<double, true, 2, 4, numVec>(commBase, param, conf);

    rootLogger.info("START MEASURING CHIRAL CONDENSATE IN SINGLE");
    measure_condensate<float, true, 2, 4, numVec>(commBase, param, conf);

    return 0;
}

