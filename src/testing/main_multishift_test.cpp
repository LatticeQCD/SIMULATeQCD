#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/hisq/hisqSmearing.h"


template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NShifts, bool onDevice>
void run_func(CommunicationBase &commBase, RhmcParameters &param, RationalCoeff &rat)
{
    // bool success = true;


    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    
    std::string gauge_file;

    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_naik);

    rootLogger.info("Initialize random state");

    grnd_state<onDevice> d_rand;
    initialize_rng(1536, d_rand);

    if (param.load_conf() == 0) {
        rootLogger.info("Starting from unit configuration");
        gauge.one();
    } else if(param.load_conf() == 1) {
        rootLogger.info("Starting from random configuration");
        gauge.random(d_rand.state);
    } else if(param.load_conf() == 2) {
        gauge_file = param.gauge_file() + std::to_string(param.confnumber());
        rootLogger.info("Starting from configuration: " ,  gauge_file);
        gauge.readconf_nersc(gauge_file);
    }

    gauge.updateAll();
    gauge.su3latunitarize();

    smearing.SmearAll();
    StopWatch<true> timer;


    rootLogger.info("Initialize spinors");
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorOut_single(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NShifts> spinorOut_shifts(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NShifts> spinor_ref(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NShifts> diff(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NShifts> pi_out(commBase);

    rootLogger.info("Randomize spinor");
    spinorIn.gauss(d_rand.state);
    
    SimpleArray<COMPLEX(double), 1> dot(0.0);

    rootLogger.info("||spinorIn||^2 = ", dot[0]);

    rootLogger.info("Init Dslash");
    // HisqDslash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NShifts> dslash(gauge_smeared, gauge_naik, 0.0);
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslash_m(gauge_smeared, gauge_naik, 0.0);

    AdvancedMultiShiftCG<floatT, NShifts> cgM;

    SimpleArray<floatT, NShifts> shifts(0.0);

    for (size_t i = 1; i < rat.r_inv_sf_den.get().size(); ++i){
        shifts[i] = rat.r_inv_sf_den[i] - rat.r_inv_sf_den[0];
    }
    shifts[0] = rat.r_inv_sf_den[0] + param.m_s()*param.m_s();
    auto tmp = shifts[1];
    shifts[1] = shifts[0];
    shifts[0] = tmp;
    rootLogger.info("Multishift inversion:");
    
    timer.reset();
    timer.start();
    cgM.invert(dslash_m, spinorOut_shifts, spinorIn, shifts, param.cgMax(), param.residue());
    
    timer.stop();
    rootLogger.info("Multishift inversion converged. Timing: ", timer);
    timer.reset();
    
    rootLogger.info("Multishift inversion with concurrent comms:");
    timer.start();
    cgM.invert_concurrent_comms(dslash_m, spinorOut_shifts, spinorIn, shifts, param.cgMax(), param.residue());
    timer.stop();
    rootLogger.info("Multishift inversion converged. Timing: ", timer);
    
}

int main(int argc, char **argv) {
    stdLogger.setVerbosity(DEBUG);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;

    param.readfile(commBase, "../parameter/tests/inverterTest.param", argc, argv);

    RationalCoeff rat;
    rat.readfile(commBase, param.rat_file());
    
    commBase.init(param.nodeDim(), param.gpuTopo());
    rootLogger.info("init Indexer");

    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin, param, commBase);

    rootLogger.info("-------------------------------------");
    rootLogger.info("Running on Device");
    rootLogger.info("-------------------------------------");

    run_func<double, Even, Odd, 14, true>(commBase, param, rat);

}