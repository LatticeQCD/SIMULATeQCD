#include "../define.h"
#include "../spinor/spinorfield.h"
#include "../gauge/gaugefield.h"
#include "../base/staticArray.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

//Run function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, size_t Multistack, bool onDevice>
void run_func(CommunicationBase &commBase, RhmcParameters &param, RationalCoeff &rat)
{
    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);


    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayoutRHS, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);
    // Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared_stored(commBase);
    // Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik_stored(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    
    rootLogger.info() << "Read configuration";
    // gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");

    rootLogger.info() << "Initialize random state";
    grnd_state<onDevice> d_rand;
    initialize_rng(13333, d_rand);

    // gauge.random(d_rand.state);
    gauge.one();
    gauge.updateAll();

    smearing.SmearAll();

    rootLogger.info() << "Testing multishift inverter:";
    // r_2f test
    rootLogger.info() << "Starting r_2f test";
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_ud(gauge_smeared, gauge_Naik, 0.0);
    // HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_s(gauge_smeared, gauge_Naik, param.m_s());

    SimpleArray<floatT, Multistack> shifts(0.0);

    for (size_t i = 1; i <rat.r_inv_1f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_1f_den[i]-rat.r_inv_1f_den[0];
    }
    shifts[0] = rat.r_inv_1f_den[0] + param.m_s()*param.m_s();

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorInMulti(commBase);
    spinorInMulti.gauss(d_rand.state);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, Multistack> spinorOutMulti(commBase);

    AdvancedMultiShiftCG<floatT, Multistack> cgM;
    // MultiShiftCG<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, Multistack> cgM;


    // make phi:
    gpuProfilerStart();
    // timer.start();
    gpuEventRecord(start);
    cgM.invert(dslashMulti_ud, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float milliseconds = 0;
    gpuEventElapsedTime(&milliseconds, start, stop);
    // timer.stop();
    gpuProfilerStop();
    rootLogger.info() << "Time for inversion with multishift CG: " << milliseconds;

    // phi = floatT(rat.r_2f_const()) * spinorInMulti;

    // for (size_t i = 0; i < rat.r_2f_den.get().size(); ++i)
    // {
    //     spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
    //     phi = phi + floatT(rat.r_2f_num[i])*spinortmp;
    // }
}



int main(int argc, char **argv) {
try {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;
    param.readfile(commBase, "../parameter/profiling/InverterProfile.param", argc, argv);
    param.cgMax.set(4000);

    RationalCoeff rat;
    rat.readfile(commBase, param.rat_file());

    commBase.init(param.nodeDim());

    // const int HaloDepth = 2;
    // initIndexer(HaloDepth,param, commBase);
    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin, param, commBase);

    rootLogger.info() << "-------------------------------------";
    rootLogger.info() << "Running on Device";
    rootLogger.info() << "-------------------------------------";

    // rootLogger.info() << "Testing All - All";
    // rootLogger.info() << "------------------";
    // run_func<float, All, All, 1, 14, true>(commBase, param, rat);

    rootLogger.info() << "------------------";
    rootLogger.info() << "Testing Even - Odd";
    rootLogger.info() << "------------------";
    run_func<float, Even, Odd, 1, 14, true>(commBase, param, rat);
    // rootLogger.info() << "------------------";
    // rootLogger.info() << "Testing Odd - Even";
    // rootLogger.info() << "------------------";
    // run_func<double, Odd, Even, 1, 14, true>(commBase, param, rat);

    /// Apparently the host has trouble to store a configuration.
    //    rootLogger.info() << "-------------------------------------";
    //    rootLogger.info() << "Running on Host";
    //    rootLogger.info() << "-------------------------------------";
    //    rootLogger.info() << "Testing All - All";
    //    rootLogger.info() << "------------------";
    //    run_func<double, All, All, 1, 14, false>(commBase, param, rat);
    //    rootLogger.info() << "------------------";
    //    rootLogger.info() << "Testing Even - Odd";
    //    rootLogger.info() << "------------------";
    //    run_func<double, Even, Odd, 1, 14, false>(commBase, param, rat);
    //    rootLogger.info() << "------------------";
    //    rootLogger.info() << "Testing Odd - Even";
    //    rootLogger.info() << "------------------";
    //    run_func<double, Odd, Even, 1, 14,false>(commBase, param, rat);
}
catch (const std::runtime_error &error) {
    rootLogger.error() << "There has been a runtime error!";
    return -1;
}
}

// template<Layout LatLayout, size_t HaloDepth>
// size_t getGlobalIndex(LatticeDimensions coord) {
//     typedef GIndexer<LatLayout, HaloDepth> GInd;

//     LatticeData lat = GInd::getLatData();
//     LatticeDimensions globCoord = lat.globalPos(coord);

//     return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
//         globCoord[3] * lat.globLX * lat.globLY * lat.globLZ;
// }
