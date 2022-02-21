/* 
 * main_HisqSmearing_Create_Multi.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/rhmc/rhmcParameters.h"

#define PREC double
#define USE_GPU true

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    
    CommunicationBase commBase(&argc, &argv);

    RhmcParameters param;
    param.readfile(commBase,"../parameter/tests/run.param", argc, argv);
    
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 2;

    rootLogger.info("Initialize Lattice");
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);
    grnd_state<true> d_rand;
  
    initialize_rng(1337,d_rand);
    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC, true, HaloDepth> gauge_in(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_Lv2(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_naik(commBase);

    HisqSmearing<PREC, USE_GPU, HaloDepth,R18,R18,R18,R18> smearing(gauge_in, gauge_Lv2, gauge_naik);

    gauge_in.random(d_rand.state);
    gauge_in.updateAll();
    StopWatch<true> timer;

    
    timer.start();    
    smearing.SmearAll();
    timer.stop();
    rootLogger.info("Time for full smearing: " ,  timer);
    std::string filename_out;
    if (param.nodeDim[1] == 1) {
        filename_out = "../test_conf/pgpu_naik_smearing_single.nersc";
    }
    if (param.nodeDim[1] == 2) {
        filename_out = "../test_conf/pgpu_naik_smearing_multi.nersc";
    }
    if (param.nodeDim[1] + param.nodeDim[2] == 4 ) {
        filename_out = "../test_conf/pgpu_naik_smearing_multi_x.nersc";
    }
    
    gauge_naik.writeconf_nersc(filename_out);

    return 0;
}

