/* 
 * main_rhmc.cu                                                               
 *
 * This is the main to use RHMC to generate Nf=2+1 HISQ configurations. By default it will also measure
 * the chiral condensate.
 *  
 */

#include "../SIMULATeQCD.h"
#include "../modules/rhmc/rhmc.h"
#include "../modules/observables/PolyakovLoop.h"
#include <stdlib.h>
#include "../modules/dslash/condensate.h"

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters    param;

    param.readfile(commBase, "../parameter/tests/rhmcTest.param", argc, argv);
    if (not param.confnumber.isSet()){
        param.confnumber.set(0);
    }

    const int HaloDepth = 2;
    const size_t Nmeas  = 10; // Number of RHS in Condensate measurements

    std::string rand_file;
    std::string gauge_file;

    RationalCoeff rat;

    rat.readfile(commBase, param.rat_file());

    rat.check_rat(param);

    commBase.init(param.nodeDim(), param.gpuTopo());

    StopWatch<true> timer, totaltime;

    typedef float floatT; // Define the precision here

    rootLogger.info("STARTING RHMC Update:");

    if (sizeof(floatT)==4) {
        rootLogger.info("update done in single precision");
    } else if(sizeof(floatT)==8) {
        rootLogger.info("update done in double precision");
    } else {
        rootLogger.info("update done in unknown precision");
    }

    initIndexer(4,param, commBase);

    Gaugefield<floatT, true, HaloDepth> gauge(commBase);

    grnd_state<true> d_rand;

    if (param.load_rand()) {
        grnd_state<false> h_rand;
        rand_file = param.rand_file() + std::to_string(param.confnumber());
        rootLogger.info("With random numbers from file: " ,  rand_file);
        h_rand.read_from_file(rand_file, commBase);
        d_rand=h_rand;
    } else {
        rootLogger.info("With new random numbers generated from seed: " ,  param.seed());
        initialize_rng(param.seed(), d_rand);
    }

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
    if(param.always_acc()) {
        rootLogger.warn("Skipping Metropolis step!");
    }

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);

    rootLogger.info("constructed the HMC");

    HMC.init_ratapprox();

    int acc = 0;
    floatT acceptance = 0.0;
    PolyakovLoop<floatT, true, HaloDepth, R18> ploop(gauge);
    GaugeAction<floatT, true, HaloDepth, R18> gaugeaction(gauge);

    for (int i = 1; i <= param.no_updates(); ++i) {

        timer.start();

        acc += HMC.update(!param.always_acc());
        acceptance = floatT(acc)/floatT(i);
        rootLogger.info("current acceptance = ",  acceptance);

        rootLogger.info("MEASUREMENT: ",  param.confnumber()+i);

        rootLogger.info("Polyakov Loop = ",  ploop.getPolyakovLoop());
        rootLogger.info("Plaquette = ",  gaugeaction.plaquette());
        rootLogger.info("Rectangle = ",  gaugeaction.rectangle());
        	
        SimpleArray<double,Nmeas> chi_l = measure_condensate<floatT, true, 2, 4, Nmeas>(commBase, param, true,  gauge, d_rand);
        for (int j = 0; j < Nmeas; ++j) {
            rootLogger.info("CHI_UD = ", chi_l[j]);  
        }
        SimpleArray<double,Nmeas> chi_s = measure_condensate<floatT, true, 2, 4, Nmeas>(commBase, param, false, gauge, d_rand);
        for (int j = 0; j < Nmeas; ++j) {
            rootLogger.info("CHI_S = ", chi_s[j]);  
        }

        timer.stop();
        totaltime += timer;
        rootLogger.info("Time (TTRAJ) for trajectory without IO: ", timer, " | avg traj. time : " , totaltime/i);
        timer.reset();

        if (i % param.write_every()==0) {
            gauge_file = param.gauge_file() + std::to_string(param.confnumber()+i);
            rand_file = param.rand_file() + std::to_string(param.confnumber()+i);
            gauge.writeconf_nersc(gauge_file);
            grnd_state<false> h_rand;
            h_rand=d_rand;
            h_rand.write_to_file(rand_file, commBase);
        }
    }

    rootLogger.info("Run has ended! acceptance = " ,  acceptance);

    return 0;
}
