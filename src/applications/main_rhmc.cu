/* 
 * main_rhmc.cu                                                               
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

    gpuEvent_t mcu_timer_start, mcu_timer_stop;
    float cu_timer_time=0;
    double configtime=0;
    double totaltime=0;

    RhmcParameters param;

    param.readfile(commBase, "../parameter/tests/rhmcTest.param", argc, argv);

    const int HaloDepth = 2;
    const int Nmeas = 10; // Number of RHS in Condensate measurements

    std::string rand_file;
    std::string gauge_file;

    RationalCoeff rat;

    rat.readfile(commBase, param.rat_file(), 0, NULL);

    rat.check_rat(param);

    commBase.init(param.nodeDim(), param.gpuTopo());

    typedef float floatT;

    rootLogger.info() << "STARTING RHMC Update:";

    if (sizeof(floatT)==4)
        rootLogger.info() << "update done in single precision";
    else if(sizeof(floatT)==8)
        rootLogger.info() << "update done in double precision";
    else
        rootLogger.info() << "update done in unknown precision";

    initIndexer(4,param, commBase);

    Gaugefield<floatT, true, HaloDepth> gauge(commBase);

    grnd_state<true> d_rand;

    if (param.load_rand()) {
        grnd_state<false> h_rand;
        rand_file = param.rand_file() + std::to_string(param.config_no());
        rootLogger.info() << "With random numbers from file: " << rand_file;
        h_rand.read_from_file(rand_file, commBase);
        d_rand=h_rand;
    } else {
        rootLogger.info() << "With new random numbers generated from seed: " << param.seed();
        initialize_rng(param.seed(), d_rand);
    }

    if (param.load_conf() == 0) {
        rootLogger.info() << "Starting from unit configuration";
        gauge.one();
    } else if(param.load_conf() == 1) {
        rootLogger.info() << "Starting from random configuration";
        gauge.random(d_rand.state);
    } else if(param.load_conf() == 2) {
        gauge_file = param.gauge_file() + std::to_string(param.config_no());
        rootLogger.info() << "Starting from configuration: " << gauge_file;
        gauge.readconf_nersc(gauge_file);
    }
    gauge.updateAll();
    gauge.su3latunitarize();
    if(param.always_acc())
        rootLogger.warn() << "Skipping Metropolis step!";

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);

    rootLogger.info() << "constructed the HMC";

    HMC.init_ratapprox();

    int acc = 0;
    floatT acceptance = 0.0;
    PolyakovLoop<floatT, true, HaloDepth, R18> ploop(gauge);
    GaugeAction<floatT, true, HaloDepth, R18> gaugeaction(gauge);

    gpuEventCreate( &mcu_timer_start );
    gpuEventCreate( &mcu_timer_stop );

    for (int i = 1; i <= param.no_updates(); ++i) {

        gpuEventRecord( mcu_timer_start, 0 );

        acc += HMC.update(!param.always_acc());
        acceptance = floatT(acc)/floatT(i);
        rootLogger.info() << "current acceptance = " << acceptance;

        rootLogger.info() << "MEASUREMENT: " << param.config_no()+i;

        rootLogger.info() << "Polyakov Loop = " << ploop.getPolyakovLoop();
        rootLogger.info() << "Plaquette = " << gaugeaction.plaquette();
        rootLogger.info() << "Rectangle = " << gaugeaction.rectangle();
	
        measure_condensate<floatT, true, 2, 4, Nmeas>(commBase, param, true, gauge, d_rand);
	
        measure_condensate<floatT, true, 2, 4, Nmeas>(commBase, param, false, gauge, d_rand);

        gpuEventRecord( mcu_timer_stop, 0 );
        gpuEventSynchronize(mcu_timer_stop);
        gpuEventElapsedTime( &cu_timer_time, mcu_timer_start, mcu_timer_stop );
        configtime += cu_timer_time;
        totaltime += cu_timer_time;
        rootLogger.info() << "Time (TTRAJ) for trajectory without IO: " << static_cast<int>(cu_timer_time / 1000.) << " s | avg traj. time : "
            << static_cast<int>(totaltime/1000./(i)) << " s";
        
        //IO

        if (i % param.write_every()==0)
        {
            gauge_file = param.gauge_file() + std::to_string(param.config_no()+i);
            rand_file = param.rand_file() + std::to_string(param.config_no()+i);
            gauge.writeconf_nersc(gauge_file);
            grnd_state<false> h_rand;
            h_rand=d_rand;
            h_rand.write_to_file(rand_file, commBase);
        }
    }

    rootLogger.info() << "Run has ended! acceptance = " << acceptance;

    gpuEventDestroy(mcu_timer_stop);
    gpuEventDestroy(mcu_timer_start);

    return 0;
}