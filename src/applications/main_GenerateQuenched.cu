/*
 * main_GenerateQuenched.cu
 *
 * Luis Altenkort, 2 Apr 2019
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gauge_updates/PureGaugeUpdates.h"

#include <chrono>

#define PREC double
#define ONDEVICE true

struct generateQuenchedParameters : LatticeParameters {
    Parameter<std::string> output_dir;
    Parameter<int> nconfs;
    Parameter<int> nsweeps_ORperHB;
    Parameter<int> nsweeps_HBwithOR;
    Parameter<std::string> start;
    Parameter<int> nsweeps_thermal_HB_only;
    Parameter<int> nsweeps_thermal_HBwithOR;
    Parameter<int64_t> seed;
    Parameter<std::string> prev_conf;
    Parameter<std::string> prev_rand;

    // constructor
    generateQuenchedParameters() {
        addDefault(output_dir, "output_dir", std::string("."));
        addDefault(nconfs, "nconfs", 1000);
        addDefault(nsweeps_ORperHB, "nsweeps_ORperHB", 4);
        addDefault(nsweeps_HBwithOR, "nsweeps_HBwithOR", 500);
        addOptional(start, "start"); //valid options: one, fixed_random, all_random
        addOptional(nsweeps_thermal_HB_only, "nsweeps_thermal_HB_only"); //thermalization e.g. 500
        addOptional(nsweeps_thermal_HBwithOR, "nsweeps_thermal_HBwithOR"); //thermalization e.g. 4000
        addOptional(seed, "seed"); //default: time since unix epoch in milliseconds (see below)
        addOptional(prev_conf, "prev_conf");
        addOptional(prev_rand, "prev_rand");
    }
};

void set_seed( CommunicationBase &commBase, Parameter<int64_t> &seed ){
    if ( not seed.isSet() ) {
        int64_t root_seed = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
        commBase.root2all(root_seed);
        seed.set(root_seed);
        rootLogger.info("No seed was specified. Using time since epoch in milliseconds.");
    }
    rootLogger.info("Seed for random numbers is " ,  seed());
    return;
}


int main(int argc, char* argv[]) {

    const size_t HaloDepth = 1;

    ///Initialize Base
    typedef GIndexer<All,HaloDepth> GInd;
    stdLogger.setVerbosity(INFO);
    StopWatch                              timer;
    generateQuenchedParameters              lp;
    CommunicationBase                       commBase(&argc, &argv);
    lp.readfile(commBase, "../parameter/applications/GenerateQuenched.param", argc, argv);
    commBase.init(lp.nodeDim());
    initIndexer(HaloDepth,lp,commBase);
    Gaugefield<PREC,ONDEVICE,HaloDepth>     gauge(commBase);
    GaugeAction<PREC,ONDEVICE,HaloDepth>    gaugeAction(gauge);
    GaugeUpdate<PREC,ONDEVICE,HaloDepth>    gaugeUpdate(gauge);

    grnd_state<false> host_state;
    grnd_state<true> dev_state;
    ///Initialization Complete

    ///Start new stream or continue existing one?
    if ( lp.prev_conf.isSet()
        and not lp.nsweeps_thermal_HB_only.isSet()
        and not lp.nsweeps_thermal_HBwithOR.isSet()
        and lp.confnumber.isSet()) {
        rootLogger.info("Resuming previous run.");

        gauge.readconf_nersc(lp.prev_conf());
        gauge.updateAll();

        ///Initialize RNG for resuming
        if ( lp.prev_rand.isSet() ) {
            host_state.read_from_file(lp.prev_rand(), commBase);
        } else {
            rootLogger.warn("No prev_rand was specified!");
            set_seed(commBase, lp.seed);
            host_state.make_rng_state(lp.seed());
        }
        dev_state = host_state;

        lp.confnumber.set(lp.confnumber() + lp.nsweeps_HBwithOR());
        rootLogger.info("Next conf_number will be " ,  lp.confnumber());

    } else if ( not lp.prev_conf.isSet()
                and not lp.prev_rand.isSet()
                and lp.start.isSet()
                and lp.nsweeps_thermal_HB_only.isSet()
                and lp.nsweeps_thermal_HBwithOR.isSet()
                and not lp.confnumber.isSet()) {
        rootLogger.info("Starting new stream.");

        /// Initialize RNG for new stream
        set_seed(commBase, lp.seed);
        host_state.make_rng_state(lp.seed());
        dev_state = host_state;

        ///Initialize gaugefield
        if ( lp.start() == "fixed_random" ) {
            rootLogger.info("Starting with all U = some single arbitrary SU3");
            gSite first_site; //by default = 0 0 0 0
            GSU3<PREC> some_SU3;
            some_SU3.random(host_state.getElement(first_site));
            gauge.iterateWithConst(some_SU3);
        } else if ( lp.start() == "all_random"  ) {
            rootLogger.info("Starting with some random configuration");
            gauge.random(host_state.state);
        } else if ( lp.start() == "one" ) {
            rootLogger.info("Starting with all U = 1");
            gauge.one();
        } else {
            throw std::runtime_error(stdLogger.fatal("Error! Choose from 'start = {one, fixed_random, all_random}!"));
        }

        rootLogger.info("On stream " ,  lp.streamName());

        lp.confnumber.set(lp.nsweeps_HBwithOR());
        rootLogger.info("Start thermalization. Doing " ,  lp.nsweeps_thermal_HB_only() ,  " pure HB sweeps.");
        for (int i = 0; i < lp.nsweeps_thermal_HB_only(); ++i) {
            gaugeUpdate.updateHB(dev_state.state,lp.beta());
        }
        rootLogger.info("Now do " ,  lp.nsweeps_thermal_HBwithOR() ,  " HB sweeps with " ,  lp.nsweeps_ORperHB() , 
                            " OR sweeps per HB.");
        for (int i = 0; i < lp.nsweeps_thermal_HBwithOR(); ++i) {
            gaugeUpdate.updateHB(dev_state.state,lp.beta());
            for (int j = 0; j < lp.nsweeps_ORperHB(); j++) {
                gaugeUpdate.updateOR();
            }
        }
        rootLogger.info("Thermalization finished");
    } else {
        throw std::runtime_error(stdLogger.fatal("Error! Parameters unclear. To start a new stream, specify nsweeps_thermal_HB_only,"
                       "nsweeps_thermal_HBwithOR and start (one, fixed_random or all_random). To continue "
                       "existing stream, specify"
                       "(previous) conf_nr, prev_conf and (optionally) prev_rand. Do not specify unused"
                       "parameters."));
    }

    rootLogger.info("Generating up to " ,  lp.nconfs() ,  " confs with a separation of " , 
                        lp.nsweeps_HBwithOR() ,  " HBOR sweeps (OR/HB = " ,  lp.nsweeps_ORperHB() ,  ") ...");
    for (int i = 0; i < lp.nconfs(); i++ ){
        rootLogger.info("======================================================================");
        rootLogger.info("Start sweeping...");
        ///do separation sweeps
        timer.start();
        for (int i = 0; i < lp.nsweeps_HBwithOR(); ++i) {
            gaugeUpdate.updateHB(dev_state.state, lp.beta());
            for (int j = 0; j < lp.nsweeps_ORperHB(); j++) {
                gaugeUpdate.updateOR();
            }
        }
        timer.stop();
        rootLogger.info("It took " ,  timer.seconds() ,  " seconds to do " ,  lp.nsweeps_HBwithOR() ,  " HBOR "
                             "sweeps.");
        timer.reset();

        rootLogger.info("Plaquette = " ,  gaugeAction.plaquette());

        std::string conf_path = lp.output_dir()+"/conf"+lp.fileExt();
        std::string rand_path = lp.output_dir()+"/rand"+lp.fileExt();

        rootLogger.info("Writing conf to disk...");
        timer.start();
        gauge.writeconf_nersc(conf_path, 2, 2);
        host_state = dev_state;
        host_state.write_to_file(rand_path, commBase);
        timer.stop();

        rootLogger.info("Writing to disk took " ,  timer.seconds() ,  " seconds");
        timer.reset();
        lp.confnumber.set(lp.confnumber() + lp.nsweeps_HBwithOR());
    }

    return 0;
}


