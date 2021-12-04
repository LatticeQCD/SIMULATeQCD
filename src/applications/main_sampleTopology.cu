/* 
 * main_sampleTopology.cu                                                               
 * 
 * L. Altenkort
 * 
 * Executable to generate quenched configurations with non-zero topological charge.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gauge_updates/PureGaugeUpdates.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include "../modules/observables/Topology.h"
#include <cstdio>
#include <chrono>

#define PREC double
#define ONDEVICE true
#define USE_GPU true

template<class floatT>
struct sampleTopologyParameters : LatticeParameters {

    //! GenerateQuenched part

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
    Parameter<int> nsweeps_btwn_topology_meas;

    Parameter<bool> prev_conf_has_nonzero_Q;

    //! gradientFlow part
    //! ---------------------------------basic options you probably care about------------------------------------------
    Parameter<std::string> measurements_dir; //! where the output gets stored
    Parameter<std::string> force; //! wilson or zeuthen flow
    Parameter<std::string> RK_method; //! RK_method = {fixed_stepsize, adaptive_stepsize, adaptive_stepsize_allgpu}
    Parameter<floatT> start_step_size;
    Parameter<floatT> accuracy; //! only used for adaptive stepsize. difference between 2nd and 3rd order RK
    DynamicParameter<floatT> necessary_flow_times; //! these flow times will not be skipped over in the integration
    Parameter<floatT, 2> measurement_intervall; //! measurement_intervall[0]: start, [1]: stop

    //! ---------------------------------which observables should be measured on the flowed configuration?--------------
    Parameter<bool> plaquette;
    Parameter<bool> topCharge_imp;
    Parameter<bool> topChargeTimeSlices_imp;

      //! ---------------------------------advanced options---------------------------------------------------------------

    Parameter<bool> use_unit_conf; //! for testing (or benchmarking purposes using fixed stepsize)
    Parameter<bool> save_conf;
    //! ignore start_step_size and integrate to the necessary_flow_times without steps in between.
    //! only useful when using RK_method fixed_stepsize
    Parameter<bool> ignore_fixed_startstepsize;

    Parameter<bool> print_all_flowtimes;

    // constructor
    sampleTopologyParameters() {

        //! GenerateQuenched
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
        addDefault(nsweeps_btwn_topology_meas, "nsweeps_btwn_topology_meas", 10);

        //! gradientFlow part
        addDefault(force, "force", std::string("zeuthen"));

        add(start_step_size, "start_step_size");

        addDefault(RK_method, "RK_method", std::string("adaptive_stepsize"));
        addDefault(accuracy, "accuracy", floatT(1e-5));

        add(measurements_dir, "measurements_dir");

        addOptional(necessary_flow_times, "necessary_flow_times");
        addDefault(ignore_fixed_startstepsize, "ignore_start_step_size", false);

        addDefault(save_conf, "save_configurations", false);

        addDefault(use_unit_conf, "use_unit_conf", false);

        add(measurement_intervall, "measurement_intervall");

        addDefault(plaquette, "plaquette", true);
        addDefault(topCharge_imp, "topCharge_imp", true);
        addDefault(topChargeTimeSlices_imp, "topChargeTimeSlices_imp", true);
        addDefault(prev_conf_has_nonzero_Q, "prev_conf_has_nonzero_Q", false);
        addDefault(print_all_flowtimes, "print_all_flowtimes", false);
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
}

template<class floatT, size_t HaloDepth, typename gradFlowClass>
void run_flow(gradFlowClass &gradFlow, Gaugefield<PREC, USE_GPU, HaloDepth> &gauge, sampleTopologyParameters<PREC> &lp, floatT& topchar_out, bool writeFiles) {

    rootLogger.info("Applying gradient flow...");

    //! -------------------------------prepare file output--------------------------------------------------------------
    std::stringstream prefix, datName, datNameConf, datNameTopChSlices, datNameTopChSlices_imp;

    // fill stream with 0's
    datName.fill('0');
    // get the data file name
    if ( lp.RK_method() == "adaptive_stepsize" || lp.RK_method() == "adaptive_stepsize_allgpu" )
        prefix << lp.force() << "Flow_acc" << std::fixed << std::setprecision(6)
               << lp.accuracy() << "_sts" << std::fixed << std::setprecision(6) << lp.start_step_size();
    else {
        prefix << lp.force() << "Flow";
        if (not lp.ignore_fixed_startstepsize()) {
            prefix << "_sts" << std::fixed << std::setprecision(6) << lp.start_step_size();
        }
    }
    datName << lp.measurements_dir() << prefix.str() << lp.fileExt();
    datNameConf << lp.measurements_dir()<< "conf_" << prefix.str() << lp.fileExt();
    datNameTopChSlices << lp.measurements_dir() << prefix.str() << "_TopChTimeSlices" << lp.fileExt();
    datNameTopChSlices_imp << lp.measurements_dir() << prefix.str() << "_TopChTimeSlicesImp" << lp.fileExt();

    FileWriter file(gauge.getComm(), lp);

    if (writeFiles){
        file.createFile(datName.str());
    }

    //! write header
    LineFormatter header = file.header();
    header << "Flow time ";
    if (lp.plaquette()) header << "Plaquette ";
    if (lp.topCharge_imp()) header << "Impr. top. Charge ";
    header.endLine();

    FileWriter fileTopChSl_imp(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices_imp() and writeFiles) {
        fileTopChSl_imp.createFile(datNameTopChSlices_imp.str());
        LineFormatter headerThSl_imp = fileTopChSl_imp.header();
        headerThSl_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp.endLine();
    }

    gauge.updateAll();

    //! -------------------------------set up observable measurement classes--------------------------------------------
    GaugeAction<PREC, USE_GPU, HaloDepth> gAction(gauge);
    Topology<PREC, USE_GPU, HaloDepth> topology(gauge);

    //! -------------------------------variables for the observables----------------------------------------------------
    PREC plaq, topChar;
    std::vector<PREC> resultClSl;
    std::vector<PREC> resultThSl;
    std::vector<PREC> resultThSl_imp;

    //! -------------------------------flow the field until max flow time-----------------------------------------------
    std::stringstream logStream;
    MicroTimer timer;
    timer.start();
    PREC flow_time = 0.;
    bool continueFlow = true;
    while (continueFlow) {
        continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached


        //! -------------------------------prepare log output-----------------------------------------------------------
        logStream.str("");
        logStream << std::fixed << std::setprecision(7) << "   t = " << flow_time << ": ";


        LineFormatter newTag = file.tag("");
        if (writeFiles) newTag << flow_time;


        //! -------------------------------calculate observables on flowed field----------------------------------------
        if (lp.save_conf() && gradFlow.checkIfnecessaryTime()){
            gauge.writeconf_nersc( datNameConf.str() + "_FT" + std::to_string(flow_time));
        }
        if (lp.plaquette()) {
            plaq = gAction.plaquette();
            logStream << std::fixed << std::setprecision(6) << "   Plaquette = " << plaq;
            if (writeFiles) newTag << plaq;
        }
        if (lp.topChargeTimeSlices_imp() && gradFlow.checkIfnecessaryTime() && writeFiles) {
            LineFormatter newTagTh = fileTopChSl_imp.tag("");
            topology.template topChargeTimeSlices<true>(resultThSl_imp);
            if (writeFiles) newTagTh << flow_time;
            if (writeFiles) for (auto &elem : resultThSl_imp) {
                newTagTh << elem;
            }
            topology.dontRecomputeField();
        }
        if (lp.topCharge_imp() && (gradFlow.checkIfnecessaryTime() && writeFiles or gradFlow.checkIfEndTime()) ) {
            topChar = topology.template topCharge<true>();
            topchar_out = topChar;
            logStream << std::fixed << std::setprecision(6) << "   topCharge_imp = " << topChar;
            if (writeFiles) newTag << topChar;
            topology.recomputeField();
        }

        if (gradFlow.checkIfEndTime() or (lp.print_all_flowtimes()) ){
            rootLogger.info(logStream.str());
        }

        flow_time += gradFlow.updateFlow(); //! integrate flow equation up to next flow time
        gauge.updateAll();

        gAction.recomputeField();
        topology.recomputeField();
    }
    timer.stop();
}


template<class floatT, bool onDevice, const size_t HaloDepth, RungeKuttaMethod input_RK_method, template<class, const size_t, RungeKuttaMethod> class gradFlowClass >
void init(CommunicationBase &commBase,
               sampleTopologyParameters<floatT> &lp) {

    initIndexer(HaloDepth,lp,commBase);
    MicroTimer                              timer;

    Gaugefield<PREC,ONDEVICE,HaloDepth>     gauge(commBase);
    Gaugefield<PREC,false,HaloDepth>     gauge_backup(commBase);
    Gaugefield<PREC,false,HaloDepth>     gauge_nonflowed(commBase);
    GaugeAction<PREC,ONDEVICE,HaloDepth>    gaugeAction(gauge);
    GaugeUpdate<PREC,ONDEVICE,HaloDepth>    gaugeUpdate(gauge);

    grnd_state<false> host_state;
    grnd_state<true> dev_state;

    rootLogger.info("===================================================================");

    ///Start new stream or continue existing one?
    //! for thermalization we don't care about topology for now (maybe add this later)
    if ( lp.prev_conf.isSet()
         and not lp.nsweeps_thermal_HB_only.isSet()
         and not lp.nsweeps_thermal_HBwithOR.isSet()
         and lp.confnumber.isSet()){
        rootLogger.info("Resuming previous run.");

        gauge.readconf_nersc(lp.prev_conf());
        gauge.updateAll();

        ///Initialize RNG for resuming
        if ( lp.prev_rand.isSet() ){
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
                and not lp.confnumber.isSet()){
        rootLogger.info("Starting new stream.");

        /// Initialize RNG for new stream
        set_seed(commBase, lp.seed);
        host_state.make_rng_state(lp.seed());
        dev_state = host_state;

        ///Initialize gaugefield
        if ( lp.start() == "fixed_random" ){
            rootLogger.info("Starting with all U = some single arbitrary SU3");
            gSite first_site; //by default = 0 0 0 0
            GSU3<PREC> some_SU3;
            some_SU3.random(host_state.getElement(first_site));
            gauge.iterateWithConst(some_SU3);
        } else if ( lp.start() == "all_random"  ){
            rootLogger.info("Starting with some random configuration");
            gauge.random(host_state.state);
        } else if ( lp.start() == "one" ){
            rootLogger.info("Starting with all U = 1");
            gauge.one();
        } else {
            throw PGCError("Error! Choose from 'start = {one, fixed_random, all_random}!");
        }

        rootLogger.info("On stream " ,  lp.streamName());

        lp.confnumber.set(lp.nsweeps_HBwithOR());
        rootLogger.info("Start thermalization. Doing " ,  lp.nsweeps_thermal_HB_only() ,  " pure HB sweeps.");
        for (int i = 0; i < lp.nsweeps_thermal_HB_only(); ++i){
            gaugeUpdate.updateHB(dev_state.state,lp.beta());
        }
        rootLogger.info("Now do " ,  lp.nsweeps_thermal_HBwithOR() ,  " HB sweeps with " ,  lp.nsweeps_ORperHB() , 
                          " OR sweeps per HB.");
        for (int i = 0; i < lp.nsweeps_thermal_HBwithOR(); ++i){
            gaugeUpdate.updateHB(dev_state.state,lp.beta());
            for (int j = 0; j < lp.nsweeps_ORperHB(); j++) {
                gaugeUpdate.updateOR();
            }
        }
        rootLogger.info("Thermalization finished");
    } else {
        throw PGCError("Error! Parameters unclear. To start a new stream, specify nsweeps_thermal_HB_only,"
                       "nsweeps_thermal_HBwithOR and start (one, fixed_random or all_random). To continue "
                       "existing stream, specify"
                       "(previous) conf_nr, prev_conf and (optionally) prev_rand. Do not specify unused"
                       "parameters.");
    }



    rootLogger.info("Generating up to " ,  lp.nconfs() ,  " confs with a separation of " , 
                      lp.nsweeps_HBwithOR() ,  " HBOR sweeps (OR/HB = " ,  lp.nsweeps_ORperHB() ,  ") ...");


    int n_topo_meas_btwn_saved_confs = lp.nsweeps_HBwithOR()/lp.nsweeps_btwn_topology_meas();
    rootLogger.info("Checking top. charge " ,  n_topo_meas_btwn_saved_confs ,  "times between saved confs.");

    {
        gradFlowClass<floatT, HaloDepth, input_RK_method> gradFlow(gauge, lp.start_step_size(),
                                                                   lp.measurement_intervall()[0],
                                                                   lp.measurement_intervall()[1],
                                                                   lp.necessary_flow_times.get(), lp.accuracy());
        floatT topchar_tmp;
        run_flow<floatT, HaloDepth, gradFlowClass<floatT, HaloDepth, input_RK_method>>(gradFlow, gauge, lp, topchar_tmp,
                                                                                       (false));
        rootLogger.info("Top. charge of start conf: Q=" ,  topchar_tmp);
    }

    int n_accumulated_sweeps = 0;

    for (int i = 0; i < lp.nconfs(); i++ ){
        rootLogger.info("======================================================================");
        rootLogger.info("Start sweeping...");
        ///do separation sweeps
        timer.start();

        int k = 0;
        while ( k < n_topo_meas_btwn_saved_confs )  {
            gauge_backup = gauge;
            int m = 0;
            while ( m < lp.nsweeps_btwn_topology_meas()){
                gaugeUpdate.updateHB(dev_state.state, lp.beta());
                for (int j = 0; j < lp.nsweeps_ORperHB(); j++) { gaugeUpdate.updateOR();}
                n_accumulated_sweeps++;
                m++;
            }
            //! flow and check topo
            rootLogger.info("Accumulated sweeps = " ,  n_accumulated_sweeps);
            gauge_nonflowed = gauge;
            //! flow and measure topology
            gradFlowClass<floatT, HaloDepth, input_RK_method> gradFlow(gauge, lp.start_step_size(),
                                                                       lp.measurement_intervall()[0],
                                                                       lp.measurement_intervall()[1],
                                                                       lp.necessary_flow_times.get(), lp.accuracy());
            floatT topchar;
            run_flow<floatT, HaloDepth, gradFlowClass<floatT, HaloDepth, input_RK_method>>(gradFlow, gauge, lp, topchar, (k+1 == n_topo_meas_btwn_saved_confs)  );
            rootLogger.info("Done top. charge meas. " ,  k+1 ,  "/" ,  n_topo_meas_btwn_saved_confs ,  " Q=" ,  topchar);

            if ( topchar >= 0.5 or topchar <= -0.5 ){
                rootLogger.info("Continue sweeping");
                gauge = gauge_nonflowed; //! continue with the current conf but undo the flow first
            } else {
                if (i == 0 && not lp.prev_conf_has_nonzero_Q() ){
                    rootLogger.info("Top. charge not non-zero but we're on the first conf so we keep the update sweeps!");
                    gauge = gauge_nonflowed;    //! if we're still on the first conf we can accumulate the updates
                } else {
                    rootLogger.warn("Top. charge not non-zero! Discarding last update sweeps.");
                    n_accumulated_sweeps-=lp.nsweeps_btwn_topology_meas();
                    gauge = gauge_backup; //! continue with the conf from before the updates
                }
                //! reset counts
                k--;
            }
            k++;
        }

        timer.stop();

        timer.reset();

        rootLogger.info("Plaquette = " ,  gaugeAction.plaquette());

        std::string conf_path = lp.output_dir()+"/conf"+lp.fileExt();
        std::string rand_path = lp.output_dir()+"/rand"+lp.fileExt();

        rootLogger.info("Writing conf to disk...");
        timer.start();
        gauge.writeconf_nersc(conf_path, 2, 2);
        host_state = dev_state;
        timer.stop();

        rootLogger.info("Writing to disk took " ,  timer.ms()/1000 ,  " seconds");
        timer.reset();
        lp.confnumber.set(lp.confnumber() + lp.nsweeps_HBwithOR());
    }
}


int main(int argc, char* argv[]) {

    ///Initialize Base
    stdLogger.setVerbosity(INFO);

    sampleTopologyParameters<PREC> lp;
    CommunicationBase commBase(&argc, &argv);
    lp.readfile(commBase, "../parameter/applications/GenerateQuenched.param", argc, argv);
    commBase.init(lp.nodeDim());

    if (  lp.nsweeps_HBwithOR() % lp.nsweeps_btwn_topology_meas() != 0){
        throw PGCError("nsweeps_HBwithOR has to be a multiple of nsweeps_btwn_topology_meas");
    }

    ///Convert input strings to enum for switching
    Force input_force = Force_map[lp.force()];
    RungeKuttaMethod input_RK_method = RK_map[lp.RK_method()];

    if (input_RK_method == fixed_stepsize && lp.ignore_fixed_startstepsize() && lp.necessary_flow_times.isSet()) {
        rootLogger.info("Ignoring fixed start_step_size. "
                             "Stepsizes are dynamically deduced from necessary_flow_times.");
        lp.start_step_size.set(lp.measurement_intervall()[1]);
    }

    ///Set HaloDepth. The ifdefs can reduce compile time (only define what you need in CMakeLists).
    ///Wilson flow with topological charge (correlator) needs HaloDepth=2, without 1.
    ///Zeuthen flow always needs 3.
    switch (input_force) {
#ifdef WILSON_FLOW
        case wilson: {
            const size_t HaloDepth = 2;
            switch (input_RK_method) {
#ifdef FIXED_STEPSIZE
                case fixed_stepsize:
                    init<PREC, USE_GPU, HaloDepth, fixed_stepsize, wilsonFlow>(commBase, lp);
                    break;
#endif
#ifdef ADAPTIVE_STEPSIZE
                case adaptive_stepsize:
                    init<PREC, USE_GPU, HaloDepth, adaptive_stepsize, wilsonFlow>(commBase, lp);
                    break;
                case adaptive_stepsize_allgpu:
                    init<PREC, USE_GPU, HaloDepth, adaptive_stepsize_allgpu, wilsonFlow>(commBase, lp);
                    break;
#endif
                default:
                    throw PGCError("Invalid RK_method. Did you set the compile definitions accordingly?");
            }
            break;
        }
#endif
#ifdef ZEUTHEN_FLOW
        case zeuthen: {
                const size_t HaloDepth = 3;
                switch (input_RK_method) {
#ifdef FIXED_STEPSIZE
                    case fixed_stepsize:
                        init<PREC, USE_GPU, HaloDepth, fixed_stepsize, zeuthenFlow>(commBase, lp);
                        break;
#endif
#ifdef ADAPTIVE_STEPSIZE
                    case adaptive_stepsize:
                        init<PREC, USE_GPU, HaloDepth, adaptive_stepsize, zeuthenFlow>(commBase, lp);
                        break;
                    case adaptive_stepsize_allgpu:
                        init<PREC, USE_GPU, HaloDepth, adaptive_stepsize_allgpu, zeuthenFlow>(commBase, lp);
                        break;
#endif
                    default:
                        throw PGCError("Invalid RK_method. Did you set the compile definitions accordingly?");
                }
                break;
            }
#endif
        default:
            throw PGCError("Invalid force. Did you set the compile definitions accordingly?");
    }

    return 0;
}


