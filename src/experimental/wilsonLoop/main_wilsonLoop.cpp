/* 
 * main_wilson_lines_correlator_stacked.cu                                                               
 * 
 * Rasmus Larsen, 25 Feb 2021
 * 
 */

#include "../simulateqcd.h"
#include "../modules/observables/wilsonLineCorrelatorMultiGPU.h"
#include "../modules/gradientFlow/gradientFlow.h"

//#include <iostream>
//using namespace std;

#define PREC double
#define STACKS 64 


template<class floatT>
struct WLParam : LatticeParameters {
    Parameter<floatT>      gtolerance;
    Parameter<int,1>       maxgfsteps;
    Parameter<int,1>       numunit;
    Parameter<int> load_conf;
    Parameter <std::string> gauge_file;
    Parameter <std::string> directory;
    Parameter <std::string> file_type;

    Parameter<floatT>  wilson_step;
    Parameter<floatT> wilson_start;
    Parameter<floatT> wilson_stop;
    Parameter<int,1> use_wilson;

    Parameter<int,1>       cutRadius;
    Parameter<int,1>       useInfoFile;


    WLParam() {
        addDefault (gtolerance,"gtolerance",1e-6);
        addDefault (maxgfsteps,"maxgfsteps",9000);
        addDefault (numunit   ,"numunit"   ,20);
        addDefault(load_conf, "load_conf", 0);
        addOptional(gauge_file, "gauge_file");
        add(directory, "directory");
        add(file_type, "file_type");

	addDefault (use_wilson,"use_wilson",0);
	addDefault (wilson_step,"wilson_step",0.0);
	addDefault (wilson_start,"wilson_start",0.0);
	addDefault (wilson_stop,"wilson_stop",0.0);
        addDefault(cutRadius, "cutRadius", 100000);
        addDefault(useInfoFile, "useInfoFile", 1);
    }
};

template<class floatT>
struct milcInfo : ParameterList {
    Parameter<floatT>  ssplaq;
    Parameter<floatT>  stplaq;
    Parameter<floatT>  linktr;

    milcInfo() {
        add(ssplaq, "gauge.ssplaq");
        add(stplaq, "gauge.stplaq");
        add(linktr, "gauge.nersc_linktr");
    }
};


int main(int argc, char *argv[]) {

    /// Controls whether DEBUG statements are shown as it runs; could also set to INFO, which is less verbose.
    stdLogger.setVerbosity(INFO);

    /// Initialize parameter class.
    WLParam<PREC> param;

    /// Initialize the CommunicationBase.
    CommunicationBase commBase(&argc, &argv);

    param.readfile(commBase, "../parameter/test.param", argc, argv);


    commBase.init(param.nodeDim());

//    cout << param.nodeDim[0] << " param 0 " <<  param.nodeDim[1] << " param 1 " << param.nodeDim[2] << " param 2 " << param.nodeDim[3] << " param 3 " <<endl; 

    /// Set the HaloDepth.
    const size_t HaloDepth = 2;

    rootLogger.info("Initialize Lattice");

    /// Initialize the Lattice class.
    initIndexer(HaloDepth,param,commBase);

    /// Initialize the Gaugefield.
    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);
    Gaugefield<PREC,true,HaloDepth> gaugeOriginal(commBase);


    /// Initialize gaugefield with unit-matrices.
    gauge.one();

    std::string gauge_file;

    // load gauge file, 0 start from 1, 1 and 2 load file, 2 will also gauge fix
    if (param.load_conf() == 0)
    {
        rootLogger.info("Starting from unit configuration");
        gauge.one();
    }
    else if(param.load_conf() == 2 || param.load_conf() == 1)
    {
        std::string file_path = param.directory();
        file_path.append(param.gauge_file()); 
        rootLogger.info("Starting from configuration: ", file_path);
//	rootLogger.info() << param.gauge_file() << endl;
        if(param.file_type() == "nersc"){
            gauge.readconf_nersc(file_path);
        }
        else if(param.file_type() == "milc"){
            gauge.readconf_milc(file_path);

            gauge.updateAll();         
            GaugeAction<PREC,true,HaloDepth> enDensity(gauge);
            PREC SpatialPlaq  = enDensity.plaquetteSS();
            PREC TemporalPlaq = enDensity.plaquette()*2.0-SpatialPlaq;
            rootLogger.info( "plaquetteST: "   , TemporalPlaq);
            rootLogger.info( "plaquetteSS: " , SpatialPlaq);


            if(param.useInfoFile()){
                std::string info_path = file_path;
                info_path.append(".info");
                milcInfo<PREC> paramMilc;
                paramMilc.readfile(commBase,info_path);
                rootLogger.info( "plaquette SS info file: " ,  (paramMilc.ssplaq())/3.0  );
                rootLogger.info( "plaquette ST info file: " ,  (paramMilc.stplaq())/3.0  );
                rootLogger.info( "linktr info file: " , paramMilc.linktr()  );
                if(abs((paramMilc.ssplaq())/3.0-SpatialPlaq) > 1e-5){
                    throw std::runtime_error(stdLogger.fatal("Error ssplaq!"));
                }
                if(abs((paramMilc.stplaq())/3.0-TemporalPlaq) > 1e-5){
                    throw std::runtime_error(stdLogger.fatal("Error stplaq!"));
                }
            }

        }
    }

    /// Exchange Halos
    gauge.updateAll();

   

    /// Initialize ReductionBase.
    LatticeContainer<true,PREC> redBase(commBase);

    /// We need to tell the Reductionbase how large our array will be. Again it runs on the spacelike volume only,
    /// so make sure you adjust this parameter accordingly, so that you don't waste memory.
    typedef GIndexer<All,HaloDepth> GInd;
    redBase.adjustSize(GInd::getLatData().vol4);
    rootLogger.info( "volume size " , GInd::getLatData().globvol4  );

//// Wilson Flow

    if(param.use_wilson()){
        rootLogger.info( "Start Wilson Flow"  );

        std::vector<PREC> flowTimes = {100000.0};
        PREC start = param.wilson_start();
        PREC stop  = param.wilson_stop();
        PREC step_size = param.wilson_step();
        const auto force = static_cast<Force>(static_cast<int>(0));
        gradientFlow<PREC, HaloDepth, fixed_stepsize,force> gradFlow(gauge,step_size,start,stop,flowTimes,0.0001);

        bool continueFlow =  gradFlow.continueFlow();
//	rootLogger.info() << "step " << gradFlow._step_size;
//	rootLogger.info() << "continueFlow " << continueFlow;
//	rootLogger.info() << "step " << gradFlow._step_size;
        while (continueFlow) {
            gradFlow.updateFlow();
//            rootLogger.info() << "step " << gradFlow._step_size;
            continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached
//	    rootLogger.info() << "step " << gradFlow._step_size;
//	    gradFlow.updateFlow();
	}

        gauge.updateAll();

        rootLogger.info( "End Wilson Flow"  );
    }

    gaugeOriginal = gauge;


    std::string Name = "WLoop_";
    if(param.load_conf() == 2 || param.load_conf() == 1){
        Name.append(param.gauge_file());
  	if(param.use_wilson()){
	    Name.append("_");
	    string s = std::to_string(param.wilson_stop());
            Name.append(s);
	}	
    }
    else{
        Name.append("one");
    }
    FileWriter file(gauge.getComm(), param, Name);


    rootLogger.info( "start wilson  loop" );

         StopWatch<true> timer;

    /// Start timer.
    timer.start();

    /// Exchange Halos
    gauge.updateAll();


     WilsonLineCorrelatorMultiGPU<PREC,HaloDepth,STACKS> WilsonClass;

    std::vector<PREC> dotVector;
    PREC * results;
    results = new PREC[STACKS*GInd::getLatData().globLT];
    ///  
    timer.start();
    //// loop over length of wilson lines
    for(int length = 1; length<GInd::getLatData().globLT+1;length++){

        /// calculate the wilson line starting from any spacetime point save in mu=0 direction
        WilsonClass.gWilson(gauge, length);

        dotVector = WilsonClass.gWilsonLoop(gauge,gaugeOriginal,length,redBase);

        for(int j = 0;j < STACKS ; j++){
            results[j+STACKS*(length-1)] = dotVector[j];
        }
            


        for(int j = 0;j < STACKS ; j++){
            file << j << " "  << length << " " << dotVector[j] << "\n";
        }
    }

    delete [] results;


    return 0;
}
