
#include "../modules/gaugeFixing/gfix.h"
#include "../modules/observables/polyakovLoop.h"
#include "../modules/gaugeFixing/polyakovLoopCorrelator.h"
#include "../modules/observables/wilsonLineCorrelator.h"

#include "../modules/gradientFlow/gradientFlow.h"

#define PREC double 

template<class floatT>
struct gfixParam : LatticeParameters {
    Parameter<floatT>      tolerance;
    Parameter<int,1>       maxgfsteps;
    Parameter<int,1>       numunit;
//    Parameter<std::string> measurements_dir;
    Parameter<std::string> SavedConfName;
    Parameter<bool>        SaveConfig;
    Parameter<bool>        SaveConfigNoZ;
    Parameter<bool>        SaveConfigZ;

    Parameter<int,1>       SaveRows;

    Parameter<floatT>  wilson_step;
    Parameter<floatT> wilson_start;
    Parameter<floatT> wilson_stop;
    Parameter<int,1> use_wilson;


    gfixParam() {
        addDefault (tolerance,"tolerance",1e-7);
        addDefault (maxgfsteps,"maxgfsteps",10000);
        addDefault (numunit   ,"numunit"   ,20);
        addDefault(SaveConfig         , "SaveConfig"         , false);
	addDefault(SaveConfigNoZ      , "SaveConfigNoZ"      , true);
	addDefault(SaveConfigZ        , "SaveConfigZ"        , true);
//        add(measurements_dir, "measurements_dir");
        add(SavedConfName   , "SavedConfName");

	addDefault(SaveRows   , "SaveRows", 2);

        addDefault (use_wilson,"use_wilson",0);
        addDefault (wilson_step,"wilson_step",0.0);
        addDefault (wilson_start,"wilson_start",0.0);
        addDefault (wilson_stop,"wilson_stop",0.0);


    }
};


int main(int argc, char *argv[]) {

/*
  Matrix4x4<double> test;
  test.a[0] =0.24202893435281503;
  test.a[1] =1.5208415494682805;
  test.a[2] =0.83357502790222;
  test.a[3] =1.6766244108634882;
  test.a[4] =1.5208415494682805;
  test.a[5] =0.5351580384621899;
  test.a[6] =1.0885253469748624;
  test.a[7] =0.8139163303289123;
  test.a[8] =0.8335750279022245;
  test.a[9] =1.0885253469748624;
  test.a[10] =0.6209110313314601;
  test.a[11] =1.8414683813904642;
  test.a[12] =1.6766244108634882;
  test.a[13] =0.8139163303289123;
  test.a[14] =1.8414683813904642;
  test.a[15] =1.6642400233510313;

  Matrix4x4<double> vec;
  double val[4];
  QR(vec,val,test);

  for( int i = 0; i < 4; i++){
     std::cout << val[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  for( int i = 0; i < 4; i++){
     for( int j = 0; j < 4; j++){
         std::cout << vec.a[i+4*j] << " ";
     }
     std::cout << std::endl;
  }
  std::cout << std::endl;

  double vecg[4];
  double vecb[4];
  vecb[0] = 0.3982472712749993;
  vecb[1] = 0.5993186712595366;
  vecb[2] = 0.8613324223284085;
  vecb[3] = 0.1056473671257283;

  getSU2Rotation(vecg,val,vecb, vec);

   for( int i = 0; i < 4; i++){
       std::cout << vecg[i] << " ";
   }
   std::cout << std::endl;
*/
////// actual program part

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 2;

     /// Read in parameters and initialize communication base.
    rootLogger.info("Initialization");
    CommunicationBase commBase(&argc, &argv);
    gfixParam<PREC> param;
    param.readfile(commBase, "../parameter/applications/gaugeFixing.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;

    Gaugefield<PREC,true,HaloDepth> gauge(commBase);
    Gaugefield<PREC,true,HaloDepth> gaugeZ(commBase);

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    //////////// wilson smearing
    if(param.use_wilson()){
        rootLogger.info( "Start Wilson Flow"  );

        std::vector<PREC> flowTimes = {100000.0};
        PREC start = param.wilson_start();
        PREC stop  = param.wilson_stop();
        PREC step_size = param.wilson_step();
        const auto force = static_cast<Force>(static_cast<int>(0));
        gradientFlow<PREC, HaloDepth, fixed_stepsize,force> gradFlow(gauge,step_size,start,stop,flowTimes,0.0001);

        bool continueFlow =  gradFlow.continueFlow();
        while (continueFlow) {
            gradFlow.updateFlow();
            continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached
        }

        gauge.updateAll();

        rootLogger.info( "End Wilson Flow"  );
    }


    //////////

    GaugeFixing<PREC,true,HaloDepth> GFixing(gauge);        /// gauge fixing class
    PolyakovLoop<PREC,true,HaloDepth> ploopClass(gauge);

    const int nunit= param.numunit();
    PREC gfR=GFixing.getR();
    PREC gfR_old = -10.0;
    rootLogger.info("R value " ,  gfR);
    for( int i = 0; i < param.maxgfsteps(); i++){
        GFixing.gaugefixR();

        gfR=GFixing.getR();
        rootLogger.info("R value " ,  gfR);

        COMPLEX(PREC) ploop = ploopClass.getPolyakovLoop();
        rootLogger.info("# POLYAKOV LOOP :: " ,  ploop);

        if ( (i%nunit) == 0 ) {
             gauge.su3latunitarize();
        }


        if(abs(gfR-gfR_old) < param.tolerance()) break;

	gfR_old = gfR;
    }

    if(param.SaveConfig()){
        gauge.writeconf_nersc(param.SavedConfName(), param.SaveRows(), 1);
    }
    GFixing.projectZ(gaugeZ);

    if(param.SaveConfigNoZ()){
        gauge.writeconf_nersc(param.SavedConfName()+"noZ", param.SaveRows(), 1);
    }

    if(param.SaveConfigZ()){    
        gaugeZ.writeconf_nersc(param.SavedConfName()+"Z", param.SaveRows(), 1);
    }


}

