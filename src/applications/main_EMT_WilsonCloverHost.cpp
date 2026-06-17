#include "../simulateqcd.h"
#include "../experimental/fullSpinor.h"
#include "../experimental/DWilson.h"
#include "../experimental/source.h"
#include "../modules/hyp/hypSmearing.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include "../experimental/derivatives.h"


template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;
    //Parameter<double,1>  mass; 
    //Parameter<double,1>  mass2;
    //Parameter<double,1>  csw;
    Parameter<int, 4> sourcePos;
    Parameter<int, 4> sources;
    //Parameter<double,1>  smear1;
    //Parameter<int,1>  smearSteps1;
    //Parameter<double,1>  smear2;
    //Parameter<int,1>  smearSteps2;
    Parameter<double,1> tolerance;
    Parameter<int,1> maxiter;
    Parameter<int,1> use_hyp;
    //Parameter<int,1> use_mass2;
    Parameter<floatT> wilson_step;
    Parameter<floatT> wilson_start;
    Parameter<floatT> wilson_stop;
    Parameter<int,1> use_wilson;
    Parameter<int,1> source_type;
    DynamicParameter<double> phase; 
    DynamicParameter<double> phaseU;
    Parameter<int,1> seed;
    DynamicParameter<double> input_masses;
    Parameter <std::string> name_end;
    DynamicParameter<double> input_flows;
    DynamicParameter<double> input_csw;

    wilsonParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
        //add(mass, "mass");
        //add(mass2, "mass2");
        //add(csw, "csw");
        add(sourcePos, "sourcePos");
        add(sources, "sources");
        //add(smear1, "smear1");
        //add(smearSteps1, "smearSteps1");
        //add(smear2, "smear2");
        //add(smearSteps2, "smearSteps2");
        add(maxiter, "maxiter");
        add(tolerance, "tolerance");
        addDefault (use_hyp,"use_hyp",0);
        //add(use_mass2, "use_mass2");
        addDefault (use_wilson,"use_wilson",0);
        addDefault (wilson_step,"wilson_step",0.0);
        addDefault (wilson_start,"wilson_start",0.0);
        addDefault (wilson_stop,"wilson_stop",0.0);
	addDefault (source_type,"source_type",0);
        add(phase, "phase");
	add(phaseU, "phaseU");
	add(seed, "seed");
	add(input_masses, "input_masses");
	add(name_end, "name_end");
	add(input_flows, "input_flows");
	add(input_csw, "input_csw");
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    using PREC = double;
    const size_t mrhs = 1;

    wilsonParam<PREC> param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/test.param", argc, argv);

    commBase.init(param.nodeDim());

    const size_t HaloDepth = 2;


    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    
    typedef GIndexer<All,HaloDepth> GInd;    

    // set parameters
    PREC mass; // = param.mass();
    //PREC mass2 = param.mass2();
    PREC csw;// = param.csw();
    size_t sourcePos[4];
    sourcePos[0]=param.sourcePos()[0];
    sourcePos[1]=param.sourcePos()[1];
    sourcePos[2]=param.sourcePos()[2];
    sourcePos[3]=param.sourcePos()[3];

    //PREC lambda1 = param.smear1();
    //int smearSteps1 = param.smearSteps1();
    //PREC lambda2 = param.smear2();
    //int smearSteps2 = param.smearSteps2();

    PREC tolerance = param.tolerance();
    int maxiter = param.maxiter();


    // file write
    std::string Name = "emt_";
    Name.append(param.gauge_file());
    Name.append(param.name_end());
    Name.append(".txt");
    FileWriter fileOut(commBase, param, Name);

    // timer
    StopWatch<true> timer;
    timer.start();

    // set up containers
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);

    std::string file_path = param.gauge_file_folder();
    file_path.append(param.gauge_file());

    gauge.readconf_nersc(file_path);
    gauge.updateAll();

////////////   hyp smearing

    if(param.use_hyp() > 0){
        for(int i = 0; i<param.use_hyp();i++){
            rootLogger.info( "Start hyp smearing"  );
            Gaugefield<PREC, true, HaloDepth> gauge_out(commBase);
            HypSmearing<PREC, true, HaloDepth ,R18> smearing(gauge);
            smearing.SmearAll(gauge_out);
            gauge = gauge_out;

       }
       rootLogger.info( "end hyp smearing"  );
       gauge.updateAll();
    }


////// wilson flow
    if(param.use_wilson() == 1){
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

    if(param.use_wilson() == 2){

	if(HaloDepth < 3 &&  !(param.nodeDim()[0] == 1 && param.nodeDim()[1] == 1 && param.nodeDim()[2] == 1 && param.nodeDim()[3] == 1 ) ){
            throw std::runtime_error(stdLogger.fatal("Error in Zeuthen flow: Zeuthen flow needs halo size 3, but rest of program is currently only initialized with halosize 2, sorry for inconvinience"));
        }

        rootLogger.info( "Start Z Flow"  );

        std::vector<PREC> flowTimes = {100000.0};
        PREC start = param.wilson_start();
        PREC stop  = param.wilson_stop();
        PREC step_size = param.wilson_step();
        const auto force = static_cast<Force>(static_cast<int>(1));
        gradientFlow<PREC, HaloDepth, fixed_stepsize,force> gradFlow(gauge,step_size,start,stop,flowTimes,0.0001);

        bool continueFlow =  gradFlow.continueFlow();
        while (continueFlow) {
            gradFlow.updateFlow();
            continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached
        }

        gauge.updateAll();

        rootLogger.info( "End Z Flow"  );
    }



/// spinors after flow to save on maximum memory used

    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out_s(commBase);


    //calculate plaq
    GaugeAction<PREC, true, HaloDepth, R18> gaugeaction(gauge);
    PREC AveragePlaq = gaugeaction.plaquette();


    // make source class used to manipulate or create the source
    Source source;

    // start timer
    timer.reset();
    timer.start();

    /*
    // dont split the t direction
    size_t lt = GInd::getLatData().globLT;
    COMPLEX(PREC) CC_l_I[lt];
    COMPLEX(PREC) CC_l_g5[lt];
    COMPLEX(PREC) CC_l_gi[lt];
    COMPLEX(PREC) CC_l_gig5[lt];
    COMPLEX(PREC) CC_l_g4[lt];
    COMPLEX(PREC) CC_l_gig4[lt];

    COMPLEX(PREC) CC_s_I[lt];
    COMPLEX(PREC) CC_s_g5[lt];
    COMPLEX(PREC) CC_s_gi[lt];
    COMPLEX(PREC) CC_s_gig5[lt];
    COMPLEX(PREC) CC_s_g4[lt];
    COMPLEX(PREC) CC_s_gig4[lt];

    COMPLEX(PREC) CC_ls_I[lt];
    COMPLEX(PREC) CC_ls_g5[lt];
    COMPLEX(PREC) CC_ls_gi[lt];
    COMPLEX(PREC) CC_ls_gig5[lt];
    COMPLEX(PREC) CC_ls_g4[lt];
    COMPLEX(PREC) CC_ls_gig4[lt];



    //initialise results
    for (int t=0; t<GInd::getLatData().globLT; t++){
        CC_l_I[t] = 0.0;
        CC_l_g5[t] = 0.0;
        CC_l_gi[t] = 0.0;
        CC_l_gig5[t] = 0.0;
        CC_l_g4[t] = 0.0;
        CC_l_gig4[t] = 0.0;

        CC_s_I[t] = 0.0;
        CC_s_g5[t] = 0.0;
        CC_s_gi[t] = 0.0;
        CC_s_gig5[t] = 0.0;
        CC_s_g4[t] = 0.0;
        CC_s_gig4[t] = 0.0;

        CC_ls_I[t] = 0.0;
        CC_ls_g5[t] = 0.0;
        CC_ls_gi[t] = 0.0;
        CC_ls_gig5[t] = 0.0;
        CC_ls_g4[t] = 0.0;
        CC_ls_gig4[t] = 0.0;

    }
     */


/*
  // gauge contribution

    COMPLEX(double) dmu_res_g_s =  calc_emt_Fmunu_Individual<PREC,true,HaloDepth,0>(gauge);
    rootLogger.info( "dmu_res_g_s      " ,  dmu_res_g_s/GInd::getLatData().globvol4 );
    fileOut <<       "dmu_res_g_s         " << real(dmu_res_g_s/GInd::getLatData().globvol4) << " " << imag(dmu_res_g_s/GInd::getLatData().globvol4) << "\n";

    COMPLEX(double) dmu_res_g_t =  calc_emt_Fmunu_Individual<PREC,true,HaloDepth,1>(gauge);
    rootLogger.info( "dmu_res_g_t      " ,  dmu_res_g_t/GInd::getLatData().globvol4 );
    fileOut <<       "dmu_res_g_t         " << real(dmu_res_g_t/GInd::getLatData().globvol4) << " " << imag(dmu_res_g_t/GInd::getLatData().globvol4) << "\n";

    COMPLEX(double) dmu_res_g =  calc_emt_Fmunu(gauge);
    rootLogger.info( "dmu_res_g*2/3    " ,  dmu_res_g/GInd::getLatData().globvol4 );
    fileOut <<       "dmu_res_g*2/3    " << (2.0/3.0)*real(dmu_res_g/GInd::getLatData().globvol4) << " " << (2.0/3.0)*imag(dmu_res_g/GInd::getLatData().globvol4) << "\n";
*/

  // random part
  grnd_state<true> d_rand;
  initialize_rng(param.seed(), d_rand);

   // make class for inversion
   //DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,mrhs> _dslashinverseSC4(gauge,param.input_masses.get()[0],csw);
   
   COMPLEX(PREC) phase, phase1, phase_alt;
   int count = 0;
   int count2 = 0;
   PREC lastFlow = 0.0;
   for (const auto& flowIn : param.input_flows.get()) {
      fileOut << "\n";

    ////// wilson flow
      if(param.use_wilson() == 3){
	   fileOut << "wilson_flow       " << flowIn << "\n";
           rootLogger.info( "Start Wilson Flow"  );

           std::vector<PREC> flowTimes = {100000.0};
           PREC start = 0.0;
           PREC stop  = flowIn-lastFlow;
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
	   lastFlow = flowIn;
       }

       if(param.use_wilson() == 4){

          if(HaloDepth < 3 &&  !(param.nodeDim()[0] == 1 && param.nodeDim()[1] == 1 && param.nodeDim()[2] == 1 && param.nodeDim()[3] == 1 ) ){
             throw std::runtime_error(stdLogger.fatal("Error in Zeuthen flow: Zeuthen flow needs halo size 3, but rest of program is currently only initialized with halosize 2, sorry for inconvinience"));
          }
          fileOut << "zeuthen_flow     " << flowIn << "\n";
          rootLogger.info( "Start Z Flow"  );

          std::vector<PREC> flowTimes = {100000.0};
          PREC start = 0.0;
          PREC stop  = flowIn-lastFlow;
          PREC step_size = param.wilson_step();
          const auto force = static_cast<Force>(static_cast<int>(1));
          gradientFlow<PREC, HaloDepth, fixed_stepsize,force> gradFlow(gauge,step_size,start,stop,flowTimes,0.0001);

          bool continueFlow =  gradFlow.continueFlow();
          while (continueFlow) {
            gradFlow.updateFlow();
            continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached
          }

          gauge.updateAll();

          rootLogger.info( "End Z Flow"  );
	  lastFlow = flowIn;
        }


  // gauge contribution

        COMPLEX(double) dmu_res_g_s =  calc_emt_Fmunu_Individual<PREC,true,HaloDepth,0>(gauge);
        rootLogger.info( "dmu_res_g_s      " ,  dmu_res_g_s/GInd::getLatData().globvol4 );
        fileOut <<       "dmu_res_g_s      " << real(dmu_res_g_s/GInd::getLatData().globvol4) << " " << imag(dmu_res_g_s/GInd::getLatData().globvol4) << "\n";

        COMPLEX(double) dmu_res_g_t =  calc_emt_Fmunu_Individual<PREC,true,HaloDepth,1>(gauge);
        rootLogger.info( "dmu_res_g_t      " ,  dmu_res_g_t/GInd::getLatData().globvol4 );
        fileOut <<       "dmu_res_g_t      " << real(dmu_res_g_t/GInd::getLatData().globvol4) << " " << imag(dmu_res_g_t/GInd::getLatData().globvol4) << "\n";

        COMPLEX(double) dmu_res_g =  calc_emt_Fmunu(gauge);
        rootLogger.info( "dmu_res_g*2/3    " ,  dmu_res_g/GInd::getLatData().globvol4 );
        fileOut <<       "dmu_res_g*2/3    " << (2.0/3.0)*real(dmu_res_g/GInd::getLatData().globvol4) << " " << (2.0/3.0)*imag(dmu_res_g/GInd::getLatData().globvol4) << "\n";

        csw=param.input_csw.get()[count2];
        DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,mrhs> _dslashinverseSC4(gauge,param.input_masses.get()[0],param.input_csw.get()[count2]);
	count2 ++;

        for (int mm = 0; mm < param.input_masses.get().size()/param.input_flows.get().size(); mm++){

        mass = param.input_masses.get()[count];
        phase = COMPLEX(PREC)(cos(param.phase.get()[count]),sin(param.phase.get()[count]));
        phase1 = COMPLEX(PREC)(cos(param.phase.get()[count]/GInd::getLatData().globLT),sin(param.phase.get()[count]/GInd::getLatData().globLT));
	phase_alt = COMPLEX(PREC)(cos(param.phaseU.get()[count]/GInd::getLatData().globLT),sin(param.phaseU.get()[count]/GInd::getLatData().globLT));
	rootLogger.info( "mass   " ,  mass );
	rootLogger.info( "phase  "  ,  phase  );
	fileOut << "mass             " << mass-4.0 << "\n";
	fileOut << "phase            " << param.phase.get()[count] << "\n";
        fileOut << "phase_U          " << param.phaseU.get()[count] << "\n";
	fileOut << "csw              " << csw << "\n";

        count ++;
   
    
    for (int px=0; px<GInd::getLatData().globLX; px+= GInd::getLatData().globLX/(param.sources()[0])){
        for (int py=0; py<GInd::getLatData().globLY; py+= GInd::getLatData().globLY/(param.sources()[1])){
            for (int pz=0; pz<GInd::getLatData().globLZ; pz+= GInd::getLatData().globLZ/(param.sources()[2])){
                for (int pt=0; pt<GInd::getLatData().globLT; pt+= GInd::getLatData().globLT/(param.sources()[3])){

                     int pos[4];
                     pos[0] = (sourcePos[0]+px)%GInd::getLatData().globLX;
                     pos[1] = (sourcePos[1]+py)%GInd::getLatData().globLY;
                     pos[2] = (sourcePos[2]+pz)%GInd::getLatData().globLZ;
                     pos[3] = (sourcePos[3]+pt)%GInd::getLatData().globLT;

                     //version that gives the correlator from input spinor only (spinorAll inside class instead)

                        //grnd_state<true> d_rand;
                        //initialize_rng(seed+count, d_rand);
                        spinor_in.gauss(d_rand.state);

                        // light mass
                        _dslashinverseSC4.setMass(mass);
			if( param.source_type() == 0){
                            source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);
                        }
		        else if( param.source_type() == 1){
                             source.makeWallSource(spinor_in,pos[3]);
			}
			else if( param.source_type() == 2){
                              spinor_in.gauss(d_rand.state);
			}
                       anyperiodicBoundaries<PREC,HaloDepth>(gauge,-phase);
                       setU4<PREC,HaloDepth>(gauge,phase_alt);
		       //_dslashinverseSC4.antiperiodicBoundaries();
               //        if(smearSteps1 > 0){
               //            source.smearSource(gauge,spinor_out,spinor_in,lambda1,smearSteps1);
               //        }
                       _dslashinverseSC4.correlator(spinor_out,spinor_in,maxiter,tolerance);
               //        if(smearSteps1 > 0){
               //           source.smearSource(gauge,spinor_in,spinor_out,lambda1,smearSteps1);
               //        }
                       //_dslashinverseSC4.antiperiodicBoundaries();
		       setU4<PREC,HaloDepth>(gauge,1.0/phase_alt);
                       anyperiodicBoundaries<PREC,HaloDepth>(gauge,-1.0/phase);


                     // get normalization of random vector

		     COMPLEX(PREC) dmu_res_f, dmu_res_f_diff, dmu_res_f_x, dmu_res_f_y, dmu_res_f_z, dmu_res_f_t, dmu_res_f_p, dmu_res_f_p_diff, dmu_res_f_t_p;
                     COMPLEX(PREC) norm = 0.0;
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         norm +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_in);
                     }		     
  
		     anyperiodicBoundaries<PREC,HaloDepth>(gauge,-phase);

                     //T_f_x
		     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,0>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_x = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_x +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_x/norm     " ,  dmu_res_f_x/norm );

                     fileOut << "dmu_res_f_x      " << real(dmu_res_f_x/norm) << " " << imag(dmu_res_f_x/norm) << "\n";

		     //T_f_y
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,1>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_y = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_y +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_y/norm     " ,  dmu_res_f_y/norm );

                     fileOut << "dmu_res_f_y      " << real(dmu_res_f_y/norm) << " " << imag(dmu_res_f_y/norm) << "\n";

		     //T_f_z
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,2>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_z = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_z +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_z/norm     " ,  dmu_res_f_z/norm );

                     fileOut << "dmu_res_f_z      " << real(dmu_res_f_z/norm) << " " << imag(dmu_res_f_z/norm) << "\n";

		     //T_f_t
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,3>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_t = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_t +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_t/norm     " ,  dmu_res_f_t/norm );

                     fileOut << "dmu_res_f_t      " << real(dmu_res_f_t/norm) << " " << imag(dmu_res_f_t/norm) << "\n";

                     //T_f_t_phase
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,3>(gauge,spinor_out_s,spinor_out,1.0/phase_alt);

                     dmu_res_f_t_p = 0.0;
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_t_p +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_t/norm     " ,  dmu_res_f_t_p/norm );

                     fileOut << "dmu_res_f_t_p    " << real(dmu_res_f_t_p/norm) << " " << imag(dmu_res_f_t_p/norm) << "\n";

                     // (T_f_x+T_f_y+T_f_z)/3-T_f_t

		     applyDmu(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f = 0.0;// spinor_out.dotProduct(spinor_in); 
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f/norm       " ,  dmu_res_f/norm );
                     
                     fileOut << "dmu_res_f        " << real(dmu_res_f/norm) << " " << imag(dmu_res_f/norm) << "\n";

                     dmu_res_f_diff = (dmu_res_f_x+dmu_res_f_y+dmu_res_f_z)/3.0-dmu_res_f_t-dmu_res_f;
                     fileOut << "dmu_res_f_diff   " << real(dmu_res_f_diff/norm) << " " << imag(dmu_res_f_diff/norm) << "\n";

                     // (T_f_x+T_f_y+T_f_z)/3-T_f_t_plus*phase-T_f_t_minus*phase^-1

                     applyDmu(gauge,spinor_out_s,spinor_out,1.0/phase_alt);

                     dmu_res_f_p = 0.0;// spinor_out.dotProduct(spinor_in); 
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_p +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_p/norm       " ,  dmu_res_f_p/norm );

                     fileOut << "dmu_res_f_p      " << real(dmu_res_f_p/norm) << " " << imag(dmu_res_f_p/norm) << "\n";

                     dmu_res_f_p_diff = (dmu_res_f_x+dmu_res_f_y+dmu_res_f_z)/3.0-dmu_res_f_t_p-dmu_res_f_p;
                     fileOut << "dmu_res_f_p_diff " << real(dmu_res_f_p_diff/norm) << " " << imag(dmu_res_f_p_diff/norm) << "\n";

                     anyperiodicBoundaries<PREC,HaloDepth>(gauge,-1.0/phase);
                     

                     /*
                     // for other boundary combination

		     anyperiodicBoundaries<PREC,HaloDepth>(gauge,-1.0);

                     //T_f_x
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,0>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_x = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_x +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_x/norm     " ,  dmu_res_f_x/norm );

                     fileOut << "dmu_res_f_2_x    " << real(dmu_res_f_x/norm) << " " << imag(dmu_res_f_x/norm) << "\n";

                     //T_f_y
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,1>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_y = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_y +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_y/norm     " ,  dmu_res_f_y/norm );

                     fileOut << "dmu_res_f_2_y    " << real(dmu_res_f_y/norm) << " " << imag(dmu_res_f_y/norm) << "\n";

                     //T_f_z
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,2>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_z = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_z +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_z/norm     " ,  dmu_res_f_z/norm );

                     fileOut << "dmu_res_f_2_z    " << real(dmu_res_f_z/norm) << " " << imag(dmu_res_f_z/norm) << "\n";

                     //T_f_t
                     applyDmu_Individual<PREC,true,HaloDepth,HaloDepth,12,3>(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f_t = 0.0;// spinor_out.dotProduct(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f_t +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f_t/norm     " ,  dmu_res_f_t/norm );

                     fileOut << "dmu_res_f_2_t    " << real(dmu_res_f_t/norm) << " " << imag(dmu_res_f_t/norm) << "\n";

                     // (T_f_x+T_f_y+T_f_z)/3-T_f_t

                     applyDmu(gauge,spinor_out_s,spinor_out,phase1);

                     dmu_res_f = 0.0;// spinor_out.dotProduct(spinor_in); 
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         dmu_res_f +=  _dslashinverseSC4.sumXYZ_TrMdaggerM(t,spinor_in,spinor_out_s);
                     }

                     rootLogger.info( "dmu_res_f/norm       " ,  dmu_res_f/norm );

                     fileOut << "dmu_res_f_2      " << real(dmu_res_f/norm) << " " << imag(dmu_res_f/norm) << "\n";

                     dmu_res_f_diff = (dmu_res_f_x+dmu_res_f_y+dmu_res_f_z)/3.0-dmu_res_f_t-dmu_res_f;
                     fileOut << "dmu_res_f_2_diff " << real(dmu_res_f_diff/norm) << " " << imag(dmu_res_f_diff/norm) << "\n";

                     anyperiodicBoundaries<PREC,HaloDepth>(gauge,-1.0);
                     */
                     //_dslashinverseSC4.antiperiodicBoundaries();

                }
            }
        }
    }

    }

    }
    timer.stop();
    timer.print("Time for all inversions and contractions");

/*

    fileOut << "Average Plaquette " << AveragePlaq << "\n";

    fileOut << "t" << " " << "real(I)"     << " " << "imag(I)"     <<
                      " " << "real(g5)"    << " " << "imag(g5)"    <<
                      " " << "real(gi)"    << " " << "imag(gi)"    <<
                      " " << "real(gi g5)" << " " << "imag(gi g5)" <<
                      " " << "real(g4)"    << " " << "imag(g4)"    <<
                      " " << "real(gi g4)" << " " << "imag(gi g4)" << "\n";

    fileOut << "mass1" << "\n";
    for (int t=0; t<lt; t++){   
        fileOut << t << " " << real(CC_l_I[t])    << " " << imag(CC_l_I[t])    <<
                        " " << real(CC_l_g5[t])   << " " << imag(CC_l_g5[t])   <<
                        " " << real(CC_l_gi[t])   << " " << imag(CC_l_gi[t])   <<
                        " " << real(CC_l_gig5[t]) << " " << imag(CC_l_gig5[t]) <<
                        " " << real(CC_l_g4[t])   << " " << imag(CC_l_g4[t])   <<
                        " " << real(CC_l_gig4[t]) << " " << imag(CC_l_gig4[t]) << "\n";
    }
    if(param.use_mass2()>0){
    fileOut << "mass2" << "\n";
    for (int t=0; t<lt; t++){
        fileOut << t << " " << real(CC_s_I[t])    << " " << imag(CC_s_I[t])    <<
                        " " << real(CC_s_g5[t])   << " " << imag(CC_s_g5[t])   <<
                        " " << real(CC_s_gi[t])   << " " << imag(CC_s_gi[t])   <<
                        " " << real(CC_s_gig5[t]) << " " << imag(CC_s_gig5[t]) <<
                        " " << real(CC_s_g4[t])   << " " << imag(CC_s_g4[t])   <<
                        " " << real(CC_s_gig4[t]) << " " << imag(CC_s_gig4[t]) << "\n";
    }
    fileOut << "mass to mass2" << "\n";
    for (int t=0; t<lt; t++){
        fileOut << t << " " << real(CC_ls_I[t])    << " " << imag(CC_ls_I[t])    <<
                        " " << real(CC_ls_g5[t])   << " " << imag(CC_ls_g5[t])   <<
                        " " << real(CC_ls_gi[t])   << " " << imag(CC_ls_gi[t])   <<
                        " " << real(CC_ls_gig5[t]) << " " << imag(CC_ls_gig5[t]) <<
                        " " << real(CC_ls_g4[t])   << " " << imag(CC_ls_g4[t])   <<
                        " " << real(CC_ls_gig4[t]) << " " << imag(CC_ls_gig4[t]) << "\n";
    }
    }

     */

    return 0;
}
