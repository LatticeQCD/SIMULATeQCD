#include "../simulateqcd.h"
#include "../experimental/fullSpinor.h"
#include "../experimental/DWilson.h"
#include "../experimental/source.h"
#include "../modules/hyp/hypSmearing.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include "../modules/gaugeFixing/gfix.h"
#include "../experimental/fourierNon2.h"


template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;
    Parameter<double,1>  mass; 
    Parameter<double,1>  mass2;
    Parameter<double,1>  csw;
    Parameter<int, 4> sourcePos;
    Parameter<int, 4> sources;
    Parameter<double,1>  smear1;
    Parameter<int,1>  smearSteps1;
    Parameter<double,1>  smear2;
    Parameter<int,1>  smearSteps2;
    Parameter<double,1> tolerance;
    Parameter<int,1> maxiter;
    Parameter<int,1> use_hyp;
    Parameter<int,1> use_mass2;
    Parameter<floatT> wilson_step;
    Parameter<floatT> wilson_start;
    Parameter<floatT> wilson_stop;
    Parameter<int,1> use_wilson;

    Parameter<floatT>      gtolerance;
    Parameter<int,1>       maxgfsteps;
    Parameter<int,1>       numunit;

    Parameter<int,1>       nSources;

    Parameter <std::string> source1_file;
    Parameter <std::string> source1F_file;
    Parameter <std::string> source2_file;
    Parameter <std::string> source2F_file;
    Parameter <std::string> source3_file;
    Parameter <std::string> source3F_file;
    Parameter <std::string> source4_file;
    Parameter <std::string> source4F_file;

    Parameter<int,1> nP;
    Parameter<int,1> measure_I;
    Parameter<int,1> measure_g5;
    Parameter<int,1> measure_gi;
    Parameter<int,1> measure_gig5;
    Parameter<int,1> measure_g4;
    Parameter<int,1> measure_gig4;


    wilsonParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
        add(mass, "mass");
        add(mass2, "mass2");
        add(csw, "csw");
        add(sourcePos, "sourcePos");
        add(sources, "sources");
        add(smear1, "smear1");
        add(smearSteps1, "smearSteps1");
        add(smear2, "smear2");
        add(smearSteps2, "smearSteps2");
        add(maxiter, "maxiter");
        add(tolerance, "tolerance");
        addDefault (use_hyp,"use_hyp",0);
        add(use_mass2, "use_mass2");
        addDefault (use_wilson,"use_wilson",0);
        addDefault (wilson_step,"wilson_step",0.0);
        addDefault (wilson_start,"wilson_start",0.0);
        addDefault (wilson_stop,"wilson_stop",0.0);

        addDefault (gtolerance,"gtolerance",1e-6);
        addDefault (maxgfsteps,"maxgfsteps",9000);
        addDefault (numunit   ,"numunit"   ,20);

        add(nSources   ,"nSources"  );
        add(source1_file, "source1_file");
        add(source1F_file, "source1F_file");
        add(source2_file, "source2_file");
        add(source2F_file, "source2F_file");
        add(source3_file, "source3_file");
        add(source3F_file, "source3F_file");
        add(source4_file, "source4_file");
        add(source4F_file, "source4F_file");
        
        add(nP,"nP");
        add(measure_I,"measure_I");
        add(measure_g5,"measure_g5");
        add(measure_gi,"measure_gi");
        add(measure_gig5,"measure_gig5");
        add(measure_g4,"measure_g4");
        add(measure_gig4,"measure_gig4");

    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    using PREC = double;
    const size_t mrhs = 4;

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
    PREC mass = param.mass();
    PREC mass2 = param.mass2();
    PREC csw = param.csw();
    size_t sourcePos[4];
    sourcePos[0]=param.sourcePos()[0];
    sourcePos[1]=param.sourcePos()[1];
    sourcePos[2]=param.sourcePos()[2];
    sourcePos[3]=param.sourcePos()[3];

    int measure_I = param.measure_I();
    int measure_g5 = param.measure_g5();
    int measure_gi = param.measure_gi();
    int measure_gig5 = param.measure_gig5();
    int measure_g4 = param.measure_g4();
    int measure_gig4 = param.measure_gig4();

    PREC lambda1 = param.smear1();
    int smearSteps1 = param.smearSteps1();
    PREC lambda2 = param.smear2();
    int smearSteps2 = param.smearSteps2();

    PREC tolerance = param.tolerance();
    int maxiter = param.maxiter();


    // file write
    std::string Name = "Mesons_m";
    Name.append(std::to_string(param.mass()));
    Name.append("_c");
    Name.append(std::to_string(param.csw()));
    Name.append("_"); 
    Name.append(param.gauge_file());
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


///////////// gauge fixing

//    if(param.load_conf() ==2){
        GaugeFixing<PREC,true,HaloDepth>    GFixing(gauge);
        int ngfstep=0;
        PREC gftheta=1e10;
        const PREC gtol = param.gtolerance();        //1e-6;          /// When theta falls below this number, stop...
        const int ngfstepMAX = param.maxgfsteps() ;  //9000;     /// ...or stop after a fixed number of steps; this way the program doesn't get stuck.
        const int nunit= param.numunit();            //20;            /// Re-unitarize every 20 steps.
        while ( (ngfstep<ngfstepMAX) && (gftheta>gtol) ) {
            /// Compute starting GF functional and update the lattice.
            GFixing.gaugefixOR();
            /// Due to the nature of the update, we have to re-unitarize every so often.
            if ( (ngfstep%nunit) == 0 ) {
                 gauge.su3latunitarize();
            }
            /// Re-calculate theta to determine whether we are sufficiently fixed.
            gftheta=GFixing.getTheta();
            ngfstep+=1;
        }
        gauge.su3latunitarize(); /// One final re-unitarization.

        rootLogger.info( "Gauge fixing finished in " , ngfstep , " steps, with gftheta = " , gftheta );
//    }



/// spinors after flow to save on maximum memory used

    Spinorfield<PREC, true, All, HaloDepth, 12, 12> ** spinor_out;
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_delta(commBase);    
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_wave(commBase);


    int nWave = param.nSources();
//    int Size = 1+nWave;

//    if(param.use_mass2()>0){
//         Size = 1+1;
//    }
    spinor_out = new Spinorfield<PREC, true, All, HaloDepth, 12, 12>*[nWave];
    for (int i = 0; i < nWave; i++) {
       spinor_out[i] = new Spinorfield<PREC, true, All, HaloDepth, 12, 12>(commBase); 
    }

    LatticeContainer<true,COMPLEX(PREC)> redBaseDevice(commBase);
    LatticeContainer<false,COMPLEX(PREC)> redBaseHost(commBase);
    Spinorfield<PREC, false, All, HaloDepth, 3, 1> spinor_host(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 3, 1> spinor_device(commBase);
    Spinorfield<PREC, false, All, HaloDepth, 12, 12> spinor_host12(commBase);

    std::string fname = param.source1_file();
    loadWave(fname, spinor_device,spinor_host,0, 0,commBase);

    fname = param.source1F_file();
    loadWave(fname, spinor_device,spinor_host,1, 0,commBase);

    if(nWave > 1){
       fname = param.source2_file();
       loadWave(fname, spinor_device,spinor_host,2, 0,commBase);
       fname = param.source2F_file();
       loadWave(fname, spinor_device,spinor_host,3, 0,commBase);
    }
    if(nWave > 2){
       fname = param.source3_file();
       loadWave(fname, spinor_device,spinor_host,4, 0,commBase);
       fname = param.source3F_file();
       loadWave(fname, spinor_device,spinor_host,5, 0,commBase);
    }
    if(nWave > 3){
       fname = param.source4_file();
       loadWave(fname, spinor_device,spinor_host,6, 0,commBase);
       fname = param.source4F_file();
       loadWave(fname, spinor_device,spinor_host,7, 0,commBase);
    }


    //calculate plaq
    GaugeAction<PREC, true, HaloDepth, R18> gaugeaction(gauge);
    PREC AveragePlaq = gaugeaction.plaquette();


    // make source class used to manipulate or create the source
    Source source;

    // start timer
    timer.reset();
    timer.start();

    int nP = param.nP();
    int nMomentum = (2*nP+1)*(2*nP+1)*(2*nP+1);

    // dont split the t direction
    size_t lt = GInd::getLatData().globLT;
    COMPLEX(PREC) CC_I[lt*nWave*nWave*nMomentum];
    COMPLEX(PREC) CC_g5[lt*nWave*nWave*nMomentum];
    COMPLEX(PREC) CC_gi[lt*nWave*nWave*nMomentum];
    COMPLEX(PREC) CC_gig5[lt*nWave*nWave*nMomentum];
    COMPLEX(PREC) CC_g4[lt*nWave*nWave*nMomentum];
    COMPLEX(PREC) CC_gig4[lt*nWave*nWave*nMomentum];


    //initialise results
    for (int t=0; t<nWave*nWave*GInd::getLatData().globLT*nMomentum; t++){
        CC_I[t] = 0.0;
        CC_g5[t] = 0.0;
        CC_gi[t] = 0.0;
        CC_gig5[t] = 0.0;
        CC_g4[t] = 0.0;
        CC_gig4[t] = 0.0;

    }

   // make class for inversion
   DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,mrhs> _dslashinverseSC4(gauge,mass,csw);

    //makeWaveSource(spinor_in,spinor_device,(1-1)*2,0,0);
    //fourier3D(spinor_out[0][0],spinor_in,redBaseDevice,redBaseHost,commBase);

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

                     for(int ss = 0; ss < nWave; ss ++){

                        _dslashinverseSC4.setMass(mass);
               //         if(ss==0){
               //             source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);
               //         }
               //         else{
                            if(pos[0] == 0 && pos[1] == 0 && pos[2] == 0){
                                makeWaveSource(spinor_in,spinor_device,(ss)*2,0,pos[3]);
                            }
                            else{
                                moveWave(spinor_device,spinor_host,pos[0],pos[1],pos[2],(ss)*2, 2,(ss)*2, 0 , commBase);
                                makeWaveSource(spinor_in,spinor_device,(ss)*2,2,pos[3]);
                           }                        
              //          }
 
                       _dslashinverseSC4.antiperiodicBoundaries();
                       _dslashinverseSC4.correlator(spinor_out[ss][0],spinor_in,maxiter,tolerance);
                       _dslashinverseSC4.antiperiodicBoundaries();

                       if(ss==0){
                           spinor_delta = spinor_out[ss][0];
		       }

                       fourier3D(spinor_out[ss][0],spinor_out[ss][0],redBaseDevice,redBaseHost,commBase);

                     }

                     for(int ss = 0; ss < nWave; ss ++){
                         for(int ss2 = 0; ss2 < nWave; ss2 ++){
                             
			     rootLogger.info( "check 1" , ss, ss2);	 
		             spinor_wave.template iterateOverBulk(SpinorXdaggerwave<PREC, HaloDepth,12>(spinor_out[ss][0],spinor_device,2*ss2+1,0));
                             fourier3D(spinor_wave,spinor_wave,redBaseDevice,redBaseHost,commBase,-1,12); 

                           rootLogger.info( "check 2" , ss, ss2);

                     /////////pion
                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
		     if(measure_g5 == 1){
		     spinor_in = spinor_delta;
                     tr_spinorXspinor(spinor_in,spinor_wave);

		     rootLogger.info( "check 3" , ss, ss2);
                     //tr_spinorXspinor(spinor_in,spinor_delta);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);
                      
		     gatherMomentumT(CC_g5,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
			     rootLogger.info( "check 4" , ss, ss2);
                     //    gatherMomentum(CC_g5,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                     //}
                     }


                     /////////rho
                     // tr( (g5*M^d*g5) gi M gi )
                     if(measure_gi == 1){
                     //x direction
                     spinor_in = spinor_delta;    
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gi,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gi,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);    
                         //CC_gi[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                 //    }
                     //y direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gi,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gi,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gi[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                 //    }
                     //z direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gi,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
			// gatherMomentum(CC_gi,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gi[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                    // }
                     }


                     //  scalar
                     // tr( (g5*M^d*g5) M )
		     if(measure_I == 1){
                     spinor_in = spinor_delta; 
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_I,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
//                     for (int t=0; t<GInd::getLatData().globLT; t++){
//			 gatherMomentum(CC_I,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_I[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
//                     }
                     }
                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )
                     if(measure_gig5 == 1){
                     //x direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig5,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig5,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                       //  CC_gig5[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
                     //y direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig5,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig5,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gig5[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
                     //z direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig5,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig5,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gig5[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }

                     }
                     // tr( (g5*M^d*g5) g4 M g4 )
		     if(measure_g4 == 1){
                     spinor_in = spinor_delta;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_g4,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_g4,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_g4[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
		     }

                     // tr( (g5*M^d*g5) gi g4 M gi g4 )
                     if(measure_gig4 == 1){
                     //x direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
		     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig4,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig4,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gig4[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
                     //y direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig4,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig4,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gig4[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
                     //z direction
                     spinor_in = spinor_delta;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_wave);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

		     gatherMomentumT(CC_gig4,spinor_in,spinor_host12, 0 ,GInd::getLatData().globLT*nMomentum*(ss+nWave*(ss2)),nP,pos[3],commBase);
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
		//	 gatherMomentum(CC_gig4,spinor_in,spinor_host12,t, 0 ,nMomentum*(ss+nWave*(ss2+nWave*t)),nMomentum,commBase);
                         //CC_gig4[ss+nWave*(ss2+nWave*t)] += sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(lt)),spinor_out[0][0],spinor_in,spinor_device,redBaseDevice,2*ss2+1,0,2);
                  //   }
                     }
  
                   }}
                }
            }
        }
    }
    timer.stop();
    timer.print("Time for all inversions and contractions");

    fileOut << "Average Plaquette " << AveragePlaq << "\n";
 
    fileOut << "time source_i source_j kx ky kz real(phi_j(r)*Gdelta(x+r)^dagger*G_i(x)) imag(phi_j(r)^dagger*Gdelta(x)^dagger*G_i(x+r))" << "\n";
    fileOut << "states ";
    if(measure_I == 1){     fileOut << " " << "real(I)"        << " " << "imag(I)";}
    if(measure_g5 == 1){    fileOut << " " << "real(g5)"       << " " << "imag(g5)";}
    if(measure_gi == 1){    fileOut << " " << "real(gi)"       << " " << "imag(gi)";}
    if(measure_gig5 == 1){  fileOut << " " << "real(gig5)"     << " " << "imag(gig5)";}
    if(measure_g4 == 1){    fileOut << " " << "real(g4)"       << " " << "imag(g4)";}
    if(measure_gig4 == 1){  fileOut << " " << "real(gig4)"     << " " << "imag(gig4)";}    
    fileOut << "\n";

    PREC norm = GInd::getLatData().globLX*GInd::getLatData().globLY*GInd::getLatData().globLZ;

    fileOut << "mass1" << "\n";
    int ktotal = -1;
    for(int kz = -nP; kz < nP+1; kz ++){
    for(int ky = -nP; ky < nP+1; ky ++){
    for(int kx = -nP; kx < nP+1; kx ++){
    ktotal++;
    for(int ss = 0; ss < nWave; ss ++){
       for(int ss2 = 0; ss2 < nWave; ss2 ++){
       for (int t=0; t<lt; t++){   
        fileOut << t << " " << ss << " " << ss2 << " " << kx  << " " << ky << " " << kz;
                    if(measure_I == 1){    fileOut << " " << norm*real(CC_I[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])    << " " << norm*imag(CC_I[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])    ;}
                    if(measure_g5 == 1){   fileOut << " " << norm*real(CC_g5[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   << " " << norm*imag(CC_g5[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   ;}
                    if(measure_gi == 1){   fileOut << " " << norm*real(CC_gi[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   << " " << norm*imag(CC_gi[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   ;}
                    if(measure_gig5 == 1){ fileOut << " " << norm*real(CC_gig5[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))]) << " " << norm*imag(CC_gig5[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))]) ;}
                    if(measure_g4 == 1){   fileOut << " " << norm*real(CC_g4[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   << " " << norm*imag(CC_g4[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))])   ;}
                    if(measure_gig4 == 1){ fileOut << " " << norm*real(CC_gig4[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))]) << " " << norm*imag(CC_gig4[t+lt*(ktotal+nMomentum*(ss+nWave*(ss2)))]) ;}
        fileOut << "\n";
    }}}
    }}}
    for (int i = 0; i < nWave; i++) {
        delete spinor_out[i];
    }
    delete spinor_out;
    return 0;
}
