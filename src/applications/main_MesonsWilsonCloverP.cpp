#include "../simulateqcd.h"
#include "../experimental/fullSpinor.h"
#include "../experimental/DWilson.h"
#include "../experimental/source.h"
#include "../modules/hyp/hypSmearing.h"
#include "../modules/gradientFlow/gradientFlow.h"
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

    PREC lambda1 = param.smear1();
    int smearSteps1 = param.smearSteps1();
    PREC lambda2 = param.smear2();
    int smearSteps2 = param.smearSteps2();

    PREC tolerance = param.tolerance();
    int maxiter = param.maxiter();

    int measure_I = param.measure_I();
    int measure_g5 = param.measure_g5();
    int measure_gi = param.measure_gi();
    int measure_gig5 = param.measure_gig5();
    int measure_g4 = param.measure_g4();
    int measure_gig4 = param.measure_gig4();

    int nP = param.nP();
    int nMomentum = (2*nP+1)*(2*nP+1)*(2*nP+1);


    // file write
    std::string Name = "MesonsS_m";
    Name.append(std::to_string(param.mass()));
    Name.append("_c");
    Name.append(std::to_string(param.csw()));
    Name.append("_nS");
    Name.append(std::to_string(param.sources()[0]));
    Name.append(std::to_string(param.sources()[1]));
    Name.append(std::to_string(param.sources()[2]));
    Name.append(std::to_string(param.sources()[3]));
    Name.append("_Sp");
    Name.append(std::to_string(param.sourcePos()[0]));
    Name.append("_");
    Name.append(std::to_string(param.sourcePos()[1]));
    Name.append("_");
    Name.append(std::to_string(param.sourcePos()[2]));
    Name.append("_");
    Name.append(std::to_string(param.sourcePos()[3]));
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


/// spinors after flow to save on maximum memory used

    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);

    Spinorfield<PREC, true, All, HaloDepth, 12, 12> * spinor_out_s;

    if(param.use_mass2()>0){
         spinor_out_s = new Spinorfield<PREC, true, All, HaloDepth, 12, 12>(commBase);
    }

    LatticeContainer<true,COMPLEX(PREC)> redBaseDevice(commBase);
    LatticeContainer<false,COMPLEX(PREC)> redBaseHost(commBase);
    Spinorfield<PREC, false, All, HaloDepth, 12, 12> spinor_host12(commBase);



    //calculate plaq
    GaugeAction<PREC, true, HaloDepth, R18> gaugeaction(gauge);
    PREC AveragePlaq = gaugeaction.plaquette();


    // make source class used to manipulate or create the source
    Source source;

    // start timer
    timer.reset();
    timer.start();

    // dont split the t direction
    size_t lt = GInd::getLatData().globLT;
    COMPLEX(PREC) CC_l_I[lt*nMomentum];
    COMPLEX(PREC) CC_l_g5[lt*nMomentum];
    COMPLEX(PREC) CC_l_gi[lt*nMomentum];
    COMPLEX(PREC) CC_l_gig5[lt*nMomentum];
    COMPLEX(PREC) CC_l_g4[lt*nMomentum];
    COMPLEX(PREC) CC_l_gig4[lt*nMomentum];

    COMPLEX(PREC) CC_s_I[lt*nMomentum];
    COMPLEX(PREC) CC_s_g5[lt*nMomentum];
    COMPLEX(PREC) CC_s_gi[lt*nMomentum];
    COMPLEX(PREC) CC_s_gig5[lt*nMomentum];
    COMPLEX(PREC) CC_s_g4[lt*nMomentum];
    COMPLEX(PREC) CC_s_gig4[lt*nMomentum];

    COMPLEX(PREC) CC_ls_I[lt*nMomentum];
    COMPLEX(PREC) CC_ls_g5[lt*nMomentum];
    COMPLEX(PREC) CC_ls_gi[lt*nMomentum];
    COMPLEX(PREC) CC_ls_gig5[lt*nMomentum];
    COMPLEX(PREC) CC_ls_g4[lt*nMomentum];
    COMPLEX(PREC) CC_ls_gig4[lt*nMomentum];



    //initialise results
    for (int t=0; t<GInd::getLatData().globLT*nMomentum; t++){
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

   // make class for inversion
   DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,mrhs> _dslashinverseSC4(gauge,mass,csw);

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

                        // light mass
                        _dslashinverseSC4.setMass(mass);
                        source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);

                       _dslashinverseSC4.antiperiodicBoundaries();
                       if(smearSteps1 > 0){
                           source.smearSource(gauge,spinor_out,spinor_in,lambda1,smearSteps1);
                       }
                       _dslashinverseSC4.correlator(spinor_out,spinor_in,maxiter,tolerance);
                       if(smearSteps1 > 0){
                          source.smearSource(gauge,spinor_in,spinor_out,lambda1,smearSteps1);
                       }
                       _dslashinverseSC4.antiperiodicBoundaries();


                    if(param.use_mass2()>0){
                        // heavier mass
                        _dslashinverseSC4.setMass(mass2);
                        source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);

                       _dslashinverseSC4.antiperiodicBoundaries();
                       if(smearSteps2 > 0){
                           source.smearSource(gauge,*spinor_out_s,spinor_in,lambda2,smearSteps2);
                       }
                       _dslashinverseSC4.correlator(*spinor_out_s,spinor_in,maxiter,tolerance);
                       if(smearSteps2 > 0){
                          source.smearSource(gauge,spinor_in,*spinor_out_s,lambda2,smearSteps2);
                       }
                       _dslashinverseSC4.antiperiodicBoundaries();
                    }

                     /////////pion
                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
                     if(measure_g5 == 1){
	             spinor_in = spinor_out;
                     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_g5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

		     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_out);
                     //}

                     /////////rho
                     // tr( (g5*M^d*g5) gi M gi )
                     if(measure_gi == 1){
                     //x direction
                     spinor_in = spinor_out;    
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);
                     
                     gatherMomentumT(CC_l_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     }
                     //  scalar
                     // tr( (g5*M^d*g5) M )
		     if(measure_I == 1){
                     spinor_in = spinor_out; 
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     
		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_I,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
		     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )
                     if(measure_gig5 == 1){
                     //x direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     }
                     
                     // tr( (g5*M^d*g5) g4 M g4 )
                     if(measure_g4 == 1){
		     spinor_in = spinor_out;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     
		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_g4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     

                     // tr( (g5*M^d*g5) gi g4 M gi g4 )
                     if(measure_gig4 == 1){
                     //x direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_l_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     }

/////////////////s quark
                     if(param.use_mass2()>0){
                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
		     if(measure_g5 == 1){ 
                     spinor_in = *spinor_out_s;

	             tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_g5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,*spinor_out_s);
                     //}

                     // tr( (g5*M^d*g5) gi M gi )
                     if(measure_gi == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     }
                     //  scalar
                     // tr( (g5*M^d*g5) M )
		     if(measure_I == 1){
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     
		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_I,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,spinor_in);
                     //}

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )
                     if(measure_gig5 == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                    
                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     }

                     // tr( (g5*M^d*g5) g4 M g4 )
                     if(measure_g4 == 1){
		     spinor_in = *spinor_out_s;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     
		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_g4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,spinor_in);
                     //}


                     // tr( (g5*M^d*g5) gi g4 M gi g4 )
                     if(measure_gig4 == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),*spinor_out_s,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,*spinor_out_s);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_s_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),*spinor_out_s,spinor_in);
                     //}
                     }

///////////////  l s quarks

                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
		     if(measure_g5 == 1){
                     spinor_in = *spinor_out_s;
		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_g5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,*spinor_out_s);
                     //}

                     // tr( (g5*M^d*g5) gi M gi )
                     if(measure_gi == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gi,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     }
                     //  scalar
                     // tr( (g5*M^d*g5) M )
		     if(measure_I == 1){
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_I,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
		     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )
                     if(measure_gig5 == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig5,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     //}
                     }

                     // tr( (g5*M^d*g5) g4 M g4 )
		     if(measure_g4 == 1){
                     spinor_in = *spinor_out_s;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     
		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_g4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);
                     }
		     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}


                     // tr( (g5*M^d*g5) gi g4 M gi g4 )
                     if(measure_gig4 == 1){
                     //x direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //y direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     //z direction
                     spinor_in = *spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

		     tr_spinorXspinor(spinor_in,spinor_out);
                     fourier3D(spinor_in,spinor_in,redBaseDevice,redBaseHost,commBase,1,1);

                     gatherMomentumT(CC_ls_gig4,spinor_in,spinor_host12, 0 ,0,nP,pos,commBase);

                     //for (int t=0; t<GInd::getLatData().globLT; t++){
                     //    CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     //}
                     }
                    }//end check for use of mass2
                }
            }
        }
    }
    timer.stop();
    timer.print("Time for all inversions and contractions");

    fileOut << "Average Plaquette " << AveragePlaq << "\n";


    fileOut << "t";
    if(measure_I   == 1){ fileOut << " " << "real(I)"     << " " << "imag(I)"      ;}
    if(measure_g5   == 1){ fileOut << " " << "real(g5)"    << " " << "imag(g5)"    ;}
    if(measure_gi   == 1){ fileOut << " " << "real(gi)"    << " " << "imag(gi)"    ;}
    if(measure_gig5 == 1){ fileOut << " " << "real(gi g5)" << " " << "imag(gi g5)" ;}
    if(measure_g4   == 1){ fileOut << " " << "real(g4)"    << " " << "imag(g4)"    ;}
    if(measure_gig4 == 1){ fileOut << " " << "real(gi g4)" << " " << "imag(gi g4)" ;}
    fileOut << "\n";

    PREC norm = sqrt(1.0*GInd::getLatData().globLX*GInd::getLatData().globLY*GInd::getLatData().globLZ);

    fileOut << "mass1" << "\n";
    int ktotal = -1;
    for(int kz = -nP; kz < nP+1; kz ++){
    for(int ky = -nP; ky < nP+1; ky ++){
    for(int kx = -nP; kx < nP+1; kx ++){
        ktotal++;
        for (int t=0; t<lt; t++){
            fileOut << t << " " << kx  << " " << ky << " " << kz;
            if(measure_I   == 1){ fileOut << " " << norm*real(CC_l_I[t+lt*ktotal])    << " " << norm*imag(CC_l_I[t+lt*ktotal])    ;}
            if(measure_g5  == 1){ fileOut << " " << norm*real(CC_l_g5[t+lt*ktotal])   << " " << norm*imag(CC_l_g5[t+lt*ktotal])   ;}
            if(measure_gi  == 1){ fileOut << " " << norm*real(CC_l_gi[t+lt*ktotal])   << " " << norm*imag(CC_l_gi[t+lt*ktotal])   ;}
            if(measure_gig5== 1){ fileOut << " " << norm*real(CC_l_gig5[t+lt*ktotal]) << " " << norm*imag(CC_l_gig5[t+lt*ktotal]) ;}
            if(measure_g4  == 1){ fileOut << " " << norm*real(CC_l_g4[t+lt*ktotal])   << " " << norm*imag(CC_l_g4[t+lt*ktotal])   ;}
            if(measure_gig4== 1){ fileOut << " " << norm*real(CC_l_gig4[t+lt*ktotal]) << " " << norm*imag(CC_l_gig4[t+lt*ktotal]) ;}
            fileOut	<< "\n";
        }
    }}}
    
    if(param.use_mass2()>0){
    fileOut << "mass2" << "\n";
    ktotal = -1;
    for(int kz = -nP; kz < nP+1; kz ++){
    for(int ky = -nP; ky < nP+1; ky ++){
    for(int kx = -nP; kx < nP+1; kx ++){
        ktotal++;
        for (int t=0; t<lt; t++){
            fileOut << t << " " << kx  << " " << ky << " " << kz;
            if(measure_I   == 1){ fileOut << " " << norm*real(CC_s_I[t+lt*ktotal])    << " " << norm*imag(CC_s_I[t+lt*ktotal])    ;}
            if(measure_g5  == 1){ fileOut << " " << norm*real(CC_s_g5[t+lt*ktotal])   << " " << norm*imag(CC_s_g5[t+lt*ktotal])   ;}
            if(measure_gi  == 1){ fileOut << " " << norm*real(CC_s_gi[t+lt*ktotal])   << " " << norm*imag(CC_s_gi[t+lt*ktotal])   ;}
            if(measure_gig5== 1){ fileOut << " " << norm*real(CC_s_gig5[t+lt*ktotal]) << " " << norm*imag(CC_s_gig5[t+lt*ktotal]) ;}
            if(measure_g4  == 1){ fileOut << " " << norm*real(CC_s_g4[t+lt*ktotal])   << " " << norm*imag(CC_s_g4[t+lt*ktotal])   ;}
            if(measure_gig4== 1){ fileOut << " " << norm*real(CC_s_gig4[t+lt*ktotal]) << " " << norm*imag(CC_s_gig4[t+lt*ktotal]) ;}
            fileOut     << "\n";
        }
    }}}
    fileOut << "mass to mass2" << "\n";
    ktotal = -1;
    for(int kz = -nP; kz < nP+1; kz ++){
    for(int ky = -nP; ky < nP+1; ky ++){
    for(int kx = -nP; kx < nP+1; kx ++){
        ktotal++;
        for (int t=0; t<lt; t++){
            fileOut << t << " " << kx  << " " << ky << " " << kz;
            if(measure_I   == 1){ fileOut << " " << norm*real(CC_ls_I[t+lt*ktotal])    << " " << norm*imag(CC_ls_I[t+lt*ktotal])    ;}
            if(measure_g5  == 1){ fileOut << " " << norm*real(CC_ls_g5[t+lt*ktotal])   << " " << norm*imag(CC_ls_g5[t+lt*ktotal])   ;}
            if(measure_gi  == 1){ fileOut << " " << norm*real(CC_ls_gi[t+lt*ktotal])   << " " << norm*imag(CC_ls_gi[t+lt*ktotal])   ;}
            if(measure_gig5== 1){ fileOut << " " << norm*real(CC_ls_gig5[t+lt*ktotal]) << " " << norm*imag(CC_ls_gig5[t+lt*ktotal]) ;}
            if(measure_g4  == 1){ fileOut << " " << norm*real(CC_ls_g4[t+lt*ktotal])   << " " << norm*imag(CC_ls_g4[t+lt*ktotal])   ;}
            if(measure_gig4== 1){ fileOut << " " << norm*real(CC_ls_gig4[t+lt*ktotal]) << " " << norm*imag(CC_ls_gig4[t+lt*ktotal]) ;}
            fileOut     << "\n";
        }
    }}}
    }

    if(param.use_mass2()>0){
        delete spinor_out_s;
    }
    return 0;
}
