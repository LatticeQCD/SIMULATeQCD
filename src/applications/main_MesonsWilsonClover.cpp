#include "../simulateqcd.h"
#include "../experimental/fullSpinor.h"
#include "../experimental/DWilson.h"
#include "../experimental/source.h"

template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;
    Parameter<double,1>  mass; 
    Parameter<double,1>  mass2;
    Parameter<double,1>  csw;
    Parameter<int, 4> sourcePos;
    Parameter<int, 4> sources;

    wilsonParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
        add(mass, "mass");
        add(mass2, "mass2");
        add(csw, "csw");
        add(sourcePos, "sourcePos");
        add(sources, "sources");
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    using PREC = double;

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
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);

    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out_s(commBase);

    std::string file_path = param.gauge_file_folder();
    file_path.append(param.gauge_file());

    gauge.readconf_nersc(file_path);
    gauge.updateAll();

    // make source
    Source source;

    // start timer
    timer.reset();
    timer.start();

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

   // make class for inversion
   DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,4> _dslashinverseSC4(gauge,mass,csw);

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
                     _dslashinverseSC4.setMass(mass);
                     source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);

                    _dslashinverseSC4.antiperiodicBoundaries();
                    _dslashinverseSC4.correlator(spinor_out,spinor_in,10000,1e-12);
                    _dslashinverseSC4.antiperiodicBoundaries();

                     // heavier mass
                     _dslashinverseSC4.setMass(mass2);
                     source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);

                    _dslashinverseSC4.antiperiodicBoundaries();
                    _dslashinverseSC4.correlator(spinor_out_s,spinor_in,10000,1e-12);
                    _dslashinverseSC4.antiperiodicBoundaries();


                     /////////pion
                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_out);
                     }

                     /////////rho
                     // tr( (g5*M^d*g5) gi M gi )
    
                     //x direction
                     spinor_in = spinor_out;    
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }

                     //  scalar
                     // tr( (g5*M^d*g5) M )
                     spinor_in = spinor_out; 
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )

                     //x direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }


                     // tr( (g5*M^d*g5) g4 M g4 )
                     spinor_in = spinor_out;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     

                     // tr( (g5*M^d*g5) gi g4 M gi g4 )

                     //x direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_l_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }


/////////////////s quark

                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_out_s);
                     }

                     // tr( (g5*M^d*g5) gi M gi )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }

                     //  scalar
                     // tr( (g5*M^d*g5) M )
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_in);
                     }

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }


                     // tr( (g5*M^d*g5) g4 M g4 )
                     spinor_in = spinor_out_s;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_in);
                     }


                     // tr( (g5*M^d*g5) gi g4 M gi g4 )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out_s,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_s_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out_s,spinor_in);
                     }


///////////////  l s quarks

                     // tr( (g5*M^d*g5)*g5*M*g5) = tr(M^d *M)
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_g5[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_out_s);
                     }

                     // tr( (g5*M^d*g5) gi M gi )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gi[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }

                     //  scalar
                     // tr( (g5*M^d*g5) M )
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_I[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }

                     // Axial vector
                     // tr( (g5*M^d*g5) gi g5 M gi g5 )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig5[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_in);
                     }


                     // tr( (g5*M^d*g5) g4 M g4 )
                     spinor_in = spinor_out_s;

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);
                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_g4[t] +=  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }


                     // tr( (g5*M^d*g5) gi g4 M gi g4 )

                     //x direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,0>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,0>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //y direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,1>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,1>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }
                     //z direction
                     spinor_in = spinor_out_s;
                     source.gammaMuRight<PREC,All,HaloDepth,2>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,2>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,3>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,3>(spinor_in);

                     source.gammaMuRight<PREC,All,HaloDepth,5>(spinor_in);
                     source.gammaMu<PREC,All,HaloDepth,12,5>(spinor_in);

                     for (int t=0; t<GInd::getLatData().globLT; t++){
                         CC_ls_gig4[t] +=_dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(lt),spinor_out,spinor_in);
                     }


                }
            }
        }
    }
    timer.stop();
    timer.print("Time for all inversions and contractions");
   
    fileOut << "t" << " " << "real(I)"     << " " << "imag(I)"     <<
                      " " << "real(g5)"    << " " << "imag(g5)"    <<
                      " " << "real(gi)"    << " " << "imag(gi)"    <<
                      " " << "real(gi g5)" << " " << "imag(gi g5)" <<
                      " " << "real(g4)"    << " " << "imag(g4)"    <<
                      " " << "real(g4 )"   << " " << "imag(g4 )"   << 
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


    return 0;
}
