#include "../simulateqcd.h"
#include "../experimental/fullSpinor.h"
#include "../experimental/source.h"
#include "../experimental/fourier.h"
#include "../experimental/DWilson.h"


template<class floatT>
struct fourierParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;
    Parameter <std::string> source1_file;
    Parameter <std::string> source1F_file;
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

    fourierParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
        add(source1_file, "source1_file");
        add(source1F_file, "source1F_file");
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

    }
};


int main(int argc, char *argv[]) {

    using PREC = double;

    fourierParam<PREC> param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/test.param", argc, argv);

    commBase.init(param.nodeDim());

    std::cout << commBase.mycoords()[0] << " " << commBase.mycoords()[1] << " " << commBase.mycoords()[2] << " " << commBase.mycoords()[3] << " " << std::endl;

    const size_t HaloDepth = 2;

    initIndexer(HaloDepth,param,commBase);

    LatticeContainer<true,COMPLEX(PREC)> redBaseDevice(commBase);
    LatticeContainer<false,COMPLEX(PREC)> redBaseHost(commBase);

    PREC tolerance = param.tolerance();
    int maxiter = param.maxiter();


    typedef GIndexer<All,HaloDepth> GInd;
    size_t sourcePos[4];
    sourcePos[0]=param.sourcePos()[0];
    sourcePos[1]=param.sourcePos()[1];
    sourcePos[2]=param.sourcePos()[2];
    sourcePos[3]=param.sourcePos()[3];
    int pos[4];
    pos[0] = (sourcePos[0]+0)%GInd::getLatData().globLX;
    pos[1] = (sourcePos[1]+0)%GInd::getLatData().globLY;
    pos[2] = (sourcePos[2]+0)%GInd::getLatData().globLZ;
    pos[3] = (sourcePos[3]+0)%GInd::getLatData().globLT;

    //// gauge field
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);

    std::string file_path = param.gauge_file_folder();
    file_path.append(param.gauge_file());

    gauge.readconf_nersc(file_path);
    gauge.updateAll();

    /// spinors

    Source source;
    
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_out(commBase);

    source.makePointSource(spinor_in,pos[0],pos[1],pos[2],pos[3]);
    spinor_in.updateAll();

    DWilsonInverseShurComplement<PREC,true,HaloDepth,HaloDepth,1> _dslashinverseSC4(gauge,4.0,1.0);
    _dslashinverseSC4.antiperiodicBoundaries();
    _dslashinverseSC4.correlator(spinor_out,spinor_in,maxiter,tolerance);
    _dslashinverseSC4.antiperiodicBoundaries();

    for (int t=0; t<GInd::getLatData().globLT; t++){
        COMPLEX(PREC) output =  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_out);
        rootLogger.info( "output " , output ); 
   }

    spinor_in = 0.0*spinor_in;
    fourier3D(spinor_in,spinor_out,redBaseDevice,redBaseHost,commBase);

    for (int t=0; t<GInd::getLatData().globLT; t++){
        COMPLEX(PREC) output =  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_in,spinor_in);
        rootLogger.info( "output2 " , output );
   }


//    for (int t=0; t<GInd::getLatData().globLT; t++){
//        COMPLEX(PREC) val = sumXYZ_TrMdaggerM((int)((t+pos[3])%(GInd::getLatData().globLT)),spinor_in,spinor_in,redBaseDevice);
//        std::cout << "val " << val.cREAL << " " << val.cIMAG << std::endl;
//    }

    Spinorfield<PREC, false, All, HaloDepth, 3, 1> spinor_host(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 3, 1> spinor_device(commBase);

    std::string fname = param.source1_file();
    loadWave(fname, spinor_device,spinor_host,0, 0,commBase);

    fname = param.source1F_file();
    loadWave(fname, spinor_device,spinor_host,1, 0,commBase);


    makeWaveSource(spinor_in,spinor_device,0,0,pos[3]);
    spinor_in.updateAll();
    _dslashinverseSC4.antiperiodicBoundaries();
    _dslashinverseSC4.correlator(spinor_out,spinor_in,maxiter,tolerance);
    _dslashinverseSC4.antiperiodicBoundaries();

    for (int t=0; t<GInd::getLatData().globLT; t++){
        COMPLEX(PREC) output =  _dslashinverseSC4.sumXYZ_TrMdaggerM((t+pos[3])%(GInd::getLatData().globLT),spinor_out,spinor_out);
        rootLogger.info( "output3 " , output );
   }


/*    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();
    global[3] = 1;
    local[3] = 1;

    commBase.initIOBinary(fname, 0, sizeof(PREC), 0, global, local, READ);

    std::vector<char> buf;
    buf.resize(local[0]*local[1]*local[2]*sizeof(PREC));
    commBase.readBinary(&buf[0], local[0]*local[1]*local[2]);
    int ps = 0;
    Vect3<PREC> tmp3;
    for ( int i = 0; i < 3; i ++){
        tmp3.data[i] = 0.0;
    }
    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++) 
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        PREC *data = (PREC *) &buf[ps];
        ps += sizeof(PREC);
        tmp3.data[0] = data[0];
        //std::cout << "data " << data[0] << std::endl;
        spinor_host.getAccessor().setElement(GInd::getSite(x,y, z, 0),tmp3);
    }

    commBase.closeIOBinary();

    spinor_device = spinor_host;
*/

    fourier3D(spinor_in,spinor_out,redBaseDevice,redBaseHost,commBase);   

    for (int t=0; t<GInd::getLatData().globLT; t++){
        COMPLEX(PREC) val = sumXYZ_TrMdaggerMwave((int)((t+pos[3])%(GInd::getLatData().globLT)),spinor_in,spinor_in,spinor_device,redBaseDevice,1,0);
        rootLogger.info( "output4 " , val*sqrt(GInd::getLatData().globLX*GInd::getLatData().globLY*GInd::getLatData().globLZ) );
    }

    // mpi test
    int coord[4];
//    for (int rank=0; rank<4; rank++){
//        MPI_Cart_coords(commBase.getCart_comm(), rank,4, coord);
//        rootLogger.info( "xyzt " , coord[0],coord[1],coord[2],coord[3]);
//    }

    MPI_Comm commX;
    int remain[4];
    remain[0] = 0;
    remain[1] = 0;
    remain[2] = 1;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(),remain, &commX);
    int myrank, rankSize;
    MPI_Comm_rank(commBase.getCart_comm(), &myrank);
    MPI_Comm_size(commBase.getCart_comm(), &rankSize);         

    int myrankX, rankSizeX;
    MPI_Comm_rank(commX, &myrankX);
    MPI_Comm_size(commX, &rankSizeX);
    int coordX[4];
    MPI_Cart_coords(commBase.getCart_comm(), myrank,4, coord);
    MPI_Cart_coords(commX, myrankX,4, coordX);
    std::cout << "my rank is " << myrank << "  out of " << rankSize << "   my rank X is " << myrankX << "  out of X " << rankSizeX << " cord " << coord[0] << " "  <<  coord[1] << " "  << coord[2] << " "  << coord[3] << " "  << 
                                                                                                                                      " cordX " << coordX[0] << " "  <<  coordX[1] << " "  << coordX[2] << " "  << coordX[3] << " "  <<    std::endl;   
    for (int rank=0; rank<rankSizeX; rank++){
        MPI_Cart_coords(commX, rank,4, coord);
        rootLogger.info( "xyzt " , myrank , coord[0],coord[1],coord[2],coord[3]);
    }


/*
    //all gather 4d
    int glx = 2;//GInd::getLatData().globLX;
    int lx  = 2;//GInd::getLatData().lx;
    int gly = 4;//GInd::getLatData().globLY;
    int ly  = 2;//GInd::getLatData().ly;
    int glz = 4;//GInd::getLatData().globLZ;
    int lz  = 2;//GInd::getLatData().lz;
    int glt = 2;//GInd::getLatData().globLT;
    int lt  = 2;//GInd::getLatData().lt;
    int myrank, rankSize;
    MPI_Comm_rank(commBase.getCart_comm(), &myrank);
    MPI_Comm_size(commBase.getCart_comm(), &rankSize);
    std::cout << "my rank is " << myrank << "  out of " << rankSize << std::endl;

    std::complex<double> *in = new std::complex<double>[lx*ly*lz*lt];
    std::complex<double> *buf = new std::complex<double>[glx*gly*glz*glt];
    std::complex<double> *out = new std::complex<double>[glx*gly*glz*glt];

    MPI_Cart_coords(commBase.getCart_comm(), myrank,4, coord);
    for (int x=0; x<lx; x++)
    for (int y=0; y<ly; y++)
    for (int z=0; z<lz; z++)
    for (int t=0; t<lt; t++){
        in[x+lx*(y+ly*(z+lz*(t+lt*0)))] =myrank*100+(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])+glz*((t+lt*coord[3]))));//myrank*100+x+lx*(y+ly*(z+lz*(t)));
    }
    MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX,commBase.getCart_comm() );
//    MPI_Allreduce(in, buf, nr, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
//    for (int i = 0; i < nr; i++) in[i] = buf[i];
    for (int r=0; r<rankSize; r++){
    for (int t=0; t<lt; t++)
    for (int z=0; z<lz; z++)
    for (int y=0; y<ly; y++)
    for (int x=0; x<lx; x++){
    rootLogger.info(buf[x+lx*(y+ly*(z+lz*(t+lt*r)))]);
    }    
    }

    for (int r=0; r<rankSize; r++){
    MPI_Cart_coords(commBase.getCart_comm(), r,4, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++){
            out[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])+glz*((t+lt*coord[3]))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
        }
    }

    for (int t=0; t<glt; t++)
    for (int z=0; z<glz; z++)
    for (int y=0; y<gly; y++)
    for (int x=0; x<glx; x++){
        rootLogger.info("combined    ",out[x+glx*(y+gly*(z+glz*(t)))]);
    }


    delete[] out;
    delete[] buf;
    delete[] in;
*/


    return 0;
}






