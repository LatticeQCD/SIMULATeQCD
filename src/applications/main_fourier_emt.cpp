#include "../simulateqcd.h"
#include "../experimental/fourierNon2.h"
#include "../base/math/matrix4x4SymComplex.h"
#include "../base/math/tensor4x4Symx4x4SymComplex.h"
#include "../modules/observables/energyMomentumTensor.h"
#include "../modules/observables/blocking.h"


#define USE_GPU true
// define precision
#if SINGLEPREC
    #define PREC float
#else
    #define PREC double
#endif

void customTest(bool boolean, std::string name) {
    if (boolean) {
        name.append(" passed!");
        rootLogger.testpassed(name);
    } else {
        name.append(" failed!");
        rootLogger.fatal(name);
    }
}

enum Summation {
    SpatialAndTemporal, Spatial
};

typedef int (* vFunctionCall)(int args);

template<Summation summation, typename type>
__device__ __host__ type reduceIndices(sitexyzt rt, vFunctionCall func) {
    
    int maxIndex = 0;

    if (summation == SpatialAndTemporal) {
        maxIndex = 4;
    } else if (summation == Spatial) {
        maxIndex = 3;
    }

    type result = 0;

    for (int i = 0; i < maxIndex; i++) {
        result += func(rt[i]);
    }

    return result;
}

__device__ __host__ int squareCoord(int coord) {
    return coord * coord;
}

template<Summation summation>
__device__ __host__ int r2(sitexyzt rt) {

    return reduceIndices<summation, int>(rt, squareCoord);

}


// define functor for combining two EMTs to one 4x4Symx4x4Sym
template<class floatT>
struct EMTtimesEMT {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    EMTtimesEMT(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) : _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        // int xCheck = 1;
        // int xCheck2 = 30;

        // bool check = (site.coord[0] == xCheck || site.coord[0] == xCheck2) && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 1;

        // if (check) {
        //     printGSite(site);
        // }

        // sitexyzt globalCoord = GInd::getLatData().globalPos(site.coord);

        // if (check) {
        //     printf("Global Coordinates: %i, %i, %i, %i\n", globalCoord[0], globalCoord[1], globalCoord[2], globalCoord[3]);
        // }

        // sitexyzt globalCoordRelativeToOrigin = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        // if (check) {
        //     printf("Global Coordinates relative to the origin: %i, %i, %i, %i\n", globalCoordRelativeToOrigin[0], globalCoordRelativeToOrigin[1], globalCoordRelativeToOrigin[2], globalCoordRelativeToOrigin[3]);
        // }

        // if (check) {
        //     int r2Value = r2<Spatial>(globalCoordRelativeToOrigin);
        //     printf("r^2: %i\n", r2Value);
        // }
        
        Matrix4x4SymComplex<floatT> firstElement(_firstAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));
        Matrix4x4SymComplex<floatT> secondElement(_secondAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> result(firstElement, secondElement);

        return result;
    }

};

// define functor to contract tensor indices
template<class floatT>
struct ContractGTensor {

    LatticeContainerAccessor _GAccessor;
    typedef GIndexer<All> GInd;

    ContractGTensor(LatticeContainerAccessor GAccessor) : _GAccessor(GAccessor) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        // define projector components
        // for (int mu = 0; mu <= 3; mu++)
        // for (int nu = 0; nu <= mu; nu++)
        // for (int rho = 0; rho <= 3; rho++)
        // for (int sigma = 0; sigma <= rho; sigma++) {
        //     projector(mu, nu, rho, sigma, projector_function(site, mu, nu, rho, sigma));
        // }

        // get element at the site
        Tensor4x4Symx4x4SymComplex tensor4x4Symx4x4SymComplexAtSite = _GAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        COMPLEX(floatT) result = 0.0;

        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result += tensor4x4Symx4x4SymComplexAtSite.elems[firstIndexPair][secondIndexPair];
            }
        }

        return result;

    }

};


// define functor to copy from device to host
template<class floatT>
struct CopyField {

    LatticeContainerAccessor _GAccessor;
    typedef GIndexer<All> GInd;

    CopyField(LatticeContainerAccessor GAccessor) : _GAccessor(GAccessor) {}

    __device__ __host__ inline floatT operator()(gSite site) {

        floatT elementDevice = _GAccessor.getElement<floatT>(site);

        return elementDevice;

    }

};


template<class floatT>
struct fourierParam : LatticeParameters {
    Parameter <std::string> gauge_file;
    Parameter <std::string> gauge_file_folder;

    fourierParam() {
        add(gauge_file, "gauge_file");
        add(gauge_file_folder, "gauge_file_folder");
    }
};


int main(int argc, char *argv[]) {

    // define the communication base for MPI stuff
    CommunicationBase commBase(&argc, &argv);

    rootLogger.setVerbosity(TRACE);
    
    // create a variable for the parameters
    fourierParam<PREC> param;

    // read the parameters from the 
    param.readfile(commBase, "../parameter/fourier_emt.param", argc, argv);

    // initialize the communication base based on the given parameters
    commBase.init(param.nodeDim());

    // manually set the halo depth to 2
    const size_t HaloDepth = 2;

    // initialize the indexer
    // std::cout << "Start: Initialize Index" << std::endl;
    initIndexer(HaloDepth, param, commBase);
    // std::cout << "End: Initialize Index" << std::endl;

    typedef GIndexer<All,HaloDepth> GInd;

    // define a variable for the gauge field
    // std::cout << "Start: Define Gaugefield" << std::endl;
    Gaugefield<PREC, true, HaloDepth> gaugeDevice(commBase);
    Gaugefield<PREC, false, HaloDepth> gaugeHost(commBase);
    // std::cout << "End: Define Gaugefield" << std::endl;
    
    // set path for the gauge file
    std::string file_path = param.gauge_file_folder();
    file_path.append(param.gauge_file());
    
    // read it in
    // std::cout << "Start: Read-in Gaugefield" << std::endl;
    gaugeDevice.readconf_nersc(file_path);
    gaugeHost.readconf_nersc(file_path);
    // std::cout << "End: Read-in Gaugefield" << std::endl;
    gaugeDevice.updateAll();
    gaugeHost.updateAll();
    
    // create lattice containers for EMT fields
    LatticeContainer<false, Matrix4x4SymComplex<PREC> > _redBaseEMTUComplexHost(commBase, "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUComplexDevice(commBase, "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUFourierTransformedForwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUFourierTransformedBackwards(commBase, "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUFourierTransformedForwardsBackwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS");
    
    // adjust their size
    _redBaseEMTUComplexHost.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUComplexDevice.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwards.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedBackwards.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwardsBackwards.adjustSize(GInd::getLatData().vol4);
    
    // calculate EMT on one of them
    _redBaseEMTUComplexHost.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, false, HaloDepth>(gaugeHost.getAccessor()));
    _redBaseEMTUComplexDevice.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, true, HaloDepth>(gaugeDevice.getAccessor()));
    // _redBaseEMTUFourier.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplexZero<PREC, true, HaloDepth>(gauge.getAccessor()));
    // _redBaseEMTUFourierTransformedForwardsBackwards.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplexZero<PREC, true, HaloDepth>(gauge.getAccessor()));
    
    Matrix4x4Sym<PREC> resultEMTU;
    Matrix4x4SymComplex<PREC> resultEMTUComplexDevice;
    Matrix4x4SymComplex<PREC> resultEMTUComplexHost;
    Matrix4x4SymComplex<PREC> resultEMTUFourierTransformedForwards;
    Matrix4x4SymComplex<PREC> resultEMTUFourierTransformedForwardsBackwards;
    
    // // create emt observable object
    // EnergyMomentumTensor<PREC, USE_GPU, HaloDepth> EMT(gauge);

    // // standard EMT averaging
    // EMT.EMTUAveraged(resultEMTU);

    // create Fourier class
    FourierClass<PREC> fourierClass(gaugeDevice.getComm());

    // EMT.EMTUFourierAveraged(resultEMTUFourier);

    // didn't need to declare them here
    // LatticeContainer<true, COMPLEX(PREC)> _redBaseDevice(commBase , "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device");
    // LatticeContainer<false, COMPLEX(PREC)> _redBaseHost(commBase , "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host");

    // perform Fourier transformation forwards
    fourierClass.performFourier3DEMT(_redBaseEMTUComplexDevice, _redBaseEMTUFourierTransformedForwards, 1.0);
    // perform Fourier transformation forwards
    fourierClass.performFourier3DEMT(_redBaseEMTUComplexDevice, _redBaseEMTUFourierTransformedBackwards, -1.0);
    // perform Fourier transformation backwards after the forwards
    fourierClass.performFourier3DEMT(_redBaseEMTUFourierTransformedForwards, _redBaseEMTUFourierTransformedForwardsBackwards, -1.0);

    _redBaseEMTUComplexDevice.reduce(resultEMTUComplexDevice, GInd::getLatData().vol4);
    _redBaseEMTUComplexHost.reduce(resultEMTUComplexHost, GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwards.reduce(resultEMTUFourierTransformedForwards, GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwardsBackwards.reduce(resultEMTUFourierTransformedForwardsBackwards, GInd::getLatData().vol4);
    
    // resultEMTUComplex /= (PREC) GInd::getLatData().vol4;
    // resultEMTUFourier /= (PREC) GInd::getLatData().vol4;
    // resultEMTUFourier2 /= (PREC) GInd::getLatData().vol4;

    std::cout << std::scientific << std::showpos << std::setprecision(15)
    << "Comparison for Fourier, Fouriered-back, Complex and Standard Averaged:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "\t" << i << ": " << std::scientific << std::showpos << std::setprecision(15) << resultEMTUFourierTransformedForwards.elems[i] << " "  << resultEMTUFourierTransformedForwardsBackwards.elems[i] << " " << resultEMTUComplexDevice.elems[i] << " " << resultEMTU.elems[i] << std::endl;
    }

    LatticeContainerAccessor _redBaseEMTUHostAccessor(_redBaseEMTUComplexHost.getAccessor());

    Matrix4x4SymComplex<PREC> complexAtZero = _redBaseEMTUHostAccessor.getElement<Matrix4x4SymComplex<PREC>>(GInd::getSite(0,0,0,0));

    complexAtZero *= sqrt(GInd::getLatData().vol4);

    std::cout << "Comparison for Fourier averaged, Complex at 0 and Complex averaged:" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << "\t" << i << ": " << std::scientific << std::showpos << std::setprecision(15) << resultEMTUFourierTransformedForwards.elems[i] << " " << complexAtZero.elems[i] << " " << resultEMTUComplexHost.elems[i] << std::endl;
    }

    // compare T_munu at r=0 and integrated FFT(T_munu) over all p
    customTest(compareMatrix4x4SymComplex(complexAtZero, resultEMTUFourierTransformedForwards, 1e-12), "Comparison of T_munu(r=0) and integrated T_munu(p)");

    // compare integrated T_munu(r) with integrated FFT^{-1}(FFT(T_munu))(r)
    customTest(compareMatrix4x4SymComplex(resultEMTUComplexDevice, resultEMTUFourierTransformedForwardsBackwards, 1e-12), "Comparison of integrated T_munu(r) and FFT^{-1}(FFT(T_munu))(r)");

    // -----------------------------------------------------------------------
    // Second Step: Combine Fourier-Transformed EMT Fields
    
    // define lattice containers for products
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> _redBaseGTensor(commBase, "GTensor", "GTensor", "GTensor", "GTensor");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> _redBaseGTensorFourierTransformedBackwards(commBase, "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads");
    
    // adjust their sizes
    _redBaseGTensor.adjustSize(GInd::getLatData().vol4);
    _redBaseGTensorFourierTransformedBackwards.adjustSize(GInd::getLatData().vol4);
    
    // create product out of two FFTed EMTs
    _redBaseGTensor.template iterateOverBulk<All, 0>(EMTtimesEMT<PREC>(_redBaseEMTUFourierTransformedBackwards.getAccessor(), _redBaseEMTUFourierTransformedForwards.getAccessor()));
    
    // FFT the product back
    fourierClass.performFourier3DTensor4x4Symx4x4SymComplex(_redBaseGTensor, _redBaseGTensorFourierTransformedBackwards, -1.0);
    
    // -----------------------------------------------------------------------
    // Third Step: Reduce 4x4x4x4 tensor field
    
    // define lattice container for tensor-reduced field
    LatticeContainer<true, COMPLEX(PREC)> _redBaseGLLDevice(commBase, "GLL", "GLL", "GLL", "GLL");
    LatticeContainer<false, COMPLEX(PREC)> _redBaseGLLHost(commBase, "GLL_Host", "GLL_Host", "GLL_Host", "GLL_Host");
    
    // adjust the sizes
    _redBaseGLLDevice.adjustSize(GInd::getLatData().vol4);
    _redBaseGLLHost.adjustSize(GInd::getLatData().vol4);
    
    // contract 4x4x4x4 tensor
    _redBaseGLLDevice.template iterateOverBulk<All, 0>(ContractGTensor<PREC>(_redBaseGTensorFourierTransformedBackwards.getAccessor()));
    
    // create host field from device field
    _redBaseGLLHost.copyFromLatticeContainer<true>(_redBaseGLLDevice);
    
    // reduce result
    COMPLEX(PREC) resultGLLAveragedDevice;
    COMPLEX(PREC) resultGLLAveragedHost;
    _redBaseGLLDevice.reduce(resultGLLAveragedDevice, GInd::getLatData().vol4);
    _redBaseGLLHost.reduce(resultGLLAveragedHost, GInd::getLatData().vol4);

    rootLogger.info("Result averaged G_LL (Device): ", resultGLLAveragedDevice);
    rootLogger.info("Result averaged G_LL (Host): ", resultGLLAveragedHost);
    
    // -----------------------------------------------------------------------
    // Fourth Step: Reduce field to array of radii
    
    LatticeContainerAccessor GLLAccessor(_redBaseGLLHost.getAccessor());
    
    int globLX = GInd::getLatData().globLX;
    int globLY = GInd::getLatData().globLY;
    int globLZ = GInd::getLatData().globLZ;
    int globLT = GInd::getLatData().globLT;
    
    int lx = GInd::getLatData().lx;
    int ly = GInd::getLatData().ly;
    int lz = GInd::getLatData().lz;
    int lt = GInd::getLatData().lt;
    
    int r2max = 0;
    r2max += globLX * globLX + globLY * globLY + globLZ * globLZ + globLT * globLT;
    r2max /= 4;
    
    rootLogger.info("r2max = ", r2max);
    
    COMPLEX(PREC) GLLarray[r2max+1] = {};
    int Counts[r2max+1] = {};
    
    for (int x = 0; x < lx; x++)
    for (int y = 0; y < ly; y++)
    for (int z = 0; z < lz; z++)
    for (int t = 0; t < lt; t++) {
        sitexyzt site(x, y, z, t);
        int r2 = GInd::getLatData().globalPosRelativeToOriginAbsoluteValueSquared(site); 
        if (x == 0 && y == 0 && t == 0) rootLogger.info("Site at x=", x, " y=", y, " z=", z, " t=", t, " with absolute value: ", r2);
        
        GLLarray[r2] += GLLAccessor.getElement<COMPLEX(PREC)>(GInd::getSite(x,y,z,t));
        Counts[r2] += 1;
    
    }

    rootLogger.info("For r2=1: GLL(r^2=1)=", GLLarray[2], " with ", Counts[2], " counts.");

    // -----------------------------------------------------------------------
    // Fifth Step: Write into file

    std::stringstream datNameGLL;
    datNameGLL << "/home/jwinter/Github/SIMULATeQCD_build/GLL.out";

    FileWriter file_GLL(commBase, param);
    file_GLL.createFile(datNameGLL.str());
    LineFormatter header_GLL = file_GLL.header();
    header_GLL << "r2" << "GLL.real" << "GLL.imag" << "Count";
    header_GLL.endLine();

    for (int r2 = 0; r2 < r2max + 1; r2++) {
        if (Counts[r2] != 0) {
            LineFormatter newLineGLL = file_GLL.tag("");
            newLineGLL << r2 << GLLarray[r2].cREAL << GLLarray[r2].cIMAG << Counts[r2];
        }
    }

    return 0;

}
