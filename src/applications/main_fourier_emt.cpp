#include "../simulateqcd.h"
#include "../experimental/fourierNon2.h"
#include "../base/math/matrix4x4SymComplex.h"
#include "../base/math/tensor4x4Symx4x4SymComplex.h"
#include "../modules/observables/energyMomentumTensor.h"
#include "../modules/observables/blocking.h"
#include "../testing/testing.h"


#define USE_GPU true
// define precision
#if SINGLEPREC
    #define PREC float
#else
    #define PREC double
#endif

enum Summation {
    Spatial,
    SpatialTemporal
};

enum Projector {
    SS, SL, LL, LT, TT
};

// define functor for combining two EMTs to one 4x4Symx4x4Sym
template<class floatT>
struct EMTtimesEMT {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    EMTtimesEMT(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) : _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> firstElement(_firstAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));
        Matrix4x4SymComplex<floatT> secondElement(_secondAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> result(firstElement, secondElement);

        return result;
    }

};

__device__ __host__ inline int getDimensionFunction(Summation summation) {
    switch (summation) {
        case Spatial:
            return 3;
            break;
        case SpatialTemporal:
            return 4;
            break;
        default:
            return 4;
    }
}

__device__ __host__ inline int indexMaxFunction(Summation summation) {
    switch (summation) {
        case Spatial:
            return 2;
            break;
        case SpatialTemporal:
            return 3;
            break;
        default:
            return 3;
    }
}

template<Summation summation>
__device__ __host__ inline int rSquared(sitexyzt r) {
    int indexMax = indexMaxFunction(summation);
    int r2 = 0;
    for (int i = 0; i <= indexMax; i++) {
        r2 += r[i] * r[i];
    }
    return r2;
}

__device__ __host__ inline int delta(int mu, int nu) {
    if (mu == nu) {
        return 1;
    } else {
        return 0;
    }
}

template<class floatT, Summation summation>
__device__ __host__ inline floatT deltaT(sitexyzt r, int mu, int nu) {
    int d = getDimensionFunction(summation);
    floatT r2 = rSquared<summation>(r);

    if (r2 == 0) {
        return (1.0-(1.0/d))*delta(mu,nu); // because r_mu*r_nu/r^2 approaches delta_munu/d at 0
    } else {
        return delta(mu, nu) - r[mu]*r[nu]/r2;
    }
}

template<class floatT, Summation summation>
__device__ __host__ inline floatT deltaHat(sitexyzt r, int mu, int nu) {
    int d = getDimensionFunction(summation);
    floatT r2 = rSquared<summation>(r);

    if (r2 == 0) {
        return 0.0; // because r_mu*r_nu/r^2 approaches delta_munu/d at 0
    } else {
        return delta(mu, nu) - d*r[mu]*r[nu]/r2;
    }
}

template<class floatT, Projector projector, Summation summation>
__device__ __host__ inline floatT projectorFunction(sitexyzt r, int mu, int nu, int rho, int sigma) {
    
    floatT result = 0.0;

    // add r-independent part
    // switch(projector) {
    //     case LL:
    //         result += (1.0/6.0)*delta(mu, nu)*delta(rho, sigma);
    //         break;
    //     // there is not r-independent part for LT
    //     case TT:
    //         result += (1.0/2.0) * (
    //             delta(mu, rho)*delta(nu, sigma)
    //             +delta(mu, sigma)*delta(nu, rho)
    //             -delta(mu, nu)*delta(rho, sigma)
    //         );
    //         break;
    // }

    int d = getDimensionFunction(summation);
    floatT r2 = rSquared<summation>(r);

    // if r!=0, add r-dependent part
    if (r2 != 0) {
        switch (projector) {
            case SS:
                result += (1.0/d) * delta(mu, nu) * delta(rho, sigma);
                break;
            case SL:
                result += (1.0/(d*sqrt(d-1))) * (
                    deltaHat<floatT, summation>(r, mu, nu) * delta(rho, sigma)
                    + delta(mu, nu) * deltaHat<floatT, summation>(r, rho, sigma)
                );
                break;
            case LL:
                result += (1.0/(d*(d-1))) * deltaHat<floatT, summation>(r, mu, nu) * deltaHat<floatT, summation>(r, rho, sigma);
                // result += (1.0/6.0)*delta(mu, nu)*delta(rho, sigma)
                //     -(1.0/2.0)*(
                //         r[mu]*r[nu]*delta(rho, sigma)
                //         +r[rho]*r[sigma]*delta(mu, nu)
                //     )/r2
                //     +(3.0/2.0)*(r[mu]*r[nu]*r[rho]*r[sigma])/(r2*r2);
                break;
            case LT:
                result += (
                    r[mu]*r[rho]*deltaT<floatT, summation>(r, nu, sigma)
                    +r[mu]*r[sigma]*deltaT<floatT, summation>(r, nu, rho)
                    +r[nu]*r[rho]*deltaT<floatT, summation>(r, mu, sigma)
                    +r[nu]*r[sigma]*deltaT<floatT, summation>(r, mu, rho)
                )/(2*r2);
                // result += (1.0/2.0) * (
                //     (
                //         r[mu]*r[rho]*delta(nu, sigma)
                //         +r[mu]*r[sigma]*delta(nu, rho)
                //         +r[nu]*r[rho]*delta(mu, sigma)
                //         +r[nu]*r[sigma]*delta(mu, rho)
                //     )/r2
                //     -4.0*(r[mu]*r[nu]*r[rho]*r[sigma])/(r2*r2)
                // );
                break;
            case TT:
                result += (1.0/2.0) * (
                    deltaT<floatT, summation>(r, mu, rho) * deltaT<floatT, summation>(r, nu, sigma)
                    +deltaT<floatT, summation>(r, mu, sigma) * deltaT<floatT, summation>(r, nu, rho)
                )
                - (1.0/(d-1)) * deltaT<floatT, summation>(r, mu, nu) * deltaT<floatT, summation>(r, rho, sigma);
                // result += (1.0/2.0) * (
                //         delta(mu, rho)*delta(nu, sigma)
                //         +delta(mu, sigma)*delta(nu, rho)
                //         -delta(mu, nu)*delta(rho, sigma)
                //     )
                //     +(1.0/2.0) * (
                //     (
                //         r[mu]*r[nu]*delta(rho, sigma)
                //         +r[rho]*r[sigma]*delta(mu, nu)
                //     )/r2
                //     -(
                //         r[mu]*r[rho]*delta(nu, sigma)
                //         +r[mu]*r[sigma]*delta(nu, rho)
                //         +r[nu]*r[rho]*delta(mu, sigma)
                //         +r[nu]*r[sigma]*delta(mu, rho)
                //     )/r2
                //     +(r[mu]*r[nu]*r[rho]*r[sigma])/(r2*r2)
                // );
                break;
        }
    } else {
        switch (projector) {
            case SS:
                result += (1.0/d) * delta(mu, nu) * delta(rho, sigma);
                break;
            case SL:
                result += 0.0;
                break;
            case LL:
                result += (1.0/((d+2)*(d-1))) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/d) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case LT:
                result += (1.0/(d+2)) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/d) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case TT:
                result += ((floatT) (d+1)*(d-2)/(2*(d+2)*(d-1))) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/d) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
        }
    }

    return result;

}

template<class floatT, Summation summation>
struct ProjectorSumLHS {

    typedef GIndexer<All> GInd;

    ProjectorSumLHS() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);

        // create object for projector sum
        Tensor4x4Symx4x4SymComplex<floatT> projectorSumAtSite;

        // fill projector sum
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = 0.0;
            value += projectorFunction<floatT, SS, summation>(r, mu, nu, rho, sigma);
            // value += projectorFunction<floatT, SL, summation>(r, mu, nu, rho, sigma); // not a true projector
            value += projectorFunction<floatT, LL, summation>(r, mu, nu, rho, sigma);
            value += projectorFunction<floatT, LT, summation>(r, mu, nu, rho, sigma);
            value += projectorFunction<floatT, TT, summation>(r, mu, nu, rho, sigma);
            projectorSumAtSite(mu, nu, rho, sigma, value);
        }

        return projectorSumAtSite;

    }

};

template<class floatT, Summation summation>
struct ProjectorSumRHS {

    typedef GIndexer<All> GInd;

    ProjectorSumRHS() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);

        // create object for projector sum RHS
        Tensor4x4Symx4x4SymComplex<floatT> projectorSumAtSiteRHS;

        // fill projector sum RHS
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = (1.0/2.0)*(delta(mu, rho)*delta(nu, sigma) + delta(mu, sigma)*delta(nu, rho)); // - (2.0/3.0)*delta(mu, nu)*delta(rho, sigma)
            projectorSumAtSiteRHS(mu, nu, rho, sigma, value);
        }

        return projectorSumAtSiteRHS;
        
    }

};

template<class floatT, Summation summation>
struct ProjectorProduct {
    
    typedef GIndexer<All> GInd;
    LatticeContainerAccessor projectorXAccessor;
    LatticeContainerAccessor projectorYAccessor;

    ProjectorProduct(LatticeContainerAccessor _projectorXAccessor, LatticeContainerAccessor _projectorYAccessor) : projectorXAccessor(_projectorXAccessor), projectorYAccessor(_projectorYAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Tensor4x4Symx4x4SymComplex<floatT> projectorXAtSite = projectorXAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);
        Tensor4x4Symx4x4SymComplex<floatT> projectorYAtSite = projectorYAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        Tensor4x4Symx4x4SymComplex<floatT> projectorProductAtSite = 0.0;

        // for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        //     for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
        //         for (int summationIndexPair = 0; summationIndexPair < 10; summationIndexPair++) {
        //             COMPLEX(floatT) value = projectorProductAtSite(firstIndexPair, secondIndexPair);
        //             value += projectorXAtSite(firstIndexPair, summationIndexPair) * projectorYAtSite(summationIndexPair, secondIndexPair);
        //             projectorProductAtSite(firstIndexPair, secondIndexPair, value);
        //         }
        //     }
        // }

        int indexMax = indexMaxFunction(summation);

        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int alpha = 0; alpha <= indexMax; alpha++)
        for (int beta = 0; beta <= indexMax; beta++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = projectorProductAtSite(mu, nu, rho, sigma);
            value += projectorXAtSite(mu, nu, alpha, beta) * projectorYAtSite(alpha, beta, rho, sigma);
            projectorProductAtSite(mu, nu, rho, sigma, value);
        }

        return projectorProductAtSite;
    }
};

template<class floatT, Summation summation>
struct ProjectorProductSymmetrized {
    
    typedef GIndexer<All> GInd;
    LatticeContainerAccessor projectorXAccessor;
    LatticeContainerAccessor projectorYAccessor;

    ProjectorProductSymmetrized(LatticeContainerAccessor _projectorXAccessor, LatticeContainerAccessor _projectorYAccessor) : projectorXAccessor(_projectorXAccessor), projectorYAccessor(_projectorYAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Tensor4x4Symx4x4SymComplex<floatT> projectorXAtSite = projectorXAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);
        Tensor4x4Symx4x4SymComplex<floatT> projectorYAtSite = projectorYAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        Tensor4x4Symx4x4SymComplex<floatT> projectorProductAtSite = 0.0;

        // for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        //     for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
        //         for (int summationIndexPair = 0; summationIndexPair < 10; summationIndexPair++) {
        //             COMPLEX(floatT) value = projectorProductAtSite(firstIndexPair, secondIndexPair);
        //             value += projectorXAtSite(firstIndexPair, summationIndexPair) * projectorYAtSite(summationIndexPair, secondIndexPair);
        //             projectorProductAtSite(firstIndexPair, secondIndexPair, value);
        //         }
        //     }
        // }

        int indexMax = indexMaxFunction(summation);

        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int alpha = 0; alpha <= indexMax; alpha++)
        for (int beta = 0; beta <= indexMax; beta++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = projectorProductAtSite(mu, nu, rho, sigma);
            value += projectorXAtSite(mu, nu, alpha, beta) * projectorYAtSite(alpha, beta, rho, sigma);
            value += projectorYAtSite(mu, nu, alpha, beta) * projectorXAtSite(alpha, beta, rho, sigma);
            // value /= 2.0;
            projectorProductAtSite(mu, nu, rho, sigma, value);
        }

        return projectorProductAtSite;
    }
};

template<class floatT, Summation summation>
struct ProjectorProductRHSLLLTTT {

    typedef GIndexer<All> GInd;

    ProjectorProductRHSLLLTTT() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);
        int d = getDimensionFunction(summation);
        floatT r2 = rSquared<summation>(r);

        // create object for projector sum RHS
        Tensor4x4Symx4x4SymComplex<floatT> projectorProductRHSLLLTTTAtSite;

        if (r2 == 0) {
            // fill projector RHS
            for (int mu = 0; mu <= indexMax; mu++)
            for (int nu = 0; nu <= mu; nu++)
            for (int rho = 0; rho <= indexMax; rho++)
            for (int sigma = 0; sigma <= rho; sigma++) {
                COMPLEX(floatT) value = delta(mu, rho)*delta(nu, sigma) + delta(mu, sigma)*delta(nu, rho) - (2.0/d)*delta(mu, nu)*delta(rho, sigma);
                projectorProductRHSLLLTTTAtSite(mu, nu, rho, sigma, value);
            }
        }

        return projectorProductRHSLLLTTTAtSite;
        
    }

};

template<class floatT, Summation summation>
struct ProjectorProductRHSSLSL {

    typedef GIndexer<All> GInd;

    ProjectorProductRHSSLSL() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);
        floatT r2 = rSquared<summation>(r);

        // create object for projector sum RHS
        Tensor4x4Symx4x4SymComplex<floatT> projectorProductRHSSLSLAtSite;

        if (r2 == 0) return 0.0;

        // fill projector RHS
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = 0.0;
            value += projectorFunction<floatT, SS, summation>(r, mu, nu, rho, sigma);
            value += projectorFunction<floatT, LL, summation>(r, mu, nu, rho, sigma);
            projectorProductRHSSLSLAtSite(mu, nu, rho, sigma, value);
        }

        return projectorProductRHSSLSLAtSite;
        
    }

};

template<class floatT, Projector projector, Summation summation>
struct ProjectorField {
    
    typedef GIndexer<All> GInd;

    ProjectorField() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {
        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);

        // get desired projector value for the global position
        Tensor4x4Symx4x4SymComplex<floatT> projectorAtSite;
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = 0.0;
            value = projectorFunction<floatT, projector, summation>(r, mu, nu, rho, sigma);
            projectorAtSite(mu, nu, rho, sigma, value);
        }
        return projectorAtSite;
    }
};

// template<class floatT>
// __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> projectorLL(sitexyzt r) {

//     Tensor4x4Symx4x4SymComplex<floatT> projector;

//     for (int mu = 0; mu <= 3; mu++)
//     for (int nu = 0; nu <= mu; nu++)
//     for (int rho = 0; rho <= 3; rho++)
//     for (int sigma = 0; sigma <= rho; sigma++) {
//         COMPLEX(floatT) value = projectorLLFunction<floatT>(r, mu, nu, rho, sigma);
//         projector(mu, nu, rho, sigma, value);
//     }

//     return projector;
// }

// define functor to contract tensor indices
template<class floatT, Projector projector, Summation summation>
struct ContractGTensor {

    LatticeContainerAccessor _GAccessor;
    typedef GIndexer<All> GInd;

    ContractGTensor(LatticeContainerAccessor GAccessor) : _GAccessor(GAccessor) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);

        // get desired projector value for the global position
        Tensor4x4Symx4x4SymComplex<floatT> projectorAtSite;

        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = projectorFunction<floatT, LL, summation>(r, mu, nu, rho, sigma);
            projectorAtSite(mu, nu, rho, sigma, value);
        }

        // get correlator value at the site
        Tensor4x4Symx4x4SymComplex<floatT> tensor4x4Symx4x4SymComplexAtSite = _GAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        // contract projector value with correlator value
        COMPLEX(floatT) result = 0.0;
        
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            result += projectorAtSite(mu, nu, rho, sigma) * tensor4x4Symx4x4SymComplexAtSite(mu, nu, rho, sigma);
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

    // -----------------------------------------------------------------------
    // Second Step: Combine Fourier-Transformed EMT Fields
    
    // create lattice containers for EMT fields
    LatticeContainer<false, Matrix4x4SymComplex<PREC>> EMTUComplexHost(commBase, "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST", "EMTU_COMPLEX_HOST");
    LatticeContainer<true, Matrix4x4SymComplex<PREC>> EMTUComplexDevice(commBase, "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE", "EMTU_COMPLEX_DEVICE");
    LatticeContainer<true, Matrix4x4SymComplex<PREC>> EMTUFourierTransformedForwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS");
    LatticeContainer<true, Matrix4x4SymComplex<PREC>> EMTUFourierTransformedBackwards(commBase, "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_BACKWARDS");
    LatticeContainer<true, Matrix4x4SymComplex<PREC>> EMTUFourierTransformedForwardsBackwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS");
    
    // adjust their size
    EMTUComplexHost.adjustSize(GInd::getLatData().vol4);
    EMTUComplexDevice.adjustSize(GInd::getLatData().vol4);
    EMTUFourierTransformedForwards.adjustSize(GInd::getLatData().vol4);
    EMTUFourierTransformedBackwards.adjustSize(GInd::getLatData().vol4);
    EMTUFourierTransformedForwardsBackwards.adjustSize(GInd::getLatData().vol4);
    
    // calculate EMT on one of them
    EMTUComplexHost.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, false, HaloDepth>(gaugeHost.getAccessor()));
    EMTUComplexDevice.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, true, HaloDepth>(gaugeDevice.getAccessor()));
    // EMTUFourier.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplexZero<PREC, true, HaloDepth>(gauge.getAccessor()));
    // EMTUFourierTransformedForwardsBackwards.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplexZero<PREC, true, HaloDepth>(gauge.getAccessor()));
    
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
    // LatticeContainer<true, COMPLEX(PREC)> Device(commBase , "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device");
    // LatticeContainer<false, COMPLEX(PREC)> Host(commBase , "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host");

    // perform Fourier transformation forwards
    fourierClass.performFourier3DEMT(EMTUComplexDevice, EMTUFourierTransformedForwards, 1.0);
    // perform Fourier transformation forwards
    fourierClass.performFourier3DEMT(EMTUComplexDevice, EMTUFourierTransformedBackwards, -1.0);
    // perform Fourier transformation backwards after the forwards
    fourierClass.performFourier3DEMT(EMTUFourierTransformedForwards, EMTUFourierTransformedForwardsBackwards, -1.0);

    EMTUComplexDevice.reduce(resultEMTUComplexDevice, GInd::getLatData().vol4);
    EMTUComplexHost.reduce(resultEMTUComplexHost, GInd::getLatData().vol4);
    EMTUFourierTransformedForwards.reduce(resultEMTUFourierTransformedForwards, GInd::getLatData().vol4);
    EMTUFourierTransformedForwardsBackwards.reduce(resultEMTUFourierTransformedForwardsBackwards, GInd::getLatData().vol4);
    
    LatticeContainerAccessor EMTUHostAccessor(EMTUComplexHost.getAccessor());

    Matrix4x4SymComplex<PREC> complexAtZero = EMTUHostAccessor.getElement<Matrix4x4SymComplex<PREC>>(GInd::getSite(0,0,0,0));

    // resize value by sqrt(V_4)
    complexAtZero *= sqrt(GInd::getLatData().vol4);

    // compare T_munu at r=0 and integrated FFT(T_munu) over all p
    rootLogger.info("Test FFT via ∫dp T_munu(p) = T_munu(r=0):");
    compare_elementwise_prec(complexAtZero, resultEMTUFourierTransformedForwards, 1e-12, 1e-12, "Comparison of T_munu(r=0) and integrated T_munu(p) elementwise");
    
    // compare integrated T_munu(r) with integrated FFT^{-1}(FFT(T_munu))(r)
    rootLogger.info("Test FFT invertibility:");
    compare_lattice_containers_elementwise_prec<true, Matrix4x4SymComplex<PREC>>(EMTUComplexDevice, EMTUFourierTransformedForwardsBackwards, 1e-14, "Comparison of T_munu(r) and FFT^{-1}(FFT(T_munu))(r) site by site");

    // -----------------------------------------------------------------------
    // Second Step: Combine Fourier-Transformed EMT Fields
    
    // define lattice containers for products
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> GTensor(commBase, "GTensor", "GTensor", "GTensor", "GTensor");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> GTensorFourierTransformedBackwards(commBase, "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads", "GTensorFourierTransformedBackwads");
    
    // adjust their sizes
    GTensor.adjustSize(GInd::getLatData().vol4);
    GTensorFourierTransformedBackwards.adjustSize(GInd::getLatData().vol4);
    
    // create product out of two FFTed EMTs
    GTensor.template iterateOverBulk<All, 0>(EMTtimesEMT<PREC>(EMTUFourierTransformedBackwards.getAccessor(), EMTUFourierTransformedForwards.getAccessor()));
    
    // FFT the product back
    fourierClass.performFourier3DTensor4x4Symx4x4SymComplex(GTensor, GTensorFourierTransformedBackwards, -1.0);
    
    // -----------------------------------------------------------------------
    // Third Step: Reduce 4x4x4x4 tensor field
    
    // define lattice container for tensor-reduced field
    LatticeContainer<true, COMPLEX(PREC)> GLLDevice(commBase, "GLL", "GLL", "GLL", "GLL");
    LatticeContainer<false, COMPLEX(PREC)> GLLHost(commBase, "GLL_Host", "GLL_Host", "GLL_Host", "GLL_Host");
    
    // adjust the sizes
    GLLDevice.adjustSize(GInd::getLatData().vol4);
    GLLHost.adjustSize(GInd::getLatData().vol4);
    
    // contract 4x4x4x4 tensor
    GLLDevice.template iterateOverBulk<All, 0>(ContractGTensor<PREC, LL, Spatial>(GTensorFourierTransformedBackwards.getAccessor()));
    
    // create host field from device field
    GLLHost.copyFromLatticeContainer<true>(GLLDevice);
    
    // test projector sum
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> projectorSumLHS(commBase, "projectorSumLHS", "projectorSumLHS", "projectorSumLHS", "projectorSumLHS");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> projectorSumRHS(commBase, "projectorSumRHS", "projectorSumRHS", "projectorSumRHS", "projectorSumRHS");
    
    projectorSumLHS.adjustSize(GInd::getLatData().vol4);
    projectorSumRHS.adjustSize(GInd::getLatData().vol4);
    
    projectorSumLHS.template iterateOverBulk<All, 0>(ProjectorSumLHS<PREC, Spatial>());
    projectorSumRHS.template iterateOverBulk<All, 0>(ProjectorSumRHS<PREC, Spatial>());
    
    rootLogger.info("Test sum of projectors P_SS, P_LL, P_LT, P_TT:");
    compare_lattice_containers_elementwise_prec(projectorSumLHS, projectorSumRHS, 1e-12, "Comparison of projector sum P_SS+P_LL+P_LT+P_TT to expected RHS.");

    // test projector orthonormality
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorFieldSS(commBase, "projectorFieldSS", "projectorFieldSS", "projectorFieldSS", "projectorFieldSS");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorFieldSL(commBase, "projectorFieldSL", "projectorFieldSL", "projectorFieldSL", "projectorFieldSL");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorFieldLL(commBase, "projectorFieldLL", "projectorFieldLL", "projectorFieldLL", "projectorFieldLL");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorFieldLT(commBase, "projectorFieldLT", "projectorFieldLT", "projectorFieldLT", "projectorFieldLT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorFieldTT(commBase, "projectorFieldTT", "projectorFieldTT", "projectorFieldTT", "projectorFieldTT");
    projectorFieldSS.adjustSize(GInd::getLatData().vol4);
    projectorFieldSL.adjustSize(GInd::getLatData().vol4);
    projectorFieldLL.adjustSize(GInd::getLatData().vol4);
    projectorFieldLT.adjustSize(GInd::getLatData().vol4);
    projectorFieldTT.adjustSize(GInd::getLatData().vol4);
    projectorFieldSS.template iterateOverBulk<All, 0>(ProjectorField<PREC, SS, Spatial>());
    projectorFieldSL.template iterateOverBulk<All, 0>(ProjectorField<PREC, SL, Spatial>());
    projectorFieldLL.template iterateOverBulk<All, 0>(ProjectorField<PREC, LL, Spatial>());
    projectorFieldLT.template iterateOverBulk<All, 0>(ProjectorField<PREC, LT, Spatial>());
    projectorFieldTT.template iterateOverBulk<All, 0>(ProjectorField<PREC, TT, Spatial>());
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSSSS(commBase, "projectorProductSSSS", "projectorProductSSSS", "projectorProductSSSS", "projectorProductSSSS");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLLLL(commBase, "projectorProductLLLL", "projectorProductLLLL", "projectorProductLLLL", "projectorProductLLLL");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLTLT(commBase, "projectorProductLTLT", "projectorProductLTLT", "projectorProductLTLT", "projectorProductLTLT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductTTTT(commBase, "projectorProductTTTT", "projectorProductTTTT", "projectorProductTTTT", "projectorProductTTTT");
    projectorProductSSSS.adjustSize(GInd::getLatData().vol4);
    projectorProductLLLL.adjustSize(GInd::getLatData().vol4);
    projectorProductLTLT.adjustSize(GInd::getLatData().vol4);
    projectorProductTTTT.adjustSize(GInd::getLatData().vol4);
    projectorProductSSSS.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSS.getAccessor(), projectorFieldSS.getAccessor()));
    projectorProductLLLL.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldLL.getAccessor(), projectorFieldLL.getAccessor()));
    projectorProductLTLT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldLT.getAccessor(), projectorFieldLT.getAccessor()));
    projectorProductTTTT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldTT.getAccessor(), projectorFieldTT.getAccessor()));
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSSLL(commBase, "projectorProductSSLL", "projectorProductSSLL", "projectorProductSSLL", "projectorProductSSLL");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSSLT(commBase, "projectorProductSSLT", "projectorProductSSLT", "projectorProductSSLT", "projectorProductSSLT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSSTT(commBase, "projectorProductSSTT", "projectorProductSSTT", "projectorProductSSTT", "projectorProductSSTT");
    projectorProductSSLL.adjustSize(GInd::getLatData().vol4);
    projectorProductSSLT.adjustSize(GInd::getLatData().vol4);
    projectorProductSSTT.adjustSize(GInd::getLatData().vol4);
    projectorProductSSLL.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSS.getAccessor(), projectorFieldLL.getAccessor()));
    projectorProductSSLT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSS.getAccessor(), projectorFieldLT.getAccessor()));
    projectorProductSSTT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSS.getAccessor(), projectorFieldTT.getAccessor()));
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLLLT(commBase, "projectorProductLLLT", "projectorProductLLLT", "projectorProductLLLT", "projectorProductLLLT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLLTT(commBase, "projectorProductLLTT", "projectorProductLLTT", "projectorProductLLTT", "projectorProductLLTT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLTTT(commBase, "projectorProductLTTT", "projectorProductLTTT", "projectorProductLTTT", "projectorProductLTTT");
    projectorProductLLLT.adjustSize(GInd::getLatData().vol4);
    projectorProductLLTT.adjustSize(GInd::getLatData().vol4);
    projectorProductLTTT.adjustSize(GInd::getLatData().vol4);
    projectorProductLLLT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldLL.getAccessor(), projectorFieldLT.getAccessor()));
    projectorProductLLTT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldLL.getAccessor(), projectorFieldTT.getAccessor()));
    projectorProductLTTT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldLT.getAccessor(), projectorFieldTT.getAccessor()));
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSLSL(commBase, "projectorProductSLSL", "projectorProductSLSL", "projectorProductSLSL", "projectorProductSLSL");
    projectorProductSLSL.adjustSize(GInd::getLatData().vol4);
    projectorProductSLSL.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSL.getAccessor(), projectorFieldSL.getAccessor()));
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSSSLSym(commBase, "projectorProductSSSLSym", "projectorProductSSSLSym", "projectorProductSSSLSym", "projectorProductSSSLSym");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductLLSLSym(commBase, "projectorProductLLSLSym", "projectorProductLLSLSym", "projectorProductLLSLSym", "projectorProductLLSLSym");
    projectorProductSSSLSym.adjustSize(GInd::getLatData().vol4);
    projectorProductLLSLSym.adjustSize(GInd::getLatData().vol4);
    projectorProductSSSLSym.template iterateOverBulk<All, 0>(ProjectorProductSymmetrized<PREC, Spatial>(projectorFieldSS.getAccessor(), projectorFieldSL.getAccessor()));
    projectorProductLLSLSym.template iterateOverBulk<All, 0>(ProjectorProductSymmetrized<PREC, Spatial>(projectorFieldLL.getAccessor(), projectorFieldSL.getAccessor()));
    
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSLLT(commBase, "projectorProductSLLT", "projectorProductSLLT", "projectorProductSLLT", "projectorProductSLLT");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductSLTT(commBase, "projectorProductSLTT", "projectorProductSLTT", "projectorProductSLTT", "projectorProductSLTT");
    projectorProductSLLT.adjustSize(GInd::getLatData().vol4);
    projectorProductSLTT.adjustSize(GInd::getLatData().vol4);
    projectorProductSLLT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSL.getAccessor(), projectorFieldLT.getAccessor()));
    projectorProductSLTT.template iterateOverBulk<All, 0>(ProjectorProduct<PREC, Spatial>(projectorFieldSL.getAccessor(), projectorFieldTT.getAccessor()));
    
    // adjust values of P_X@P_X and P_{LL,LT,TT}@P_{LL,LT,TT} at x=y=z=0
    Tensor4x4Symx4x4SymComplex<PREC> projectorFieldValueAtTimeSlice;
    int d = getDimensionFunction(Spatial);
    for (int t = 0; t < GInd::getLatData().lt; t++) {
        projectorFieldValueAtTimeSlice = projectorProductLLLL.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (2.0/((d+2)*(d-1)));
        projectorProductLLLL.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
        projectorFieldValueAtTimeSlice = projectorProductLTLT.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (2.0/(d+2));
        projectorProductLTLT.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
        projectorFieldValueAtTimeSlice = projectorProductTTTT.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (1.0*(d+1)*(d-2)/((d+2)*(d-1)));
        projectorProductTTTT.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
        projectorFieldValueAtTimeSlice = projectorProductLLLT.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (2.0/((d+2)*(d+2)*(d-1)));
        projectorProductLLLT.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
        projectorFieldValueAtTimeSlice = projectorProductLLTT.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (1.0*(d+1)*(d-2)/((d+2)*(d+2)*(d-1)*(d-1)));
        projectorProductLLTT.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
        projectorFieldValueAtTimeSlice = projectorProductLTTT.getAccessor().getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(0, 0, 0, t));
        projectorFieldValueAtTimeSlice /= (1.0*(d+1)*(d-2)/((d+2)*(d+2)*(d-1)));
        projectorProductLTTT.getAccessor().setElement(GInd::getSite(0, 0, 0, t), projectorFieldValueAtTimeSlice);
        
    }
    
    rootLogger.info("Test normality of projectors P_SS, P_LL, P_LT, P_TT:");
    compare_lattice_containers_elementwise_prec(projectorProductSSSS, projectorFieldSS, 1e-12, "P_SS@P_SS = P_SS.");
    compare_lattice_containers_elementwise_prec(projectorProductLLLL, projectorFieldLL, 1e-12, "P_LL@P_LL = P_LL.");
    compare_lattice_containers_elementwise_prec(projectorProductLTLT, projectorFieldLT, 1e-12, "P_LT@P_LT = P_LT.");
    compare_lattice_containers_elementwise_prec(projectorProductTTTT, projectorFieldTT, 1e-12, "P_TT@P_TT = P_TT.");
    
    rootLogger.info("Test full orthogonality of projector P_SS with projectors P_LL, P_LT, P_TT:");
    Tensor4x4Symx4x4SymComplex<PREC> zeroTensor = 0.0;
    compare_lattice_container_elementwise_prec_to_value(projectorProductSSLL, zeroTensor, 1e-12, "P_SS@P_LL = 0.");
    compare_lattice_container_elementwise_prec_to_value(projectorProductSSLT, zeroTensor, 1e-12, "P_SS@P_LT = 0.");
    compare_lattice_container_elementwise_prec_to_value(projectorProductSSTT, zeroTensor, 1e-12, "P_SS@P_TT = 0.");
    
    rootLogger.info("Test orthogonality of projectors P_LL, P_LT, P_TT pairwise except at r=0:");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductRHSLLLTTT(commBase, "projectorProductRHSLLLTTT", "projectorProductRHSLLLTTT", "projectorProductRHSLLLTTT", "projectorProductRHSLLLTTT");
    projectorProductRHSLLLTTT.adjustSize(GInd::getLatData().vol4);
    projectorProductRHSLLLTTT.template iterateOverBulk<All, 0>(ProjectorProductRHSLLLTTT<PREC, Spatial>());
    compare_lattice_containers_elementwise_prec(projectorProductLLLT, projectorProductRHSLLLTTT, 1e-12, "P_LL@P_LT = 0 expect at r=0.");
    compare_lattice_containers_elementwise_prec(projectorProductLLTT, projectorProductRHSLLLTTT, 1e-12, "P_LL@P_TT = 0 expect at r=0.");
    compare_lattice_containers_elementwise_prec(projectorProductLTTT, projectorProductRHSLLLTTT, 1e-12, "P_LT@P_TT = 0 expect at r=0.");
    
    rootLogger.info("Test projector P_SL relation: P_SL@P_SL = P_SS + P_LL except at r=0:");
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> projectorProductRHSSLSL(commBase, "projectorProductRHSSLSL", "projectorProductRHSSLSL", "projectorProductRHSSLSL", "projectorProductRHSSLSL");
    projectorProductRHSSLSL.adjustSize(GInd::getLatData().vol4);
    projectorProductRHSSLSL.template iterateOverBulk<All, 0>(ProjectorProductRHSSLSL<PREC, Spatial>());
    compare_lattice_containers_elementwise_prec(projectorProductSLSL, projectorProductRHSSLSL, 1e-12, "P_SL@P_SL = P_SS + P_LL.");
    
    rootLogger.info("Test projector P_SL relation: P_SS@P_SL + P_SL@P_SS = P_SL (same for P_LL) except at r=0:");
    compare_lattice_containers_elementwise_prec(projectorProductSSSLSym, projectorFieldSL, 1e-12, "P_SS@P_SL + P_SL@P_SS = P_SL.");
    compare_lattice_containers_elementwise_prec(projectorProductLLSLSym, projectorFieldSL, 1e-12, "P_LL@P_SL + P_SL@P_LL = P_SL.");

    rootLogger.info("Test full orthogonality of projector P_SL with projectors P_LT, P_TT:");
    compare_lattice_container_elementwise_prec_to_value(projectorProductSSLT, zeroTensor, 1e-12, "P_SL@P_LT = 0.");
    compare_lattice_container_elementwise_prec_to_value(projectorProductSSTT, zeroTensor, 1e-12, "P_SL@P_TT = 0.");
    
    // -----------------------------------------------------------------------
    // Fourth Step: Reduce field to array of radii
    
    LatticeContainerAccessor GLLAccessor(GLLHost.getAccessor());
    
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
        // if (x == 0 && y == 0 && t == 1) {
        //     sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site);
        //     rootLogger.info("Site ", site, " with global position relative to the origin r=", r, " with absolute value: ", r2);

        //     if (z == 3) {
        //         for (int mu = 0; mu <= 3; mu++)
        //         for (int nu = 0; nu <= mu; nu++)
        //         for (int rho = 0; rho <= 3; rho++)
        //         for (int sigma = 0; sigma <= rho; sigma++) {
        //             PREC projectorLL = projectorFunction<PREC, LL, Spatial>(r, mu, nu, rho, sigma);
        //             PREC projectorLT = projectorFunction<PREC, LT, Spatial>(r, mu, nu, rho, sigma);
        //             PREC projectorTT = projectorFunction<PREC, TT, Spatial>(r, mu, nu, rho, sigma);
        //             int d = getDimensionFunction(Spatial);
        //             PREC sum = projectorLL + projectorLT + projectorTT;
        //             PREC rhs = (1.0/2.0)*(delta(mu, rho)*delta(nu, sigma) + delta(mu, sigma)*delta(nu, rho) - (2.0/d)*delta(mu, nu)*delta(rho, sigma));
        //             rootLogger.info("Sum of projectors at ", mu, nu, rho, sigma, ": ", sum, " = ", rhs);
        //             rootLogger.info("P_LL=", projectorLL, " P_LT=", projectorLT, " P_TT=", projectorTT);
        //         }
        //     }
        // }
        
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
