//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"

enum class Projector {
    TT, LT,
    T, L,
    SS, LL, WW,
    SL, SW, LW
};

enum class HalfProjector {
    S, L, W
};

constexpr std::array<Projector, 10> allProjectors{{
    Projector::TT, Projector::LT,
    Projector::T, Projector::L,
    Projector::SS, Projector::LL, Projector::WW,
    Projector::SL, Projector::SW, Projector::LW
}};


__device__ __host__ inline int multiplicity(Projector projector) {
    switch (projector) {
        case Projector::TT:
            return 2;
            break;
        case Projector::LT:
            return 2;
            break;
        case Projector::T:
            return 2;
            break;
        case Projector::L:
            return 1;
            break;
        case Projector::SS:
            return 1;
            break;
        case Projector::LL:
            return 1;
            break;
        case Projector::WW:
            return 1;
            break;
        case Projector::SL:
            return 2;
            break;
        case Projector::SW:
            return 2;
            break;
        case Projector::LW:
            return 2;
            break;
        default:
            return 1;
    }
}

__device__ __host__ inline int indexMaxFunction(SpatialTemporal spatialTemporal) {
    switch (spatialTemporal) {
        case SpatialTemporal::Spatial:
            return 2;
            break;
        case SpatialTemporal::Temporal:
            return 0;
            break;
        case SpatialTemporal::Both:
            return 3;
            break;
        default:
            return 3;
    }
}

__device__ __host__ inline int getDimensionFunction(SpatialTemporal spatialTemporal) {
    switch (spatialTemporal) {
        case SpatialTemporal::Spatial:
            return 3;
            break;
        case SpatialTemporal::Temporal:
            return 1;
            break;
        case SpatialTemporal::Both:
            return 4;
            break;
        default:
            return 4;
    }
}

// helper function: compute spatial r^2=x^2+y^2+z^2 for a site (x,y,z,t)
__device__ __host__ inline int rSquared(sitexyzt rt) {
    int r2 = 0;
    for (int i = 0; i <= 2; i++) {
        r2 += rt[i] * rt[i];
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

__device__ __host__ inline int deltaSpatial(int mu, int nu) {
    sitexyzt u = {0,0,0,1};
    return delta(mu, nu) - u[mu]*u[nu];
}

// helper function: purely-spatial transversal delta function
template<class floatT>
__device__ __host__ inline floatT deltaTransverse(sitexyzt r, int mu, int nu) {
    floatT r2 = rSquared(r);
    int D = 3;

    if (r2 == 0) {
        return deltaSpatial(mu,nu) - delta(mu, nu)/((floatT) D); // because r_mu*r_nu/r^2 approaches delta_munu/d at 0
    } else {
        return deltaSpatial(mu, nu) - r[mu]*r[nu]/r2;
    }
}

template<class floatT>
__device__ __host__ inline floatT deltaHat(sitexyzt r, int mu, int nu) {
    int r2 = rSquared(r);
    return r[mu]*r[nu]/((floatT) r2) - (1.0/3.0) * deltaSpatial(mu, nu);
}

template<class floatT, HalfProjector halfProjector>
__device__ __host__ inline floatT C(sitexyzt rt, int mu, int nu) {
    // only spatial part
    sitexyzt r = {rt[0], rt[1], rt[2], 0};

    sitexyzt u = {0,0,0,1};

    floatT result = 0.0;

    switch (halfProjector) {
        case HalfProjector::S:
            result += (1.0/2.0) * delta(mu, nu);
            break;
        case HalfProjector::L:
            result += sqrt(3.0/2.0) * deltaHat<floatT>(r, mu, nu);
            break;
        case HalfProjector::W:
            result += sqrt(1.0/12.0) * (deltaSpatial(mu, nu) - 3.0 * u[mu] * u[nu]);
            break;
    }

    return result;
}

template<class floatT, Projector projector>
__device__ __host__ inline floatT projectorFunction(sitexyzt rt, int mu, int nu, int rho, int sigma) {
    // only spatial part
    sitexyzt r = {rt[0], rt[1], rt[2], 0};

    sitexyzt u = {0,0,0,1};

    // spatial dimension
    int Ds = 3;
    // calculate r^2 beforehand
    floatT r2 = rSquared(r);
    
    floatT result = 0.0;

    // if r!=0, add r-dependent part
    if (r2 != 0) {
        switch (projector) {
            case Projector::TT:
                result += (1.0/2.0) * (
                    deltaTransverse<floatT>(r, mu, rho) * deltaTransverse<floatT>(r, nu, sigma)
                    + deltaTransverse<floatT>(r, mu, sigma) * deltaTransverse<floatT>(r, nu, rho)
                )
                - (1.0/(Ds-1)) * deltaTransverse<floatT>(r, mu, nu) * deltaTransverse<floatT>(r, rho, sigma);
                break;
            case Projector::LT:
                result += (
                    r[mu]*r[rho]*deltaTransverse<floatT>(r, nu, sigma)
                    + r[mu]*r[sigma]*deltaTransverse<floatT>(r, nu, rho)
                    + r[nu]*r[rho]*deltaTransverse<floatT>(r, mu, sigma)
                    + r[nu]*r[sigma]*deltaTransverse<floatT>(r, mu, rho)
                )/(2.0*r2);
                break;
            case Projector::T:
                result += (
                    deltaTransverse<floatT>(r, mu, rho) * u[nu] * u[sigma]
                    + deltaTransverse<floatT>(r, mu, sigma) * u[nu] * u[rho]
                    + deltaTransverse<floatT>(r, nu, rho) * u[mu] * u[sigma]
                    + deltaTransverse<floatT>(r, nu, sigma) * u[mu] * u[rho]
                ) / 2.0;
                break;
            case Projector::L:
                result += (
                    (r[mu] * u[nu] + r[nu] * u[mu]) * (r[rho] * u[sigma] + r[sigma] * u[rho])
                ) / (2.0 * r2);
                break;
            case Projector::SS:
                result += C<floatT, HalfProjector::S>(r, mu, nu) * C<floatT, HalfProjector::S>(r, rho, sigma);
                break;
            case Projector::LL:
                result += C<floatT, HalfProjector::L>(r, mu, nu) * C<floatT, HalfProjector::L>(r, rho, sigma);
                break;
            case Projector::WW:
                result += C<floatT, HalfProjector::W>(r, mu, nu) * C<floatT, HalfProjector::W>(r, rho, sigma);
                break;
            case Projector::SL:
                result += (
                    C<floatT, HalfProjector::S>(r, mu, nu) * C<floatT, HalfProjector::L>(r, rho, sigma)
                    + C<floatT, HalfProjector::L>(r, mu, nu) * C<floatT, HalfProjector::S>(r, rho, sigma)
                );
                break;
            case Projector::SW:
                result += (
                    C<floatT, HalfProjector::S>(r, mu, nu) * C<floatT, HalfProjector::W>(r, rho, sigma)
                    + C<floatT, HalfProjector::W>(r, mu, nu) * C<floatT, HalfProjector::S>(r, rho, sigma)
                );
                break;
            case Projector::LW:
                result += (
                    C<floatT, HalfProjector::L>(r, mu, nu) * C<floatT, HalfProjector::W>(r, rho, sigma)
                    + C<floatT, HalfProjector::W>(r, mu, nu) * C<floatT, HalfProjector::L>(r, rho, sigma)
                );
                break;
        }
    } else {
        switch (projector) {
            case Projector::TT:
                result += ((floatT) (Ds+1)*(Ds-2)/(2*(Ds+2)*(Ds-1))) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/Ds) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case Projector::LT:
                result += (1.0/(Ds+2)) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/Ds) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case Projector::LL:
                result += (1.0/((Ds+2)*(Ds-1))) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/Ds) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case Projector::SL:
                result += 0.0;
                break;
            case Projector::SS:
                result += (1.0/4.0) * delta(mu, nu) * delta(rho, sigma);
                break;
            default:
                result += 0.0;
                break;
        }
    }

    return result;

}

template<class floatT, Projector projector>
struct ProjectorField {
    typedef GIndexer<All> GInd;

    ProjectorField() {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {
        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        // get desired projector value for the global position
        Tensor4x4Symx4x4SymComplex<floatT> projectorAtSite;
        for (int mu = 0; mu <= 3; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= 3; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            COMPLEX(floatT) value = 0.0;
            value = projectorFunction<floatT, projector>(r, mu, nu, rho, sigma);
            projectorAtSite(mu, nu, rho, sigma, value);
        }
        return projectorAtSite;
    }
};

template<class floatT, bool onDevice, Projector projector>
struct TensorFunctionField {

    LatticeContainerAccessor GAccessor;
    LatticeContainerAccessor PAccessor;
    typedef GIndexer<All> GInd;

    TensorFunctionField(
        const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& G,
        const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& P
    ) : GAccessor(G.getAccessor()), PAccessor(P.getAccessor()) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        // get correlator value at the site
        Tensor4x4Symx4x4SymComplex<floatT> G = GAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);
        Tensor4x4Symx4x4SymComplex<floatT> P = PAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        // contract projector value with correlator value
        COMPLEX(floatT) result = 0.0;
        
        for (int mu = 0; mu <= 3; mu++)
        for (int nu = 0; nu <= 3; nu++)
        for (int rho = 0; rho <= 3; rho++)
        for (int sigma = 0; sigma <= 3; sigma++) {
            result += P(mu, nu, rho, sigma) * G(mu, nu, rho, sigma);
        }

        result /= ((floatT) multiplicity(projector));

        return result;

    }

};

template<class floatT, size_t HaloDepth>
class TensorDecomposition {
    protected:
        // std::vector<LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>> projectorField;
    
    private:
        typedef GIndexer<All, HaloDepth> GInd;

        // // helper function: get the projector field for a given projector
        // template<Projector projector>
        // const LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& getProjector() {
        //     return projectorField[(int)projector];
        // }

        // // helper function: loop over all projectors and initialize their fields
        // template<std::size_t... I>
        // void loopInitProjectorFields(std::index_sequence<I ...>) {
        //     (initProjectorField<static_cast<Projector>(I)>(), ...);
        // }
    
        // // helper function: initialize the projector field for a given projector
        // template<Projector projector>
        // void initProjectorField() {
        //     projectorField[(int)projector].adjustSize(GInd::getLatData().vol4);
        //     projectorField[(int)projector].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, projector>());
        // }

        // helper function: loop over all projectors and get the corresponding tensor functions
        template<bool onDevice, std::size_t... I>
        void loopGetTensorFunction(
            std::index_sequence<I ...>,
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            (getTensorFunction<onDevice, static_cast<Projector>(I)>(tensorField, array), ...);
        }
    
        // helper function: get the r^2-dependent tensor function for a given projector
        template<bool onDevice, Projector projector>
        void getTensorFunction(
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& tensorFunctions
        ) {
            // create lattice containers for the tensor function fields and the projector
            LatticeContainer<onDevice, COMPLEX(floatT)> tensorFunctionField(tensorField.get_CommBase(), "tensorFunctionField", "tensorFunctionField", "tensorFunctionField", "tensorFunctionField");
            LatticeContainer<false, COMPLEX(floatT)> tensorFunctionFieldHost(tensorField.get_CommBase(), "tensorFunctionFieldHost", "tensorFunctionFieldHost", "tensorFunctionFieldHost", "tensorFunctionFieldHost");
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> projectorField(tensorField.get_CommBase(), "projectorField", "projectorField", "projectorField", "projectorField");
            
            // adjust their sizes
            tensorFunctionField.adjustSize(GInd::getLatData().vol4);
            tensorFunctionFieldHost.adjustSize(GInd::getLatData().vol4);
            projectorField.adjustSize(GInd::getLatData().vol4);

            // calculate projector
            projectorField.template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, projector>());
    
            // get the tensor function field by contracting with the projector
            tensorFunctionField.template iterateOverBulk<All, HaloDepth>(TensorFunctionField<floatT, onDevice, projector>(tensorField, projectorField));
    
            // copy to host for upcoming reduction
            tensorFunctionFieldHost.copyFromLatticeContainer(tensorFunctionField);
    
            // reduce to r^2-dependent function and store in array
            reduceR2(tensorFunctionFieldHost.getAccessor(), tensorFunctions[(int)projector]);

            rootLogger.info("Memory inside getTensorFunction:");
            MemoryManagement::memorySummary();
        }
    
        // helper function: reduce a lattice container to an r^2-dependent (spatial) array, ignore time coordinate
        void reduceR2(
            LatticeContainerAccessor latticeAccessor,
            std::vector<COMPLEX(floatT)>& array
        ) {
            typedef GIndexer<All> GInd;
    
            // LatticeContainerAccessor latticeAccessor(lattice.getAccessor());
    
            // sitexyzt glob = GInd::getLatData().globalLatticeXYZT();
    
            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;
            int lt = GInd::getLatData().lt;
    
            // int r2max = rSquared<SpatialTemporal>()
    
            int r2max = getR2max();
    
            // set array to zero initially
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                array[r2] = 0.0;
            }
    
            // array = new COMPLEX(floatT)[r2max+1];
    
            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++)
            for (int t = 0; t < lt; t++) {
                sitexyzt site(x, y, z, t);
                sitexyzt rt = GInd::getLatData().globalPosRelativeToOrigin(site);
                int r2 = rSquared(rt);
                array[r2] += latticeAccessor.getElement<COMPLEX(floatT)>(GInd::getSite(x,y,z,t));
            }
        }
    
    public:
        // standard constructor
        TensorDecomposition(CommunicationBase& commBase) {
            // // create lattice containers for all projector fields
            // for (Projector projector: allProjectors) {
            //     // create unique name for each projector field
            //     std::string name = "P" + std::to_string((int)projector);
            //     // create the lattice container object and place it at the end of the vector
            //     projectorField.emplace_back(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>(commBase, name, name, name, name));
            // }

            // // initialize all projector fields
            // loopInitProjectorFields(std::make_index_sequence<allProjectors.size()>{});
        }

        // main function: get maximum r^2 value for spatial coordinates
        static int getR2max() {
            typedef GIndexer<All> GInd;
            sitexyzt globL = GInd::getLatData().globalLatticeXYZT();

            // maximum spatial r^2 value is (glx/2)^2 + (gly/2)^2 + (glz/2)^2 with global extents glx, gly, glz
            return rSquared(globL) / 4.0;
        }

        // main function: get all r^2-dependent tensor functions for all projectors
        template<bool onDevice>
        void getAllTensorFunctions(
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            loopGetTensorFunction<onDevice>(std::make_index_sequence<allProjectors.size()>{}, tensorField, array);
        }

        // main function: get the counts of sites for each r^2 (spatial) value
        void getR2Counts(
            std::vector<int>& r2Counts
        ) {
            typedef GIndexer<All> GInd;

            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;

            int r2max = getR2max();

            // set r2Counts to zero initially
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                r2Counts[r2] = 0;
            }

            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++) {
                sitexyzt site(x, y, z, 0); // t is irrelevant for r^2
                sitexyzt rt = GInd::getLatData().globalPosRelativeToOrigin(site);
                int r2 = rSquared(rt);
                r2Counts[r2] += 1;
            }
        }
        
    };
