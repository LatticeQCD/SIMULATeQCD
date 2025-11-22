//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"

enum class Projector {
    LL, LT, TT, SS, SL
};

constexpr std::array<Projector, 5> allProjectors{{
    Projector::SS, Projector::SL, Projector::LL, Projector::LT, Projector::TT
    // Projector::LL
}};


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

    int d = getDimensionFunction(summation);
    floatT r2 = rSquared<summation>(r);

    // if r!=0, add r-dependent part
    if (r2 != 0) {
        switch (projector) {
            case Projector::SS:
                result += (1.0/d) * delta(mu, nu) * delta(rho, sigma);
                break;
            case Projector::SL:
                result += (1.0/(d*sqrt(d-1))) * (
                    deltaHat<floatT, summation>(r, mu, nu) * delta(rho, sigma)
                    + delta(mu, nu) * deltaHat<floatT, summation>(r, rho, sigma)
                );
                break;
            case Projector::LL:
                result += (1.0/(d*(d-1))) * deltaHat<floatT, summation>(r, mu, nu) * deltaHat<floatT, summation>(r, rho, sigma);
                break;
            case Projector::LT:
                result += (
                    r[mu]*r[rho]*deltaT<floatT, summation>(r, nu, sigma)
                    +r[mu]*r[sigma]*deltaT<floatT, summation>(r, nu, rho)
                    +r[nu]*r[rho]*deltaT<floatT, summation>(r, mu, sigma)
                    +r[nu]*r[sigma]*deltaT<floatT, summation>(r, mu, rho)
                )/(2*r2);
                break;
            case Projector::TT:
                result += (1.0/2.0) * (
                    deltaT<floatT, summation>(r, mu, rho) * deltaT<floatT, summation>(r, nu, sigma)
                    +deltaT<floatT, summation>(r, mu, sigma) * deltaT<floatT, summation>(r, nu, rho)
                )
                - (1.0/(d-1)) * deltaT<floatT, summation>(r, mu, nu) * deltaT<floatT, summation>(r, rho, sigma);
                break;
        }
    } else {
        switch (projector) {
            case Projector::SS:
                result += (1.0/d) * delta(mu, nu) * delta(rho, sigma);
                break;
            case Projector::SL:
                result += 0.0;
                break;
            case Projector::LL:
                result += (1.0/((d+2)*(d-1))) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/d) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case Projector::LT:
                result += (1.0/(d+2)) * (
                    delta(mu, rho) * delta(nu, sigma)
                    + delta(mu, sigma) * delta(nu, rho)
                    - (2.0/d) * delta(mu, nu) * delta(rho, sigma)
                );
                break;
            case Projector::TT:
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

template<class floatT, bool onDevice, Summation summation>
struct ContractTensor {

    LatticeContainerAccessor GAccessor;
    LatticeContainerAccessor PAccessor;
    typedef GIndexer<All> GInd;

    ContractTensor(
        const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& G,
        const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& P
    ) : GAccessor(G.getAccessor()), PAccessor(P.getAccessor()) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        // get global position relative to the origin
        sitexyzt r = GInd::getLatData().globalPosRelativeToOrigin(site.coord);

        int indexMax = indexMaxFunction(summation);

        // get correlator value at the site
        Tensor4x4Symx4x4SymComplex<floatT> G_at_site = GAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);
        Tensor4x4Symx4x4SymComplex<floatT> P_at_site = PAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site);

        // contract projector value with correlator value
        COMPLEX(floatT) result = 0.0;
        
        for (int mu = 0; mu <= indexMax; mu++)
        for (int nu = 0; nu <= indexMax; nu++)
        for (int rho = 0; rho <= indexMax; rho++)
        for (int sigma = 0; sigma <= indexMax; sigma++) {
            result += P_at_site(mu, nu, rho, sigma) * G_at_site(mu, nu, rho, sigma);
        }

        return result;

    }

};

template<class floatT, size_t HaloDepth>
class TensorDecomposition {
    protected:
        std::vector<LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>> projectorField;
    private:
        typedef GIndexer<All, HaloDepth> GInd;
    public:
        // standard constructor
        TensorDecomposition(CommunicationBase& commBase) {
            // create lattice containers for all projector fields
            for (Projector projector: allProjectors) {
                std::string name = "P" + std::to_string((int)projector);
                projectorField.emplace_back(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>(commBase, name, name, name, name));
                projectorField.back().adjustSize(GInd::getLatData().vol4);
            }

            loopInitProjectorFields(std::make_index_sequence<allProjectors.size()>{});
            // projectorField[(int)Projector::LL].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, Projector::LL, Spatial>());
            // projectorField[(int)Projector::LT].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, Projector::LT, Spatial>());
            // projectorField[(int)Projector::TT].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, Projector::TT, Spatial>());
            // projectorField[(int)Projector::SL].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, Projector::SL, Spatial>());
            // projectorField[(int)Projector::SS].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, Projector::SS, Spatial>());
        }

        template<std::size_t... I>
        void loopInitProjectorFields(std::index_sequence<I ...>) {
            (initProjectorField<static_cast<Projector>(I)>(), ...);
        }

        template<Projector projector>
        void initProjectorField() {
            projectorField[(int)projector].adjustSize(GInd::getLatData().vol4);
            projectorField[(int)projector].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, projector, Spatial>());
        }

        static int get_r2max() {

            typedef GIndexer<All> GInd;

            sitexyzt globL = GInd::getLatData().globalLatticeXYZT();

            int r2max = rSquared<SpatialTemporal>(globL) / 4.0;

            return r2max;
        }

        template<Projector projector>
        const LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& getProjector() {
            return projectorField[(int)projector];
        }

        template<bool onDevice>
        void getAllTensorFunctions(
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensor_field,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            loopGetTensorFunction<onDevice>(std::make_index_sequence<allProjectors.size()>{}, tensor_field, array);
            // getTensorFunction<onDevice, Projector::LL>(tensor_field, array);

            // // create lattice containers for the function fields
            // LatticeContainer<onDevice, COMPLEX(floatT)> tensor_function_field(tensor_field.get_CommBase(), "tensor_function_field", "tensor_function_field", "tensor_function_field", "tensor_function_field");
            // LatticeContainer<false, COMPLEX(floatT)> tensor_function_field_host(tensor_field.get_CommBase(), "tensor_function_field_host", "tensor_function_field_host", "tensor_function_field_host", "tensor_function_field_host");
            
            // // adjust their sizes
            // tensor_function_field.adjustSize(GInd::getLatData().vol4);
            // tensor_function_field_host.adjustSize(GInd::getLatData().vol4);

            // tensor_function_field.template iterateOverBulk<All, HaloDepth>(ContractTensor<floatT, onDevice, Spatial>(tensor_field, getProjector<Projector::LL>()));

            // tensor_function_field_host.copyFromLatticeContainer(tensor_function_field);

            // reduceR2<SpatialTemporal>(tensor_function_field_host.getAccessor(), array);
        }
        
        template<bool onDevice, std::size_t... I>
        void loopGetTensorFunction(
            std::index_sequence<I ...>,
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensor_field,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            (getTensorFunction<onDevice, static_cast<Projector>(I)>(tensor_field, array), ...);
        }

        template<bool onDevice, Projector projector>
        void getTensorFunction(
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensor_field,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            // create lattice containers for the function fields
            LatticeContainer<onDevice, COMPLEX(floatT)> tensor_function_field(tensor_field.get_CommBase(), "tensor_function_field", "tensor_function_field", "tensor_function_field", "tensor_function_field");
            LatticeContainer<false, COMPLEX(floatT)> tensor_function_field_host(tensor_field.get_CommBase(), "tensor_function_field_host", "tensor_function_field_host", "tensor_function_field_host", "tensor_function_field_host");
            
            // adjust their sizes
            tensor_function_field.adjustSize(GInd::getLatData().vol4);
            tensor_function_field_host.adjustSize(GInd::getLatData().vol4);

            tensor_function_field.template iterateOverBulk<All, HaloDepth>(ContractTensor<floatT, onDevice, Spatial>(tensor_field, getProjector<projector>()));

            tensor_function_field_host.copyFromLatticeContainer(tensor_function_field);

            reduceR2<SpatialTemporal>(tensor_function_field_host.getAccessor(), array[(int)projector]);
        }

        template<Summation summation>
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

            int r2max = get_r2max();

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
                int r2 = GInd::getLatData().globalPosRelativeToOriginAbsoluteValueSquared(site);
                array[r2] += latticeAccessor.getElement<COMPLEX(floatT)>(GInd::getSite(x,y,z,t));
            }
        }

        void get_r2Counts(
            std::vector<int>& counts
        ) {
            typedef GIndexer<All> GInd;

            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;
            int lt = GInd::getLatData().lt;

            int r2max = get_r2max();

            // set counts to zero initially
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                counts[r2] = 0;
            }

            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++)
            for (int t = 0; t < lt; t++) {
                sitexyzt site(x, y, z, t);
                int r2 = GInd::getLatData().globalPosRelativeToOriginAbsoluteValueSquared(site);
                counts[r2] += 1;
            }
        }
        
    };
