//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"

enum Projector {
    SS, SL, LL, LT, TT
};

struct TensorComponentSet {
    std::vector<Projector> projectors;
    Summation summation;
};

const TensorComponentSet stressStress = {
    {LL, LT, TT},
    Spatial
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
                break;
            case LT:
                result += (
                    r[mu]*r[rho]*deltaT<floatT, summation>(r, nu, sigma)
                    +r[mu]*r[sigma]*deltaT<floatT, summation>(r, nu, rho)
                    +r[nu]*r[rho]*deltaT<floatT, summation>(r, mu, sigma)
                    +r[nu]*r[sigma]*deltaT<floatT, summation>(r, mu, rho)
                )/(2*r2);
                break;
            case TT:
                result += (1.0/2.0) * (
                    deltaT<floatT, summation>(r, mu, rho) * deltaT<floatT, summation>(r, nu, sigma)
                    +deltaT<floatT, summation>(r, mu, sigma) * deltaT<floatT, summation>(r, nu, rho)
                )
                - (1.0/(d-1)) * deltaT<floatT, summation>(r, mu, nu) * deltaT<floatT, summation>(r, rho, sigma);
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
        TensorDecomposition(CommunicationBase& commBase) {
            for (int projector = 0; projector <= TT; projector++) {
                std::string name = "P" + std::to_string(projector);
                projectorField.emplace_back(std::move(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>(commBase, name, name, name, name)));
                projectorField.back().adjustSize(GInd::getLatData().vol4);
            }
            
            projectorField[SS].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, SS, Spatial>());
            projectorField[SL].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, SL, Spatial>());
            projectorField[LL].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, LL, Spatial>());
            projectorField[LT].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, LT, Spatial>());
            projectorField[TT].template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, TT, Spatial>());
        }

        template<Projector projector>
        const LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& getProjector() {
            return projectorField[projector];
        }

        template<bool onDevice, Projector projector>
        void getTensorFunction(
            const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensor_field,
            LatticeContainer<onDevice, COMPLEX(floatT)>& tensor_function_field
        );
};

template<class floatT, size_t HaloDepth>
template<bool onDevice, Projector projector>
void TensorDecomposition<floatT, HaloDepth>::getTensorFunction(
    const LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensor_field,
    LatticeContainer<onDevice, COMPLEX(floatT)>& tensor_function_field
) {
    tensor_function_field.template iterateOverBulk<All, HaloDepth>(ContractTensor<floatT, onDevice, Spatial>(tensor_field, this->getProjector<projector>()));
}
