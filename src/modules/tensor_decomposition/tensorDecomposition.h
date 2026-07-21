//
// created by Jonas Winter on 22.10.2025
// based on https://arxiv.org/abs/2606.10526
//

#pragma once
#include <stdexcept>
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"

// helper functions for necessary 4D vector operations

template<class floatT>
struct floatTVector {
    floatT x;
    floatT y;
    floatT z;
    floatT t;
    __device__ __host__ floatTVector(floatT x, floatT y, floatT z, floatT t) : x(x), y(y), z(z), t(t) {};
    __device__ __host__ inline floatT& operator[](const int i) {
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        if(i == 3) return t;
        x = 99999.9;
        return x;
    }
};

__device__ __host__ inline sitexyzt getSpatialPart(sitexyzt rt) {
    sitexyzt r = {rt[0], rt[1], rt[2], 0};
    return r;
}

__device__ __host__ inline int getNormSquared(sitexyzt rt) {
    int r2 = 0;
    for (int i = 0; i <= 3; i++) {
        r2 += rt[i] * rt[i];
    }
    return r2;
}

template<class floatT>
__device__ __host__ inline floatT getNorm(sitexyzt rt) {
    int normSquared = getNormSquared(rt);
    floatT norm = sqrt(normSquared);
    return norm;
}

template<class floatT>
__device__ __host__ inline floatTVector<floatT> normalize(sitexyzt rt) {
    floatT norm = getNorm<floatT>(rt);

    floatTVector<floatT> normalized = {rt[0]/norm, rt[1]/norm, rt[2]/norm, rt[3]/norm};

    return normalized;
}

// projectors

// all possible indices for the projectors
// notice: mm from general tau equals UL from averaged tau
enum class Projector {
    TT,
    RT, MT, UT,
    ss, ll, ww, mm,
    sl, sw, sm, lw, lm, wm
};

// all possible indices for the half projectors in the scalar sectors
enum class HalfProjector {
    s, l, w, m
};

// all projectors for the averaged tau scenario, in the relevant order
constexpr std::array<Projector, 10> allProjectorsAveragedTau{{
    Projector::TT, Projector::RT,
    Projector::UT, Projector::mm,
    Projector::ss, Projector::ll, Projector::ww,
    Projector::sl, Projector::sw, Projector::lw
}};

// all projectors for the general tau scenario, in the relevant order
constexpr std::array<Projector, 14> allProjectorsGeneralTau{{
    Projector::TT,
    Projector::RT, Projector::MT, Projector::UT,
    Projector::ss, Projector::ll, Projector::ww, Projector::mm,
    Projector::sl, Projector::sw, Projector::sm, Projector::lw, Projector::lm, Projector::wm
}};

__device__ __host__ inline int multiplicity(Projector projector) {
    switch (projector) {
        case Projector::TT:
            return 2;
        case Projector::RT:
            return 2;
        case Projector::MT:
            return 4;
        case Projector::UT:
            return 2;
        case Projector::ss:
            return 1;
        case Projector::ll:
            return 1;
        case Projector::ww:
            return 1;
        case Projector::mm:
            return 1;
        case Projector::sl:
            return 2;
        case Projector::sw:
            return 2;
        case Projector::sm:
            return 2;
        case Projector::lw:
            return 2;
        case Projector::lm:
            return 2;
        case Projector::wm:
            return 2;
        default:
            return 1;
    }
}

__device__ __host__ inline int delta(int mu, int nu) {
    if (mu == nu) {
        return 1;
    } else {
        return 0;
    }
}

__device__ __host__ inline int deltaSpace(int mu, int nu) {
    sitexyzt u = {0, 0, 0, 1};
    return delta(mu, nu) - u[mu] * u[nu];
}

template<class floatT>
__device__ __host__ inline floatT deltaSpaceTransversal(sitexyzt rt, int mu, int nu) {
    
    sitexyzt r = getSpatialPart(rt);
    int r2 = getNormSquared(r);
    
    if (r2 == 0) {
        return (2.0/3.0) * deltaSpace(mu, nu); // because r_mu*r_nu/r^2 approaches delta^s_munu/d at 0
    } else {
        floatTVector<floatT> rHat = normalize<floatT>(r);
        return deltaSpace(mu, nu) - rHat[mu] * rHat[nu];
    }
}

template<class floatT, HalfProjector halfProjector>
__device__ __host__ inline floatT halfProjectorFunction(sitexyzt rt, int mu, int nu) {
    
    sitexyzt r = getSpatialPart(rt);
    sitexyzt u = {0, 0, 0, 1};
    
    int r2 = getNormSquared(r);
    
    floatT result = 0.0;
    
    // handle r-dependent and r=0 parts separately
    if (r2 != 0) {
        floatTVector<floatT> rHat = normalize<floatT>(r);
        switch (halfProjector) {
            case HalfProjector::s:
                result += (1.0 / 2.0) * delta(mu, nu);
                break;
            case HalfProjector::l:
                result += (1.0 / sqrt(6.0)) * (3.0 * rHat[mu] * rHat[nu] - deltaSpace(mu, nu));
                break;
            case HalfProjector::w:
                result += (1.0 / sqrt(12.0)) * (3.0 * u[mu] * u[nu] - deltaSpace(mu, nu));
                break;
            case HalfProjector::m:
                result += (1.0 / sqrt(2.0)) * (rHat[mu] * u[nu] + u[mu] * rHat[nu]);
                break;
        }
    } else {
        switch (halfProjector) {
            case HalfProjector::s:
                result += (1.0 / 2.0) * delta(mu, nu);
                break;
            case HalfProjector::l:
                result += 0.0;
                break;
            case HalfProjector::w:
                result += (1.0 / sqrt(12.0)) * (3.0 * u[mu] * u[nu] - deltaSpace(mu, nu));
                break;
            case HalfProjector::m:
                result += 0.0;
                break;
        }
    }

    return result;
}

template<class floatT, Projector projector>
__device__ __host__ inline floatT projectorFunction(sitexyzt rt, int mu, int nu, int rho, int sigma) {

    sitexyzt r = getSpatialPart(rt);
    sitexyzt u = {0, 0, 0, 1};
    
    int r2 = getNormSquared(r);
    
    floatT result = 0.0;
    
    // handle r-dependent and r=0 parts separately
    if (r2 != 0) {
        floatTVector<floatT> rHat = normalize<floatT>(r);
        switch (projector) {
            case Projector::TT:
                result += (1.0 / 2.0) * (
                    deltaSpaceTransversal<floatT>(r, mu, rho) * deltaSpaceTransversal<floatT>(r, nu, sigma)
                    + deltaSpaceTransversal<floatT>(r, mu, sigma) * deltaSpaceTransversal<floatT>(r, nu, rho)
                    - deltaSpaceTransversal<floatT>(r, mu, nu) * deltaSpaceTransversal<floatT>(r, rho, sigma)
                );
                break;
            case Projector::RT:
                result += (1.0 / 2.0) * (
                    rHat[mu] * rHat[rho] * deltaSpaceTransversal<floatT>(r, nu, sigma)
                    + rHat[mu] * rHat[sigma] * deltaSpaceTransversal<floatT>(r, nu, rho)
                    + rHat[nu] * rHat[rho] * deltaSpaceTransversal<floatT>(r, mu, sigma)
                    + rHat[nu] * rHat[sigma] * deltaSpaceTransversal<floatT>(r, mu, rho)
                );
                break;
            case Projector::MT:
                result += (1.0 / 2.0) * (
                    (rHat[mu] * u[rho] + u[mu] * rHat[rho]) * deltaSpaceTransversal<floatT>(r, nu, sigma)
                    + (rHat[mu] * u[sigma] + u[mu] * rHat[sigma]) * deltaSpaceTransversal<floatT>(r, nu, rho)
                    + (rHat[nu] * u[rho] + u[nu] * rHat[rho]) * deltaSpaceTransversal<floatT>(r, mu, sigma)
                    + (rHat[nu] * u[sigma] + u[nu] * rHat[sigma]) * deltaSpaceTransversal<floatT>(r, mu, rho)
                );
                break;
            case Projector::UT:
                result += (1.0 / 2.0) * (
                    u[mu] * u[rho] * deltaSpaceTransversal<floatT>(r, nu, sigma)
                    + u[mu] * u[sigma] * deltaSpaceTransversal<floatT>(r, nu, rho)
                    + u[nu] * u[rho] * deltaSpaceTransversal<floatT>(r, mu, sigma)
                    + u[nu] * u[sigma] * deltaSpaceTransversal<floatT>(r, mu, rho)
                );
                break;
            case Projector::ss:
                result += halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma);
                break;
            case Projector::ll:
                result += halfProjectorFunction<floatT, HalfProjector::l>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::l>(r, rho, sigma);
                break;
            case Projector::ww:
                result += halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma);
                break;
            case Projector::mm:
                result += halfProjectorFunction<floatT, HalfProjector::m>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::m>(r, rho, sigma);
                break;
            case Projector::sl:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::l>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::l>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma)
                );
                break;
            case Projector::sw:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma)
                );
                break;
            case Projector::sm:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::m>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::m>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma)
                );
                break;
            case Projector::lw:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::l>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::l>(r, rho, sigma)
                );
                break;
            case Projector::lm:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::l>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::m>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::m>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::l>(r, rho, sigma)
                );
                break;
            case Projector::wm:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::m>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::m>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma)
                );
                break;
        }
    } else {
        switch (projector) {
            case Projector::TT:
                result += (1.0/5.0) * (
                    deltaSpace(mu, rho) * deltaSpace(nu, sigma)
                    + deltaSpace(mu, sigma) * deltaSpace(nu, rho)
                    - (2.0/3.0) * deltaSpace(mu, nu) * deltaSpace(rho, sigma)
                );
                break;
            case Projector::RT:
                result += (1.0/5.0) * (
                    deltaSpace(mu, rho) * deltaSpace(nu, sigma)
                    + deltaSpace(mu, sigma) * deltaSpace(nu, rho)
                    - (2.0/3.0) * deltaSpace(mu, nu) * deltaSpace(rho, sigma)
                );
                break;
            case Projector::MT:
                result += 0.0;
                break;
            case Projector::UT:
                result += (1.0/3.0) * (
                    u[mu] * u[rho] * deltaSpace(nu, sigma)
                    + u[mu] * u[sigma] * deltaSpace(nu, rho)
                    + u[nu] * u[rho] * deltaSpace(mu, sigma)
                    + u[nu] * u[sigma] * deltaSpace(mu, rho)
                );
                break;
            case Projector::ss:
                result += halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma);
                break;
            case Projector::ll:
                result += (1.0/10.0) * (
                    deltaSpace(mu, rho) * deltaSpace(nu, sigma)
                    + deltaSpace(mu, sigma) * deltaSpace(nu, rho)
                    - (2.0/3.0) * deltaSpace(mu, nu) * deltaSpace(rho, sigma)
                );
                break;
            case Projector::ww:
                result += halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma);
                break;
            case Projector::mm:
                result += (1.0/6.0) * (
                    u[mu] * u[rho] * deltaSpace(nu, sigma)
                    + u[mu] * u[sigma] * deltaSpace(nu, rho)
                    + u[nu] * u[rho] * deltaSpace(mu, sigma)
                    + u[nu] * u[sigma] * deltaSpace(mu, rho)
                );
                break;
            case Projector::sl:
                result += 0.0;
                break;
            case Projector::sw:
                result += (
                    halfProjectorFunction<floatT, HalfProjector::s>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::w>(r, rho, sigma)
                    + halfProjectorFunction<floatT, HalfProjector::w>(r, mu, nu) * halfProjectorFunction<floatT, HalfProjector::s>(r, rho, sigma)
                );
                break;
            case Projector::sm:
                result += 0.0;
                break;
            case Projector::lw:
                result += 0.0;
                break;
            case Projector::lm:
                result += 0.0;
                break;
            case Projector::wm:
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

// Get the component function G^X(r_vector,tau) with X in {TT, RT, ...} as a field over the full lattice
template<class floatT, bool onDevice, Projector projector>
struct ComponentFunctionField {

    LatticeContainerAccessor GAccessor;
    LatticeContainerAccessor PAccessor;
    typedef GIndexer<All> GInd;

    ComponentFunctionField(
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
    private:
        typedef GIndexer<All, HaloDepth> GInd;

        // helper function: loop over all projectors of given array and get the corresponding component functions, averaged tau
        template<bool onDevice, const auto& projectorArray, std::size_t... I>
        void loopGetComponentFunctionsAveragedTau(
            std::index_sequence<I ...>,
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            (getComponentFunctionAveragedTau<onDevice, projectorArray[I], I>(tensorField, array), ...);
        }

        // helper function: loop over all projectors of given array and get the corresponding component functions, general tau
        template<bool onDevice, const auto& projectorArray, std::size_t... I>
        void loopGetComponentFunctionsGeneralTau(
            std::index_sequence<I ...>,
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<std::vector<COMPLEX(floatT)>>>& array
        ) {
            (getComponentFunctionGeneralTau<onDevice, projectorArray[I], I>(tensorField, array), ...);
        }
    
        // helper function: get the r^2-dependent, tau-averaged component function for a given projector
        template<bool onDevice, Projector projector, std::size_t idx>
        void getComponentFunctionAveragedTau(
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& componentFunctions
        ) {
            // create lattice containers for the component function fields (device and host) and the projector field
            LatticeContainer<onDevice, COMPLEX(floatT)> GXFieldDevice(tensorField.get_CommBase(), "GXFieldDevice", "GXFieldDevice", "GXFieldDevice", "GXFieldDevice");
            LatticeContainer<false, COMPLEX(floatT)> GXFieldHost(tensorField.get_CommBase(), "GXFieldHost", "GXFieldHost", "GXFieldHost", "GXFieldHost");
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> projectorField(tensorField.get_CommBase(), "projectorField", "projectorField", "projectorField", "projectorField");
            
            // adjust their sizes
            GXFieldDevice.adjustSize(GInd::getLatData().vol4);
            GXFieldHost.adjustSize(GInd::getLatData().vol4);
            projectorField.adjustSize(GInd::getLatData().vol4);

            // calculate projector
            projectorField.template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, projector>());
    
            // get the tensor function field by contracting with the projector
            GXFieldDevice.template iterateOverBulk<All, HaloDepth>(ComponentFunctionField<floatT, onDevice, projector>(tensorField, projectorField));
    
            // copy to host for upcoming reduction
            GXFieldHost.copyFromLatticeContainer(GXFieldDevice);
    
            // reduce to r^2-dependent function and store in array
            reduceR2averageTau(GXFieldHost.getAccessor(), componentFunctions[idx]);
        }

        // helper function: get the r^2-dependent, tau-dependent component function for a given projector
        template<bool onDevice, Projector projector, std::size_t idx>
        void getComponentFunctionGeneralTau(
            LatticeContainer<onDevice, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<std::vector<COMPLEX(floatT)>>>& componentFunctions
        ) {
            // create lattice containers for the component function fields (device and host) and the projector field
            LatticeContainer<onDevice, COMPLEX(floatT)> GXFieldDevice(tensorField.get_CommBase(), "GXFieldDevice", "GXFieldDevice", "GXFieldDevice", "GXFieldDevice");
            LatticeContainer<false, COMPLEX(floatT)> GXFieldHost(tensorField.get_CommBase(), "GXFieldHost", "GXFieldHost", "GXFieldHost", "GXFieldHost");
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> projectorField(tensorField.get_CommBase(), "projectorField", "projectorField", "projectorField", "projectorField");
            
            // adjust their sizes
            GXFieldDevice.adjustSize(GInd::getLatData().vol4);
            GXFieldHost.adjustSize(GInd::getLatData().vol4);
            projectorField.adjustSize(GInd::getLatData().vol4);

            // calculate projector
            projectorField.template iterateOverBulk<All, HaloDepth>(ProjectorField<floatT, projector>());
    
            // get the tensor function field by contracting with the projector
            GXFieldDevice.template iterateOverBulk<All, HaloDepth>(ComponentFunctionField<floatT, onDevice, projector>(tensorField, projectorField));
    
            // copy to host for upcoming reduction
            GXFieldHost.copyFromLatticeContainer(GXFieldDevice);
    
            // reduce to r^2-dependent function and store in array
            reduceR2generalTau(GXFieldHost.getAccessor(), componentFunctions[idx]);
        }
    
        // helper function: reduce a lattice container to an r^2-dependent (spatial) array, sum time coordinate
        void reduceR2averageTau(
            LatticeContainerAccessor latticeAccessor,
            std::vector<COMPLEX(floatT)>& array
        ) {
            typedef GIndexer<All> GInd;
    
            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;
            int lt = GInd::getLatData().lt;
    
            int r2max = getR2max();
    
            // set array to zero initially
            for (int r2 = 0; r2 < array.size(); r2++) {
                array[r2] = 0.0;
            }
    
            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++)
            for (int t = 0; t < lt; t++) {
                sitexyzt site(x, y, z, t);
                sitexyzt rt = GInd::getLatData().globalPosRelativeToOrigin(site);
                int r2 = getNormSquared(getSpatialPart(rt));
                if (r2 < r2max + 1) {
                    array[r2] += latticeAccessor.getElement<COMPLEX(floatT)>(GInd::getSite(x,y,z,t));
                }
            }

            // divide by r2 counts to account for degeneracy
            std::vector<int> r2Counts = std::vector<int>(r2max + 1);
            getR2Counts(r2Counts);
            
            for (int r2 = 0; r2 < array.size(); r2++) {
                if (r2Counts[r2] != 0) {
                    array[r2] /= r2Counts[r2];
                }
            }

            // divide by temporal extend of the lattice as it is a temporal average
            int globLT = GInd::getLatData().globLT;
            for (int r2 = 0; r2 < array.size(); r2++) {
                array[r2] /= globLT;
            }
        }

        // helper function: reduce a lattice container to an r^2-dependent (spatial) array, keep time coordinates separate
        void reduceR2generalTau(
            LatticeContainerAccessor latticeAccessor,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            typedef GIndexer<All> GInd;
    
            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;
            int lt = GInd::getLatData().lt;
    
            int tauMax = getTauMax();
            int r2max = getR2max();
    
            // set array to zero initially
            for (int t = 0; t < array.size(); t++)
            for (int r2 = 0; r2 < array[0].size(); r2++){
                array[t][r2] = 0.0;
            }

            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++)
            for (int t = 0; t < tauMax + 1; t++) {
                sitexyzt site(x, y, z, t);
                sitexyzt rHalf = GInd::getLatData().globalPosSymAroundHalf(site);
                sitexyzt rt = GInd::getLatData().globalPosRelativeToOrigin(site);
                int r2 = getNormSquared(getSpatialPart(rt));

                if (r2 < r2max + 1) {
                    // values for tau = rHalf[3] in (0,N_tau/2) (exluding 0, excluding N_tau/2) are counted twice
                    // e.g. the values for tau=1 and tau=N_tau-1 are equal up to a sign
                    // therefore, in these cases, their values are averaged
                    // if (rHalf[3] > 0 && rHalf[3] < GInd::getLatData().lt / 2.0) {
                        // array[rHalf[3]][r2] += 0.5 * latticeAccessor.getElement<COMPLEX(floatT)>(GInd::getSite(x,y,z,t));
                    // } else {
                        array[rHalf[3]][r2] += latticeAccessor.getElement<COMPLEX(floatT)>(GInd::getSite(x,y,z,t));
                    // }
                }
            }

            // divide by r2 counts to account for degeneracy
            std::vector<int> r2Counts = std::vector<int>(r2max + 1);
            getR2Counts(r2Counts);
            
            for (int t = 0; t < array.size(); t++)
            for (int r2 = 0; r2 < array[t].size(); r2++) {
                if (r2Counts[r2] != 0) {
                    array[t][r2] /= r2Counts[r2];
                }
            }
        }
    
    public:
        // standard constructor
        TensorDecomposition(CommunicationBase& commBase) {}

        // main function: get maximum tau value for temporal coordinate
        static int getTauMax() {
            typedef GIndexer<All> GInd;
            sitexyzt globL = GInd::getLatData().globalLatticeXYZT();

            return std::floor(globL[3] / 2);
        }

        // main function: get maximum r^2 value for spatial coordinates
        static int getR2max() {
            typedef GIndexer<All> GInd;
            sitexyzt globL = GInd::getLatData().globalLatticeXYZT();

            int r2max;

            // maximum spatial r^2 value due to spatial periodicity is
            // (glx/2)^2 + (gly/2)^2 + (glz/2)^2 with global extents glx, gly, glz
            // r2max = getNormSquared(getSpatialPart(globL)) / 4;
            // other maximum spatial r^2 value is (N_{space,min}/2)^2 with the smallest, spatial extent N_{space,min}
            if (globL[0] < globL[1] && globL[0] < globL[2]) {
                r2max = globL[0] * globL[0] / 4;
            } else if (globL[1] < globL[0] && globL[1] < globL[2]) {
                r2max = globL[1] * globL[1] / 4;
            } else {
                r2max = globL[2] * globL[2] / 4;
            }

            return r2max;
        }

        // main function: get all r^2-dependent component functions for all projectors in the averaged tau scenario
        template<bool onDevice>
        void getComponentFunctionsAveragedTau(
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        ) {
            loopGetComponentFunctionsAveragedTau<onDevice, allProjectorsAveragedTau>(std::make_index_sequence<allProjectorsAveragedTau.size()>{}, tensorField, array);
        }

        // main function: get all r^2-dependent component functions for all projectors in the general tau scenario
        template<bool onDevice>
        void getComponentFunctionsGeneralTau(
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorField,
            std::vector<std::vector<std::vector<COMPLEX(floatT)>>>& array
        ) {
            loopGetComponentFunctionsGeneralTau<onDevice, allProjectorsGeneralTau>(std::make_index_sequence<allProjectorsGeneralTau.size()>{}, tensorField, array);
        }

        // main function: get the counts of sites for each r^2 (spatial) value
        static void getR2Counts(
            std::vector<int>& r2Counts
        ) {
            typedef GIndexer<All> GInd;

            int lx = GInd::getLatData().lx;
            int ly = GInd::getLatData().ly;
            int lz = GInd::getLatData().lz;

            int r2max = getR2max();

            // set r2Counts to zero initially
            for (int r2 = 0; r2 < r2Counts.size(); r2++) {
                r2Counts[r2] = 0;
            }

            for (int x = 0; x < lx; x++)
            for (int y = 0; y < ly; y++)
            for (int z = 0; z < lz; z++) {
                sitexyzt site(x, y, z, 0); // t is irrelevant for r^2
                sitexyzt rt = GInd::getLatData().globalPosRelativeToOrigin(site);
                int r2 = getNormSquared(getSpatialPart(rt)); // getSpatialPart to be safe
                if (r2 < r2max + 1) {
                    r2Counts[r2] += 1;
                }
            }
        }

        static int getNumberOfHitR2() {
            int r2max = getR2max();

            std::vector<int> r2Counts = std::vector<int>(r2max+1);

            getR2Counts(r2Counts);

            int numberOfHits = 0;

            for (int r2 = 0; r2 < r2Counts.size(); r2++) {
                if (r2Counts[r2] != 0) numberOfHits++;
            }

            return numberOfHits;
        }
        
    };
