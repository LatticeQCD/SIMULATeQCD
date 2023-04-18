/*
 * correlators.h
 *
 * D. Clarke
 *
 * Header file with function definitions for measuring correlators. The general idea is as follows:
 *
 *   1. Create CorrField objects for the fields to be correlated. CorrField objects are essentially wrappers for
 *      gMemoryPtrs that point to arrays holding the data you want to correlate. CorrField objects also have useful
 *      built-in methods, like initializing them to zero. To populate your CorrField objects, use its MemoryAccessor.
 *   2. Similarly create a Correlator object, which will hold the final correlation.
 *   3. Populate your Correlator object using methods inside the CorrelatorTools class, combined with correlation
 *      archetypes that specify what kind of correlator you want to calculate. For example if your CorrField objects
 *      field1 and field2 represent fields defined on every point in space-time, you can calculate the simple correlator
 *      <field1(x) field2(y)> by using the correlateAt("spacetime") method, feeding it the AxB archetype.
 *
 * Hopefully these tools are enough for you to get started. I encourage you write your own correlation kernels and
 * archetypes to suit your needs, if what you require isn't here already. Please keep all correlation-related methods
 * and classes within this header file so things stay organized.
 *
 */

#ifndef CORRELATORS_H
#define CORRELATORS_H

#include "../../define.h"
#include "../communication/siteComm.h"
#include "../LatticeContainer.h"
#include "gsu2.h"
#include <array>
#include <algorithm>
#include <functional>
#include <iostream>       ///TODO: move kernels to cu file
                          ///TODO: implement correlateAtRestricted

/// Initialize the correlator to zero, regardless of type. ---------------------------------- FUNCTIONS FOR CORRELATIONS
template<class floatT>
__host__ __device__ void inline initCorrToZero(int &corr) {
    corr = 0;
}
template<class floatT>
__host__ __device__ void inline initCorrToZero(floatT &corr) {
    corr = 0.;
}
template<class floatT>
__host__ __device__ void inline initCorrToZero(GSU3<floatT> &corr) {
    corr = gsu3_zero<floatT>();
}
template<class floatT>
__host__ __device__ void inline initCorrToZero(GCOMPLEX(floatT) &corr) {
    corr = GPUcomplex<floatT>(0., 0.);
}

/// Initialize the correlator to one, regardless of type.
template<class floatT>
__host__ __device__ void inline initCorrToOne(int &corr) {
    corr = 1;
}
template<class floatT>
__host__ __device__ void inline initCorrToOne(floatT &corr) {
    corr = 1.;
}
template<class floatT>
__host__ __device__ void inline initCorrToOne(GSU3<floatT> &corr) {
    corr = gsu3_one<floatT>();
}
template<class floatT>
__host__ __device__ void inline initCorrToOne(GCOMPLEX(floatT) &corr) {
    corr = GPUcomplex<floatT>(1., 0.);
}

/// Initialize vector to zero, regardless of type.
template<typename corrType>
void inline initCorrVectorToZero(MemoryAccessor _vector, int maxIndex) {
    corrType corr;
    initCorrToZero(corr);
    for(int dindex = 0; dindex < maxIndex; dindex++) {
        _vector.setValue<corrType>(dindex,corr);
    }
}

/// Initialize vector to one, regardless of type.
template<typename corrType>
void inline initCorrVectorToOne(MemoryAccessor _vector, int maxIndex) {
    corrType corr;
    initCorrToOne(corr);
    for(int dindex = 0; dindex < maxIndex; dindex++) {
        _vector.setValue<corrType>(dindex,corr);
    }
}

/// ------------------------------------------------------------------------------------------------------- BASE CLASSES

template<bool onDevice, typename fieldType>
class CorrField : public LatticeContainer<onDevice, fieldType> {
private:
    size_t _maxIndex;
    std::string _fieldName;
    gMemoryPtr<onDevice> _fieldPtr;

public:

    CorrField(CommunicationBase &commBase, size_t maxIndex, std::string fieldName = "corrfield") :
            LatticeContainer<onDevice,fieldType>(commBase,fieldName), _maxIndex(maxIndex), _fieldName(fieldName) {
        _fieldPtr = this->get_ContainerArrayPtr();
        _fieldPtr->template adjustSize<fieldType>(_maxIndex);
    }

    ~CorrField() {}

    void inline zero() {
        MemoryAccessor _fieldAccessor(_fieldPtr->getPointer());
        initCorrVectorToZero<fieldType>(_fieldAccessor,_maxIndex);
    }

    void inline one() {
        MemoryAccessor _fieldAccessor(_fieldPtr->getPointer());
        initCorrVectorToOne<fieldType>(_fieldAccessor,_maxIndex);
    }

    size_t getMaxIndex() { return _maxIndex; }

    gMemoryPtr<onDevice> &getMemPtr() { return _fieldPtr; }

    CommunicationBase &getCommBase() { return this->get_CommBase(); }

    template<bool onDeviceSrc>
    CorrField<onDevice,fieldType> &operator=(CorrField<onDeviceSrc,fieldType> &source) {
        _fieldPtr->template copyFrom<onDeviceSrc>(source.getMemPtr(), source.getMemPtr()->getSize());
        return *this;
    }
};


template<bool onDevice, typename corrType>
class Correlator : public LatticeContainer<onDevice,corrType>{
private:
    size_t _maxSeparation;
    std::string _corrName;
    gMemoryPtr<onDevice> _corrPtr;

public:
    Correlator(CommunicationBase &commBase, size_t maxSeparation, std::string corrName = "correlator") :
            LatticeContainer<onDevice,corrType>(commBase,corrName), _maxSeparation(maxSeparation), _corrName(corrName) {
        _corrPtr = this->get_ContainerArrayPtr();
        _corrPtr->template adjustSize<corrType>(_maxSeparation+1);
    }

    ~Correlator() {}

    void inline zero() {
        MemoryAccessor _corrAccessor(_corrPtr->getPointer());
        initCorrVectorToZero<corrType>(_corrAccessor,_maxSeparation+1);
    }

    void inline one() {
        MemoryAccessor _corrAccessor(_corrPtr->getPointer());
        initCorrVectorToOne<corrType>(_corrAccessor,_maxSeparation+1);
    }

    size_t getMaxSeparation() { return _maxSeparation; }

    gMemoryPtr<onDevice> &getMemPtr() { return _corrPtr; }

    CommunicationBase &getCommBase() { return this->get_CommBase(); }

    template<bool onDeviceSrc>
    Correlator<onDevice,corrType> &operator=(Correlator<onDeviceSrc,corrType> &source) {
        _corrPtr->template copyFrom<onDeviceSrc>(source.getMemPtr(), source.getMemPtr()->getSize());
        return *this;
    }
};

/// This class has a bunch of constants and methods that are useful for manipulating these correlator objects. Other
/// objects will inherit from this class, and the only way to access the variables then is this->variableName.
template<class floatT, bool onDevice, size_t HaloDepth>
class CorrelatorTools {
protected:

    typedef GIndexer<All, HaloDepth> GInd;

public:

    CorrelatorTools() {}

    ~CorrelatorTools() {}

    /// Shortcuts for commonly accessed quantities.
    const int Nx       = GInd::getLatData().lx;
    const int Ny       = GInd::getLatData().ly;
    const int Nz       = GInd::getLatData().lz;
    const int Nt       = GInd::getLatData().lt;
    const size_t vol3  = GInd::getLatData().globvol3;
    const size_t vol4  = GInd::getLatData().globvol4;

    /// Needed for restricted spatial (RS) correlators.
    const int RSxmax   = Nx/4+1;   /// N/4 is set as the maximum distance in the off-axis direction because diagonal
    const int RSymax   = Ny/4+1;   /// correlations become noisy for large distances. N/4 is somehow large enough.
    const int RSzmax   = Nz/4+1;   /// We have to add 1 because we are indexing displacement, and 0 displacement is
    const int RSonmax  = Nx/2+1;   /// also a possibility.
    const int pvol1    = RSxmax;
    const int pvol2    = RSymax*pvol1;
    const int pvol3    = RSzmax*pvol2;
    const int distmax  = (Nx/2)*(Ny/2)+1;

    /// Needed for unrestricted all (UA) and unrestricted spatial (US) correlators.
    const int Uxmax    = Nx/2+1;
    const int Uymax    = Ny/2+1;
    const int Uzmax    = Nz/2+1;
    const int Utmax    = Nt/2+1;
    const size_t svol1 = Uxmax;
    const size_t svol2 = Uymax*svol1;
    const size_t svol3 = Uzmax*svol2;
    const size_t svol4 = Utmax*svol3;
    const int UAr2max  = Nx*Nx/4 + Ny*Ny/4 + Nz*Nz/4  + Nt*Nt/4;
    const int USr2max  = Nx*Nx/4 + Ny*Ny/4 + Nz*Nz/4;

    /// Array needed to normalize restricted spatial correlations.
    void getFactorArray(std::vector<int> &vec_factor, std::vector<int> &vec_weight);

    /// field1Xfield2 = <field1(x) field2(y)>
    template<typename fieldType, typename corrType, class corrFunc>
    void correlateAt(std::string domain, CorrField<false,fieldType> &field1, CorrField<false,fieldType> &field2,
                     Correlator<false,floatT> &normalization, Correlator<false,corrType> &field1Xfield2,
                     bool XYswapSymmetry = false, std::string normFileDir = "./");

    /// Create and read normalization arrays counting the number of unique displacement vectors.
    void createNorm(std::string domain, CommunicationBase &comm);
    void readNorm(std::string domain, Correlator<false,floatT> &normalization, std::string normFileDir);

    /// Displacement vector de-indexing.
    inline __host__ __device__ void indexToSpaceTimeDisplacement(size_t dindex, int &dx, int &dy, int &dz, int &dt) {
        int rem2, rem1;
        divmod(dindex,svol3,dt,rem2);
        divmod(rem2  ,svol2,dz,rem1);
        divmod(rem1  ,svol1,dy,dx);
    }

    inline __host__ __device__ void indexToSpatialDisplacement(size_t dindex, int &dx, int &dy, int &dz) {
        int rem;
        divmod(dindex,svol2,dz,rem);
        divmod(rem   ,svol1,dy,dx);
    }

    /// Accessors for certain variables relevant to the problem at hand. TODO: Maybe make into maps or an enum
    inline int getr2max(std::string domain) {
        int r2max;
        if(domain=="spacetime") {
            r2max=UAr2max;
        } else if(domain=="spatial") {
            r2max=USr2max;
        } else {
            throw std::runtime_error(stdLogger.fatal("Correlator domain ", domain, " not valid."));
        }
        return r2max;
    }

    inline size_t getdvol(std::string domain) {
        size_t dvol;
        if(domain=="spacetime") {
            dvol=svol4;
        } else if(domain=="spatial") {
            dvol=svol3;
        } else {
            throw std::runtime_error(stdLogger.fatal("Correlator domain ", domain, " not valid."));
        }
        return dvol;
    }

    inline std::string getnormfilePrefix(std::string domain){
        std::string normfilePrefix;
        if(domain=="spacetime") {
            normfilePrefix="UA_s";
        } else if(domain=="spatial") {
            normfilePrefix="US_s";
        } else {
            throw std::runtime_error(stdLogger.fatal("Correlator domain ", domain, " not valid."));
        }
        return normfilePrefix;
    }

    /// Method that crashes upon detection of multiple processors.
    inline void verifySingleProc(CommunicationBase &comm) {
        if( comm.nodes()[0]!=1 || comm.nodes()[1]!=1 || comm.nodes()[2]!=1 || comm.nodes()[3]!=1 ) {
            throw std::runtime_error(stdLogger.fatal("These correlators do not allow partitioning for now!"));
        }
    }
};


/// The way we calculate correlators is to loop over displacement vectors, then find all correlations at that given
/// displacement. This method leads to over-counting. When this CorrelationDegeneracies object is instantiated, it will
/// calculate this over-counting ahead of time.
template<class floatT, bool onDevice, size_t HaloDepth>
class CorrelationDegeneracies : public LatticeContainer<onDevice,size_t>, public CorrelatorTools<floatT,onDevice,HaloDepth> {
private:
    gMemoryPtr<onDevice> _degenPtr;
    std::string _degenName;
    typedef GIndexer<All, HaloDepth> GInd;

public:

    CorrelationDegeneracies(std::string domain, CommunicationBase &commBase, std::string degenName = "correlatorDegeneracy") :
            LatticeContainer<onDevice,size_t>(commBase,degenName), _degenName(degenName) {

        size_t dvol;
        dvol = this->getdvol(domain);

        _degenPtr = this->get_ContainerArrayPtr();
        _degenPtr->template adjustSize<size_t>(dvol);

        size_t degeneracy;
        int dx, dy, dz, dt;
        uint8_t numBC, numZero;

        MemoryAccessor _degeneracyAccessor(_degenPtr->getPointer());
        initCorrVectorToZero<size_t>(_degeneracyAccessor,dvol);

        if(domain=="spacetime") {

            for (size_t dindex=0; dindex<dvol; dindex++) {
                this->indexToSpaceTimeDisplacement(dindex, dx, dy, dz, dt);
                if ( (dx==0) && (dy==0) && (dz==0) && (dt==0) ) {
                    /// Contact terms are over-counted 8 times because every spatial coordinate reflection does nothing.
                    degeneracy = 8;
                } else {
                    /// If a displacement vector coordinate stretches to half the lattice extension, you get an over-count
                    /// factor of 2 because of periodic BCs. E.g. (Nx/2, 0, 0, t) going from site m to n is the same
                    /// vector as (-Nx/2, 0, 0, t) going from n to m.
                    numBC      = (dx==(this->Nx)/2) + (dy==(this->Ny)/2) + (dz==(this->Nz)/2) + (dt==(this->Nt)/2);
                    /// For the spatial directions, you get an over-count factor of 2 if one of them equals zero, again
                    /// because the spatial reflection does nothing. For euclidean time direction, if it is zero, every
                    /// displacement from m to n is matched by the displacement with all its spatial coordinates reflected
                    /// going from n to m.
                    numZero    = (dx==0)            + (dy==0)            + (dz==0)            + (dt==0);
                    degeneracy = pow(2,numZero+numBC);
                }
                _degeneracyAccessor.setValue(dindex, degeneracy);
            }

        } else if(domain=="spatial") {

            for (size_t dindex=0; dindex<dvol; dindex++) {
                this->indexToSpatialDisplacement(dindex, dx, dy, dz);
                if ( (dx==0) && (dy==0) && (dz==0) ) {
                    /// Contact terms are over-counted 4 times because every x-y-coordinate reflection does nothing.
                    degeneracy = 4;
                } else {
                    /// If a displacement vector coordinate stretches to half the lattice extension, you get an over-count
                    /// factor of 2 because of periodic BCs. E.g. (Nx/2, 0, z) going from site m to n is the same
                    /// vector as (-Nx/2, 0, z) going from n to m.
                    numBC      = (dx==(this->Nx)/2) + (dy==(this->Ny)/2) + (dz==(this->Nz)/2);
                    /// For the spatial directions, you get an over-count factor of 2 if one of them equals zero, again
                    /// because the x-y-reflection does nothing. For z-direction, if it is zero, every displacement from
                    /// m to n is matched by the displacement with all its x-, y-coordinates reflected going from n to m.
                    numZero    = (dx==0)            + (dy==0)            + (dz==0);
                    degeneracy = pow(2,numZero+numBC);
                }
                _degeneracyAccessor.setValue(dindex, degeneracy);
            }

        } else {
            throw std::runtime_error(stdLogger.fatal("Correlator domain ", domain, " not valid."));
        }
    }

    ~CorrelationDegeneracies() {}

    gMemoryPtr<onDevice> &getMemPtr() { return _degenPtr; }

    CommunicationBase &getCommBase() { return this->get_CommBase(); }
};

/// ----------------------------------------------------------------------------------------------------------- INDEXING

/// Trivial read index, in case you need/want to do indexing inside the Kernel. TODO: Probably should be in indexer?
struct PassIndex {
    inline __host__ __device__ size_t operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        return blockDim.x * blockIdx.x + threadIdx.x;
    }
};

/// For fields that depend on x.
template<size_t HaloDepth>
struct ReadIndexSpacetime {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<All, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};

/// For fields that depend on spatial x.
template<size_t HaloDepth>
struct ReadIndexSpatial {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<All, HaloDepth> GInd;
        gSite site = GInd::getSiteSpatial(i);
        return site;
    }
};

/// ---------------------------------------------------------------------------------------------- CORRELATOR ARCHETYPES

/// What follows are some different kinds of correlators between the fields A and B. The class itself determines what
/// kind of correlator is being calculated. Then the convention is to name the actual correlating function inside
/// the class "orrelate", and inside your kernel, to instantiate the correlation kind as "c". This way you can
/// correlate two fields with the command c.orrelate(A,B).
///
/// You may wonder why it is implemented as templates. When you pick a correlator template, e.g. AxB, the correct
/// function will be explicitly written into the code during compile time. Originally I had the idea to use a pointer
/// to the function, but this does not work, because the pointer could be determined on the CPU, which is then not
/// valid when running on the CPU.
template<class floatT>
class AxB {
public:
    __host__ __device__ floatT inline orrelate(floatT A, floatT B) {
        return A*B;
    }
    __host__ __device__ GCOMPLEX(floatT) inline orrelate(GCOMPLEX(floatT) A, GCOMPLEX(floatT) B) {
        return A*B;
    }
    __host__ __device__ GCOMPLEX(floatT) inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_c(A*B);
    }
};

template<class floatT>
class trAxtrBt {
public:
    __host__ __device__ GCOMPLEX(floatT) inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_c(A)*tr_c(dagger(B));
    }
};

template<class floatT>
class trReAxtrReB {
public:
    __host__ __device__ floatT inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_d(A)*tr_d(B);
    }
};

template<class floatT>
class trImAxtrImB {
public:
    __host__ __device__ floatT inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_i(A)*tr_i(B);
    }
};

template<class floatT>
class trAxBt {
public:
    __host__ __device__ GCOMPLEX(floatT) inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_c(A*dagger(B));
    }
};

template<class floatT>
class polCorrAVG {
public:
    __host__ __device__ floatT inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return real(tr_c(A)*tr_c(dagger(B)))/9.;
    }
};
template<class floatT>
class polCorrSIN {
public:
    __host__ __device__ floatT inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        return tr_d(A,dagger(B))/3.;
    }
};
template<class floatT>
class polCorrOCT {
public:
    __host__ __device__ floatT inline orrelate(GSU3<floatT> A, GSU3<floatT> B) {
        floatT avg  = real(tr_c(A)*tr_c(dagger(B)));
        floatT sin  = tr_d(A,dagger(B));
        return (0.125*avg - 0.04166666666*sin);
    }
};

/// ------------------------------------------------------------------------------------------------- CORRELATOR KERNELS

/// Kernel to compute correlations between all spacetime points. It is assumed that the correlation vector field1Xfield2
/// has been initialized to zero going in. Also assumes the sites m and n at which the fields are correlated can be
/// exchanged without changing the value of the correlator; when this is true, one gets some speedup.
/// TEMPLATES: <precision, halo depth, type of fields to be correlated, type of resulting correlator, correlation func>
/// INTENT:   IN--field1, field2; OUT--field1Xfield2
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct SpacetimePairKernelSymm : CorrelatorTools<floatT, true, HaloDepth> {
    LatticeContainerAccessor _field1;
    LatticeContainerAccessor _field2;
    LatticeContainerAccessor _field1Xfield2;
    size_t _dindex;
    SpacetimePairKernelSymm(LatticeContainerAccessor field1, LatticeContainerAccessor field2, LatticeContainerAccessor field1Xfield2, size_t dindex)
    : _field1(field1), _field2(field2), _field1Xfield2(field1Xfield2), _dindex(dindex), CorrelatorTools<floatT, true, HaloDepth>() {}

    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        size_t     m, n;
        fieldType  field1m, field2n;
        corrType   corrContrib;
        int        dx, dy, dz, dt, mx, my, mz, mt;

        corrFunc c;

        this->indexToSpaceTimeDisplacement(_dindex, dx, dy, dz, dt);

        /// m is the index for field1.
        m = site.isite;
        _field1.getValue<fieldType>(m, field1m);
        mx = site.coord.x;
        my = site.coord.y;
        mz = site.coord.z;
        mt = site.coord.t;

        initCorrToZero(corrContrib);

        /// The next 8 sites are all in the cone stepping in the forward time direction.
        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Set correlator contribution.
        _field1Xfield2.setValue<corrType>(m, corrContrib);
    }
};

/// Same as above, but now it is no longer assumed that the points m and n can be exchanged.
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct SpacetimePairKernel : CorrelatorTools<floatT, true, HaloDepth> {
    LatticeContainerAccessor _field1;
    LatticeContainerAccessor _field2;
    LatticeContainerAccessor _field1Xfield2;
    size_t _dindex;
    SpacetimePairKernel(LatticeContainerAccessor field1, LatticeContainerAccessor field2, LatticeContainerAccessor field1Xfield2, size_t dindex)
            : _field1(field1), _field2(field2), _field1Xfield2(field1Xfield2), _dindex(dindex), CorrelatorTools<floatT, true, HaloDepth>() {}

    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        size_t     m, n;
        fieldType  field1m, field2n;
        corrType   corrContrib;
        int        dx, dy, dz, dt, mx, my, mz, mt;

        corrFunc c;

        this->indexToSpaceTimeDisplacement(_dindex, dx, dy, dz, dt);

        m = site.isite;
        _field1.getValue<fieldType>(m, field1m);
        mx = site.coord.x;
        my = site.coord.y;
        mz = site.coord.z;
        mt = site.coord.t;

        initCorrToZero(corrContrib);

        /// Cone in forward time direction.
        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),(mt+dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Cone in reverse time direction.
        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),(           mz+dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(           mz+dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSite(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),((this->Nt)+mt-dt)%(this->Nt)).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Set correlator contribution.
        _field1Xfield2.setValue<corrType>(m, corrContrib);
    }
};

/// Kernel to compute correlations between all spatial points. It is assumed the correlator is invariant under the
/// exchange m <--> n. For such correlators, this kernel achieves a factor 2 speedup.
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct SpatialPairKernelSymm : CorrelatorTools<floatT, true, HaloDepth> {
    LatticeContainerAccessor _field1;
    LatticeContainerAccessor _field2;
    LatticeContainerAccessor _field1Xfield2;
    size_t _dindex;
    SpatialPairKernelSymm(LatticeContainerAccessor field1, LatticeContainerAccessor field2, LatticeContainerAccessor field1Xfield2, size_t dindex)
            : _field1(field1), _field2(field2), _field1Xfield2(field1Xfield2), _dindex(dindex), CorrelatorTools<floatT, true, HaloDepth>() {}

    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        size_t     m, n;
        fieldType  field1m, field2n;
        corrType   corrContrib;
        int        dx, dy, dz, mx, my, mz;

        corrFunc c;

        this->indexToSpatialDisplacement(_dindex, dx, dy, dz);

        /// m is the index for field1.
        m = site.isite;
        _field1.getValue<fieldType>(m, field1m);
        mx = site.coord.x;
        my = site.coord.y;
        mz = site.coord.z;

        initCorrToZero(corrContrib);

        /// The next 4 sites are all in the cone stepping in the forward z-direction.
        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Set correlator contribution.
        _field1Xfield2.setValue<corrType>(m, corrContrib);
    }
};

/// Same as above, but we no longer assume m <--> n symmetry.
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct SpatialPairKernel : CorrelatorTools<floatT, true, HaloDepth> {
    LatticeContainerAccessor _field1;
    LatticeContainerAccessor _field2;
    LatticeContainerAccessor _field1Xfield2;
    size_t _dindex;
    SpatialPairKernel(LatticeContainerAccessor field1, LatticeContainerAccessor field2, LatticeContainerAccessor field1Xfield2, size_t dindex)
            : _field1(field1), _field2(field2), _field1Xfield2(field1Xfield2), _dindex(dindex), CorrelatorTools<floatT, true, HaloDepth>() {}

    __device__ __host__ void operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        size_t     m, n;
        fieldType  field1m, field2n;
        corrType   corrContrib;
        int        dx, dy, dz, mx, my, mz;

        corrFunc c;

        this->indexToSpatialDisplacement(_dindex, dx, dy, dz);

        m = site.isite;
        _field1.getValue<fieldType>(m, field1m);
        mx = site.coord.x;
        my = site.coord.y;
        mz = site.coord.z;

        initCorrToZero(corrContrib);

        /// Cone in the forward z-direction.
        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),(mz+dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Cone in the reverse z-direction.
        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),(           my+dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial((           mx+dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        n = GInd::getSiteSpatial(((this->Nx)+mx-dx)%(this->Nx),((this->Ny)+my-dy)%(this->Ny),((this->Nz)+mz-dz)%(this->Nz),0).isite;
        _field2.getValue<fieldType>(n, field2n);
        corrContrib += c.orrelate(field1m, field2n);

        /// Set correlator contribution.
        _field1Xfield2.setValue<corrType>(m, corrContrib);
    }
};

/// Kernel to compute restricted, spatial, off-axis correlations. "Restricted" in this sense means that not all possible
/// spatial correlations are calculated. It is assumed the correlator is symmetric under m <--> n.
/// TEMPLATES: <precision, halo depth, type of fields to be correlated, type of resulting correlator, correlation func>
/// INTENT:   IN--field1, field2; OUT--field1Xfield2
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct RestrictedOffAxisKernel : CorrelatorTools<floatT, true, HaloDepth> {
    MemoryAccessor _field1;           /// field arrays, indexed by spatial site
    MemoryAccessor _field2;
    MemoryAccessor _field1Xfield2;    /// correlator of the fields, indexed by separation
    RestrictedOffAxisKernel(MemoryAccessor field1, MemoryAccessor field2, MemoryAccessor field1Xfield2)
            : _field1(field1), _field2(field2), _field1Xfield2(field1Xfield2), CorrelatorTools<floatT,true,HaloDepth>() {}

    /// The idea behind this calculation is as follows: We are given a displacement (dx,dy,dz). Then we loop over all
    /// spacelike sites, here called m. If all d%>0, there are four off-axis correlations in the forward (positive z)
    /// direction. (Backward correlations will be counted from the forward correlation of some other m.) A possible
    /// displacement is (1,0,0); therefore some on-axis correlations are computed already in the off-axis kernel. This
    /// is taken into account in the on-axis kernel.
    __device__ __host__ void operator()(size_t dindex) { /// dindex indexes displacement vector

        typedef GIndexer<All,HaloDepth> GInd;
        size_t       m,n1,n2,n3,n4;
        int          dx,dy,dz,rem;
        fieldType    field1m,field2n;
        corrType     corr;

        corrFunc c;

        /// Determine the displacement vector (dx,dy,dz) from the index.
        divmod((int)dindex,this->pvol2,dz,rem);
        divmod(rem        ,this->pvol1,dy,dx );

        initCorrToZero(corr);

        for(int tx=0;tx<(this->Nx);tx++)
        for(int ty=0;ty<(this->Ny);ty++)
        for(int tz=0;tz<(this->Nz);tz++) {
            /// Our originating site, m.
            m  = GInd::getSiteSpatial(            tx       ,                    ty       ,         tz               ,0).isite;
            /// The "off-axis" sites.
            n1 = GInd::getSiteSpatial(           (tx+dx)%(this->Nx),           (ty+dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n2 = GInd::getSiteSpatial(((this->Nx)+tx-dx)%(this->Nx),           (ty+dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n3 = GInd::getSiteSpatial(           (tx+dx)%(this->Nx),((this->Ny)+ty-dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n4 = GInd::getSiteSpatial(((this->Nx)+tx-dx)%(this->Nx),((this->Ny)+ty-dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            /// Calculate contribution to correlator.
            _field1.getValue<fieldType>(m ,field1m);
            _field2.getValue<fieldType>(n1,field2n);
            corr += c.orrelate(field1m,field2n);
            _field2.getValue<fieldType>(n2,field2n);
            corr += c.orrelate(field1m,field2n);
            _field2.getValue<fieldType>(n3,field2n);
            corr += c.orrelate(field1m,field2n);
            _field2.getValue<fieldType>(n4,field2n);
            corr += c.orrelate(field1m,field2n);
        }
        /// The factor 4 comes because, as discussed earlier, and as seen directly in the above block, there are 4
        /// off-axis correlations in the positive z direction.
        corr /= (this->vol3*4.);
        _field1Xfield2.setValue<corrType>(dindex,corr);
    }
};

/// Kernel to compute restricted, spatial, on-axis correlations. For use with single GPU.
/// TEMPLATES: <precision, halo depth, type of fields to be correlated, type of resulting correlator, correlation func>
/// INTENT:   IN--field1, field2, field1Xfield2off; OUT--field1Xfield2on
template<class floatT, size_t HaloDepth, typename fieldType, typename corrType, class corrFunc>
struct RestrictedOnAxisKernel : CorrelatorTools<floatT, true, HaloDepth> {
    MemoryAccessor _field1;
    MemoryAccessor _field2;
    MemoryAccessor _field1Xfield2off;
    MemoryAccessor _field1Xfield2on;
    RestrictedOnAxisKernel(MemoryAccessor field1, MemoryAccessor field2, MemoryAccessor field1Xfield2off,
                           MemoryAccessor field1Xfield2on)
            : _field1(field1), _field2(field2), _field1Xfield2off(field1Xfield2off), _field1Xfield2on(field1Xfield2on),
              CorrelatorTools<floatT, true, HaloDepth>() {}

    __device__ __host__ void operator()(size_t dx){ /// Now dx corresponds to a separation, rather than a displacement

        typedef GIndexer<All,HaloDepth> GInd;
        size_t       m,n1,n2,n3;
        fieldType    field1m,field2n;
        corrType     corr,onx,ony,onz;

        corrFunc c;

        if(dx<(this->RSxmax)) {
            /// This is the part where we grab the on-axis calculations that were already done in the above kernel.
            _field1Xfield2off.getValue<corrType>(dx              ,onx);
            _field1Xfield2off.getValue<corrType>(dx*(this->pvol1),ony);
            _field1Xfield2off.getValue<corrType>(dx*(this->pvol2),onz);
            corr = (onx+ony+onz)/3.;
            _field1Xfield2on.setValue<corrType>(dx,corr);

        } else {
            /// And these are the on-axis correlators that haven't been computed yet.
            initCorrToZero(corr);
            for(int tx=0;tx<(this->Nx);tx++)
            for(int ty=0;ty<(this->Ny);ty++)
            for(int tz=0;tz<(this->Nz);tz++) {
                m =GInd::getSiteSpatial( tx               , ty               , tz               ,0).isite;
                n1=GInd::getSiteSpatial((tx+dx)%(this->Nx), ty               , tz               ,0).isite;
                n2=GInd::getSiteSpatial( tx               ,(ty+dx)%(this->Ny), tz               ,0).isite;
                n3=GInd::getSiteSpatial( tx               , ty               ,(tz+dx)%(this->Nz),0).isite;
                _field1.getValue<fieldType>(m ,field1m);
                _field2.getValue<fieldType>(n1,field2n);
                corr += c.orrelate(field1m,field2n);
                _field2.getValue<fieldType>(n2,field2n);
                corr += c.orrelate(field1m,field2n);
                _field2.getValue<fieldType>(n3,field2n);
                corr += c.orrelate(field1m,field2n);
            }
            corr /= ((this->vol3)*3.);
            _field1Xfield2on.setValue<corrType>(dx,corr);
        }
    }
};

/// ------------------------------------------------------------------------------------------- CORRELATOR TOOLS METHODS

/// Besides the degeneracy already accounted for by the CorrelationDegeneracies object, we have to also count the
/// number of unique vectors contributing to the correlator at each r2.
template<class floatT, bool onDevice, size_t HaloDepth>
void CorrelatorTools<floatT,onDevice,HaloDepth>::createNorm(std::string domain, CommunicationBase &comm) {

    int dx, dy, dz, dt, r2, r2max;
    std::string normfilePrefix;
    floatT norm;

    r2max = getr2max(domain);
    normfilePrefix = getnormfilePrefix(domain);

    Correlator<false,floatT> normalization(comm, r2max);

    normalization.zero();
    LatticeContainerAccessor _normalization(normalization.getAccessor());

    std::stringstream normfilename;
    std::ofstream normfile;

    normfilename << normfilePrefix << std::to_string(Nx) << "t" << std::to_string(Nt) << ".norm";
    rootLogger.info("Creating normalization file " ,  normfilename.str());
    normfile.open(normfilename.str());

    if(domain=="spacetime") {

        for(int mx=0; mx<Nx; mx++)
        for(int my=0; my<Ny; my++)
        for(int mz=0; mz<Nz; mz++)
        for(int mt=0; mt<Nt; mt++) {
            for(int nx=0; nx<Nx; nx++)
            for(int ny=0; ny<Ny; ny++)
            for(int nz=0; nz<Nz; nz++)
            for(int nt=0; nt<Nt; nt++) {

                dx = abs(mx-nx);
                if (dx>Nx/2) dx=Nx-dx;
                dy = abs(my-ny);
                if (dy>Ny/2) dy=Ny-dy;
                dz = abs(mz-nz);
                if (dz>Nz/2) dz=Nz-dz;
                dt = abs(mt-nt);
                if (dt>Nt/2) dt=Nt-dt;

                r2 = dx*dx + dy*dy + dz*dz + dt*dt;

                if (r2==0) {
                    _normalization.getValue(r2,norm);
                    norm += 1.0;
                    _normalization.setValue(r2,norm);
                } else {
                    _normalization.getValue(r2,norm);
                    norm += 0.5;
                    _normalization.setValue(r2,norm);
                }
            }
        }

    } else if(domain=="spatial") {

        for(int mx=0; mx<Nx; mx++)
        for(int my=0; my<Ny; my++)
        for(int mz=0; mz<Nz; mz++) {
                for(int nx=0; nx<Nx; nx++)
                for(int ny=0; ny<Ny; ny++)
                for(int nz=0; nz<Nz; nz++) {

                        dx = abs(mx-nx);
                        if (dx>Nx/2) dx=Nx-dx;
                        dy = abs(my-ny);
                        if (dy>Ny/2) dy=Ny-dy;
                        dz = abs(mz-nz);
                        if (dz>Nz/2) dz=Nz-dz;

                        r2 = dx*dx + dy*dy + dz*dz;

                        if (r2==0) {
                            _normalization.getValue(r2,norm);
                            norm += 1.0;
                            _normalization.setValue(r2,norm);
                        } else {
                            _normalization.getValue(r2,norm);
                            norm += 0.5;
                            _normalization.setValue(r2,norm);
                        }
                }
        }

    } else {
        throw std::runtime_error(stdLogger.fatal("Correlator domain ", domain, " not valid."));
    }

    for (int ir2=0; ir2<r2max+1; ir2++) {
        _normalization.getValue<floatT>(ir2,norm);
        normfile << ir2 << "    " << std::setprecision(16) << norm << std::endl;
    }

    normfile.close();
}

template<class floatT, bool onDevice, size_t HaloDepth>
void CorrelatorTools<floatT,onDevice,HaloDepth>::readNorm(std::string domain, Correlator<false,floatT> &normalization,
                                                          std::string normFileDir) {

    int jr2, r2max;
    floatT norm;
    std::string normfilePrefix;

    normalization.zero();
    LatticeContainerAccessor _normalization(normalization.getAccessor());

    r2max = getr2max(domain);
    normfilePrefix = getnormfilePrefix(domain);

    std::stringstream normfilename;
    std::ifstream normfile;
    normfilename << normFileDir << normfilePrefix << std::to_string(Nx) << "t" << std::to_string(Nt) << ".norm";
    rootLogger.info("Reading from normalization file " ,  normfilename.str());
    normfile.open(normfilename.str());

    if(!normfile.good()) {
        throw std::runtime_error(stdLogger.fatal("Problem opening norm file."));
    }

    for (int ir2=0; ir2<r2max+1; ir2++) {
        normfile >> jr2;
        normfile >> norm;
        if(ir2!=jr2) {
            throw std::runtime_error(stdLogger.fatal("Problem reading norm file. ir2, jr2 = ", ir2, jr2));
        }
        _normalization.setValue<floatT>(ir2,norm);
    }

    normfile.close();
}

template<class floatT, bool onDevice, size_t HaloDepth>
template<typename fieldType, typename corrType, class corrFunc>
void CorrelatorTools<floatT,onDevice,HaloDepth>::correlateAt(std::string domain, CorrField<false,fieldType> &field1,
        CorrField<false,fieldType> &field2, Correlator<false,floatT> &normalization, Correlator<false,corrType> &field1Xfield2,
        bool XYswapSymmetry, std::string normFileDir) {

    int dx, dy, dz, dt, r2, r2max;
    floatT norm;
    corrType corr, corrContrib;
    size_t degeneracy;

    verifySingleProc(field1.getCommBase());

    r2max = getr2max(domain);

    /// Initialize the correlator array.
    field1Xfield2.zero();

    /// Create GPU copies of CPU objects, so their accessors can be fed into the kernel.
    CorrField<true,fieldType> GPUfield1(field1.getCommBase(),field1.getMaxIndex());
    CorrField<true,fieldType> GPUfield2(field2.getCommBase(),field2.getMaxIndex());
    GPUfield1 = field1;
    GPUfield2 = field2;
    LatticeContainerAccessor _GPUfield1(GPUfield1.getAccessor());
    LatticeContainerAccessor _GPUfield2(GPUfield2.getAccessor());

    /// An auxiliary container needed for field1Xfield2 reduction
    CorrField<true,corrType> GPUfield1Xfield2Aux(field1Xfield2.getCommBase(),field1.getMaxIndex());
    LatticeContainerAccessor _GPUfield1Xfield2Aux(GPUfield1Xfield2Aux.getAccessor());

    /// Accessors for output containers
    LatticeContainerAccessor _normalization(normalization.getAccessor());
    LatticeContainerAccessor _field1Xfield2(field1Xfield2.getAccessor());

    /// Accessor for look-up table containing the degeneracy count
    CorrelationDegeneracies<floatT,false,HaloDepth> corrDegens(domain,field1.getCommBase());
    LatticeContainerAccessor _corrDegens(corrDegens.getAccessor());

    /// Loop over displacement vectors to compute correlations
    if(domain=="spacetime") {

        ReadIndexSpacetime<HaloDepth> calcReadIndex;
        for (size_t dindex=0; dindex<svol4; dindex++) {
            _corrDegens.getValue<size_t>(dindex,degeneracy);
            indexToSpaceTimeDisplacement(dindex, dx, dy, dz, dt);
            r2 = dx*dx + dy*dy + dz*dz + dt*dt;
            /// Given field1 and field2, compute field1Xfield2 displaced by dindex, store in temporary array
            if(XYswapSymmetry) {
                iterateFunctorNoReturn<onDevice>(
                    SpacetimePairKernelSymm<floatT,HaloDepth,fieldType,corrType,corrFunc>(_GPUfield1, _GPUfield2, _GPUfield1Xfield2Aux, dindex),
                    calcReadIndex, vol4 );
            } else {
                iterateFunctorNoReturn<onDevice>(
                    SpacetimePairKernel<floatT,HaloDepth,fieldType,corrType,corrFunc>(_GPUfield1, _GPUfield2, _GPUfield1Xfield2Aux, dindex),
                    calcReadIndex, vol4 );
            }
            /// Add this contribution to field1Xfield2
            _field1Xfield2.getValue<corrType>(r2,corr);
            GPUfield1Xfield2Aux.reduce(corrContrib,vol4);
            corr += corrContrib*(1.0/degeneracy);          /// Normalization coming from kernel calculation strategy
            _field1Xfield2.setValue<corrType>(r2,corr);
        }

    } else if(domain=="spatial") {

        ReadIndexSpatial<HaloDepth> calcReadIndex;
        for (size_t dindex=0; dindex<svol3; dindex++) {
            _corrDegens.getValue<size_t>(dindex,degeneracy);
            indexToSpatialDisplacement(dindex, dx, dy, dz);
            r2 = dx*dx + dy*dy + dz*dz;
            if(XYswapSymmetry) {
                iterateFunctorNoReturn<onDevice>(
                    SpatialPairKernelSymm<floatT,HaloDepth,fieldType,corrType,corrFunc>(_GPUfield1, _GPUfield2, _GPUfield1Xfield2Aux, dindex),
                    calcReadIndex, vol3 );
            } else {
                iterateFunctorNoReturn<onDevice>(
                    SpatialPairKernel<floatT,HaloDepth,fieldType,corrType,corrFunc>(_GPUfield1, _GPUfield2, _GPUfield1Xfield2Aux, dindex),
                    calcReadIndex, vol3 );
            }
            _field1Xfield2.getValue<corrType>(r2,corr);
            GPUfield1Xfield2Aux.reduce(corrContrib,vol3);
            corr += corrContrib*(1.0/degeneracy);
            _field1Xfield2.setValue<corrType>(r2,corr);
        }

    } else {
        throw std::runtime_error(stdLogger.fatal("Correlator domain not set correctly."));
    }

    /// Normalization coming from number of pairs at a specific r2.
    readNorm(domain,normalization, normFileDir);
    for (int ir2=0; ir2<r2max+1; ir2++) {
        _normalization.getValue<floatT>(ir2,norm);
        if(norm > 0) {
            _field1Xfield2.getValue<corrType>(ir2,corr);
            _normalization.getValue<floatT>(ir2,norm);
            if(XYswapSymmetry) {
                corr = corr*(1.0/norm);
            } else {
                corr = corr*(0.5/norm);
            }
            _field1Xfield2.setValue<corrType>(ir2,corr);
        }
    }
}

template<class floatT, bool onDevice, size_t HaloDepth>
void CorrelatorTools<floatT,onDevice,HaloDepth>::getFactorArray(std::vector<int> &vec_factor, std::vector<int> &vec_weight){

    int g,nullen,psite,qnorm;

    vec_weight = std::vector<int>(pvol3);
    vec_factor = std::vector<int>(distmax);
    /// Compute the weights.
    for (int dx=0 ; dx<RSxmax ; dx++)
    for (int dy=0 ; dy<RSymax ; dy++)
    for (int dz=0 ; dz<RSzmax ; dz++) {
        /// nullen counts how many of {dx,dy,dz} are zero, e.g. nullen==2 means two of dx, dy or dz are zero. We use
        /// this to determine how many unique vectors there given our method.
        ///     nullen == 0: No overcount, i.e. RestrictedOffAxisKernel gives 4 unique contributions
        ///            == 1: x,y-reflections do nothing, and dz=0 means a m-->n is matched by a later n-->m. 2 unique.
        ///            == 2: Same logic as above leads to 1 unique.
        ///            == 3: This is the contact term. 1 unique.
        nullen            = (dx==0) + (dy==0) + (dz==0);
        g                 = nullen==0 ? 4 : std::max(3-nullen,1);
        psite             = dx + dy*pvol1 + dz*pvol2;
        vec_weight[psite] = g;
    }
    for (int dx=0 ; dx<distmax ; dx++) {
        vec_factor[dx] = 0;
    }
    for (int dx=0 ; dx<RSxmax ; dx++)
    for (int dy=0 ; dy<RSymax ; dy++)
    for (int dz=0 ; dz<RSzmax ; dz++) {
        qnorm = dx*dx+dy*dy+dz*dz;
        if (qnorm>distmax) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
        psite              = dx + dy*pvol1 + dz*pvol2;
        g                  = vec_weight[psite];
        vec_factor[qnorm] += g;
    }
    for (int dx=RSxmax;dx<RSonmax;dx++) {
        qnorm = dx*dx;
        if (qnorm>distmax) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
        /// Here there are always 3 unique vectors corresponding to 3 spatial directions.
        g                  = 3;
        vec_factor[qnorm] += g;
    }
};

#endif //CORRELATORS_H
