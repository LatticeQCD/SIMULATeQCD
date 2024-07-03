#pragma once

#include "../define.h"
#include "../base/math/operators.h"
#include "../base/math/vect3array.h"
#include "../base/latticeContainer.h"
#include "../base/communication/siteComm.h"
#include "../base/communication/communicationBase.h"
// #include "../modules/inverter/inverter.h"
#include "spinorfield.h"

/// Abstract base class for all kind of linear operators that shall enter the inversion
template <typename Vector>
class LinearOperator{
public:
    virtual void applyMdaggM(Vector&, const Vector&, bool update = true) = 0;
};


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
class eigenpairs : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>
{
protected:
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> _spinor_lattice;
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>> spinors;

private:

    eigenpairs(const eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    std::vector<double> lambda_vect;


    explicit eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepth>(comm),
            _spinor_lattice(comm){}

    void read_evnersc(int nvec, const std::string &fname);
    void read_evnersc_host(Vect3arrayAcc<floatT> Vect3arrayAcc, int nvec, double &lambda, const std::string &fname);
    void tester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>>& dslash, int nvec);
    void start_vector(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& spinorIn);
    
    virtual Vect3arrayAcc<floatT> getAccessor() const;
};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
inline Vect3arrayAcc<floatT> eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::getAccessor() const {
    return (_spinor_lattice.getAccessor());
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Nstacks>
struct returnEigen {
    Vect3arrayAcc<floatT> _gAcc;

    explicit returnEigen(const eigenpairs<floatT, onDevice, LatLayout, HaloDepth, Nstacks> &spinorIn);
    __host__ __device__ Vect3<floatT> operator()(gSiteStack site);
};
