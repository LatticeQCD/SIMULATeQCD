#pragma once

#include "../base/math/operators.h"
#include "../define.h"
#include "../base/math/vect3array.h"
#include "../base/latticeContainer.h"
#include "../base/communication/siteComm.h"
#include "../base/communication/communicationBase.h"
#include "spinorfield.h"

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
    void tester(int nvec);
    
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
