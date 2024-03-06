#pragma once

#include "../base/math/operators.h"
#include "../define.h"
#include "../base/math/gvect3array.h"
#include "../base/LatticeContainer.h"
#include "../base/communication/siteComm.h"
#include "../base/communication/communicationBase.h"
#include "spinorfield.h"
// #include "../base/communication/siteComm.h"

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
class new_eigenpairs : public siteComm<floatT, onDevice, gVect3arrayAcc<floatT>, gVect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>
{
protected:
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> _lattice;
private:

    new_eigenpairs(const new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    explicit new_eigenpairs(CommunicationBase &comm) :
            siteComm<floatT, onDevice, gVect3arrayAcc<floatT>, gVect3<floatT>,3, NStacks, LatticeLayout, HaloDepth>(comm),
            _lattice(comm){}

    void readconf_evnersc(const std::string &fname);
    void readconf_evnersc_host(gVect3arrayAcc<floatT> gVect3arrayAcc, const std::string &fname);
    
    virtual gVect3arrayAcc<floatT> getAccessor() const;
};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
inline gVect3arrayAcc<floatT> new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::getAccessor() const {
    return (_lattice.getAccessor());
}


template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Nstacks>
struct returnEigen {
    gVect3arrayAcc<floatT> _gAcc;

    explicit returnEigen(const new_eigenpairs<floatT, onDevice, LatLayout, HaloDepth, Nstacks> &spinorIn);
    __host__ __device__ gVect3<floatT> operator()(gSiteStack site);
};
