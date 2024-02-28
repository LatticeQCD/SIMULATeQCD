#pragma once

#include "../define.h"
#include "../base/math/operators.h"
// #include "../base/math/gsu3array.h"
// #include "../base/math/gvect3array.h"
#include "spinorfield.h"
#include "../base/communication/siteComm.h"

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks = 1>
class new_eigenpairs : public siteComm<floatT, onDevice, gVect3arrayAcc<floatT>, gVect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>
{
protected:
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> _lattice;
private:
    LatticeContainer<onDevice,GCOMPLEX(double)> _redBase;
    new_eigenpairs(const new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>&) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    // explicit new_eigenpairs(CommunicationBase &comm, std::string eigenpairsName="eigenpairsName")
    //         : siteComm<floatT, onDevice, gVect3arrayAcc<floatT>, gVect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>(comm),
    //         _lattice( (int)(NStacks*( (LatticeLayout == All) ? GInd::getLatData().vol4Full : GInd::getLatData().sizehFull )), eigenpairsName),
    //         _redBase(comm)
    // {
    //     if (LatticeLayout == All){
    //         _redBase.adjustSize(GIndexer<LatticeLayout, HaloDepth>::getLatData().vol4 * NStacks);
    //     }else{
    //         _redBase.adjustSize(GIndexer<LatticeLayout, HaloDepth>::getLatData().vol4 * NStacks / 2);
    //     }
    // }

    void readconf_evnersc(const std::string &fname);
    void readconf_evnersc_host(gVect3arrayAcc<floatT> gVect3arrayAcc, const std::string &fname);

    virtual gVect3arrayAcc<floatT> getAccessor() const;
};
