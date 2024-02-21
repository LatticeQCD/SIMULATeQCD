#pragma once

#include "../define.h"
#include "../base/math/operators.h"
#include "../base/math/gvect3array.h"
#include "../base/communication/siteComm.h"

template<class floatT, bool onDevice, size_t HaloDepth>
class new_eigenpairs
{
protected:
    gVect3array<floatT, onDevice> eigenvectors;
private:

    new_eigenpairs(const new_eigenpairs<floatT, onDevice, HaloDepth> &glat) = delete;

public:

    explicit new_eigenpairs(CommunicationBase &comm, std::string gVect3arrayName="gVect3array")
            : siteComm<floatT, onDevice, gaugeAccessor<floatT,comp>, GSU3<floatT>,EntryCount<comp>::count, 4, All, HaloDepth>(comm),
              _lattice(GInd::getLatData().vol4Full * 4, gVect3arrayName){
    }

    void readconf_evnersc(const std::string &fname);
    void readconf_evnersc_host(gVect3arrayAcc<floatT> gVect3arrayAcc, const std::string &fname);


};
