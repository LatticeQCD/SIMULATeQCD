#pragma once

#include "../define.h"
#include "../base/math/operators.h"
#include "../base/gutils.h"
#include "../base/IO/misc.h"
#include "../base/communication/siteComm.h"

#define PREC double

template<class floatT, bool onDevice, size_t HaloDepth>
class new_eigenpairs
{
prorectet:
    Spinorfield
public:
    
    explicit new_eigenpairs(CommunicationBase &comm, std::string gaugefieldName="Gaugefield");

    void readconf_evnersc(const std::string &fname);
    void readconf_evnersc_host(const std::string &fname);

};
