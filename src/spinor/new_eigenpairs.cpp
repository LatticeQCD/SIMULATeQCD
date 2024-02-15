#include "new_eigenpairs.h"
#include "../base/IO/nersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, size_t HaloDepth>
void new_eigenpairs<floatT, onDevice, HaloDepth>::readconf_evnersc(const std::string &fname) {

    if(onDevice) {
        rootLogger.info("readconf_evnersc: Reading NERSC configuration ",fname);
        readconf_evnersc_host(fname);
    } else {
        readconf_evnersc_host(fname);
    }
    this->su3latunitarize();
}