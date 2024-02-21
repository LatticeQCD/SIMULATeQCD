#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, size_t HaloDepth>
void new_eigenpairs<floatT, onDevice, HaloDepth>::readconf_evnersc(const std::string &fname) {

    if(onDevice) {
        rootLogger.info("readconf_evnersc: Reading NERSC configuration ",fname);
        gVect3array<floatT, false> eigenvector_host;
        readconf_evnersc_host(eigenvector_host.getAccessor(), fname);
        eigenvectors.copyFrom(eigenvector_host);

    } else {
        readconf_evnersc_host(getAccessor(), fname);
    }
    this->su3latunitarize();
}
