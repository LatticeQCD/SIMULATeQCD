#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc(const std::string &fname) {

    if(onDevice) {
        rootLogger.info("readconf_evnersc: Reading NERSC configuration ",fname);
        new_eigenpairs<floatT, false, LatticeLayout, HaloDepth, NStacks> lattice_host(comm);
        readconf_evnersc_host(lattice_host.getAccessor(), fname);
        _lattice.copyFromStackToStack(lattice_host, onDevice, LatticeLayout, HaloDepth, NStacks);

    } else {
        readconf_evnersc_host(getAccessor(), fname);
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc_host(gVect3arrayAcc<floatT> gVect3arrayAcc, const std::string &fname)
{
    evNerscFormat<HaloDepth> evnersc(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;
    if (this->getComm().IamRoot()) {
        in.open(fname.c_str());
    }
    if (!evnersc.read_header(in)) {
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    if (!evnersc.checksums_match()) {
        throw std::runtime_error(stdLogger.fatal("Error checksum!"));
    }
}


// template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Nstacks>
// returnEigen<floatT, onDevice, LatLayout, HaloDepth, Nstacks>::returnEigen(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, Nstacks> &spinorIn) :
//         _gAcc(spinorIn.getAccessor()) {
// }

#define EIGEN_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class new_eigenpairs<floatT,false,LO,HALOSPIN,STACKS>;\
// template struct returnSpinor<floatT,false,LO,HALOSPIN,STACKS>;\

INIT_PLHSN(EIGEN_INIT_PLHSN)

#define EIGEN_INIT_PLHSN_HALF(floatT,LO,HALOSPIN,STACKS)\
template class new_eigenpairs<floatT,true,LO,HALOSPIN,STACKS>;\
// template struct returnSpinor<floatT,true,LO,HALOSPIN,STACKS>;\

INIT_PLHSN_HALF(EIGEN_INIT_PLHSN_HALF)

