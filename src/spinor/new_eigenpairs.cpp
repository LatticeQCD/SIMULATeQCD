#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/IO/nersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc(const std::string &fname) 
{   
    if(onDevice) {
        rootLogger.info("readconf_evnersc: Reading NERSC configuration ", fname);
        Spinorfield<floatT, false, LatticeLayout, HaloDepth, NStacks> lattice_host(this->getComm());
        readconf_evnersc_host(lattice_host.getAccessor(), fname);
        _lattice.copyFromStackToStack(lattice_host, NStacks, NStacks);
    } else {
        readconf_evnersc_host(getAccessor(), fname);
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc_host(gVect3arrayAcc<floatT>, const std::string &fname)
{
    NerscFormat<HaloDepth> nersc(this->getComm());
    // typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;

    int sizeh=nersc.bytes_per_site();
    rootLogger.info("sizeh ", sizeh);

    if (this->getComm().IamRoot()) {
        in.open(fname.c_str());
        // for(int i=0; i<8-1; i++){
        //     in.ignore(sizeof(double));
        //     in.read( (char*) &vec, sizeof(float)*8 );
        // }
        in.close();
    }
}


template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Nstacks>
returnEigen<floatT, onDevice, LatLayout, HaloDepth, Nstacks>::returnEigen(const new_eigenpairs<floatT, onDevice, LatLayout, HaloDepth, Nstacks> &spinorIn) :
        _gAcc(spinorIn.getAccessor()) {
}

#define EIGEN_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class new_eigenpairs<floatT,false,LO,HALOSPIN,STACKS>;\
template struct returnEigen<floatT,false,LO,HALOSPIN,STACKS>;\


INIT_PLHSN(EIGEN_INIT_PLHSN)

#define EIGEN_INIT_PLHSN_HALF(floatT,LO,HALOSPIN,STACKS)\
template class new_eigenpairs<floatT,true,LO,HALOSPIN,STACKS>;\
template struct returnEigen<floatT,true,LO,HALOSPIN,STACKS>;\

INIT_PLHSN_HALF(EIGEN_INIT_PLHSN_HALF)

