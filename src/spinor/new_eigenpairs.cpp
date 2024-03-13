#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/IO/nersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc(int nvec, const std::string &fname) {   
    if(onDevice) {
        rootLogger.info("readconf_evnersc: Reading NERSC configuration ", fname);
        Spinorfield<floatT, false, LatticeLayout, HaloDepth, NStacks> lattice_host(this->getComm());
        readconf_evnersc_host(lattice_host.getAccessor(), nvec, fname);
        // _lattice.copyFromStackToStack(lattice_host, NStacks, NStacks);
    } else {
        readconf_evnersc_host(getAccessor(), nvec, fname);
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc_host(gVect3arrayAcc<floatT>, int nvec, const std::string &fname)
{
    evNerscFormat<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> evnersc(this->getComm());
    // typedef GIndexer<All,HaloDepth> GInd;

    double vec[8];

    int sizeh=evnersc.bytes_per_site();
    rootLogger.info("sizeh ", sizeh);

    std::ifstream in;

    evnersc.read_header(in);

    if (this->getComm().IamRoot()) {
        in.open(fname.c_str());
        for(int i=0; i<nvec; i++){
            in.ignore(sizeof(double));
            in.read( (char*) &vec, sizeof(float)*8 );
            rootLogger.info(vec[0]);
        }
    }



    // LatticeDimensions global = GInd::getLatData().globalLattice();
    // LatticeDimensions local = GInd::getLatData().localLattice();

    // this->getComm().initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.header_size(), global, local, READ);
    
    // typedef GIndexer<All, HaloDepth> GInd;
    // for (size_t t = 0; t < GInd::getLatData().lt; t++) {
    //     if (evnersc.end_of_buffer()) {
    //         this->getComm().readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
    //         evnersc.process_read_data();
    //     }
    //     for (int mu = 0; mu < 4; mu++) {
    //         Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> ret = evnersc.template get<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>();
    //         gSite site = GInd::getSite(x, y, z, t);
    //         gaugeAccessor.setLink(GInd::getSiteMu(site, mu), ret);
    //     }
    // }

    // this->getComm().closeIOBinary();

    if (!evnersc.checksums_match()) {
        throw std::runtime_error(stdLogger.fatal("Error checksum!"));
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

