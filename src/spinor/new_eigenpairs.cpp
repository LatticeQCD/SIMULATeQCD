#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/IO/nersc.h"
#include "../base/latticeParameters.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc(int nvec, const std::string &fname) {   
    if(onDevice) {
        Spinorfield<floatT, false, LatticeLayout, HaloDepth, NStacks> lattice_host(this->getComm());
        for (int n = 0; n < nvec; n++) {
            spinors.emplace_back(this->getComm());
            readconf_evnersc_host(lattice_host.getAccessor(), n, fname);
            spinors[n] = lattice_host;
        }
    } else {
        readconf_evnersc_host(getAccessor(), nvec, fname);
    }
    // rootLogger.info(_spinors[0]);
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
void new_eigenpairs<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::readconf_evnersc_host(gVect3arrayAcc<floatT> spinorAccessor, int idxvec, const std::string &fname)
{
    evNerscFormat<HaloDepth> evnersc(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    int sizeh=GInd::getLatData().globvol4/2;
    int displacement=evnersc.bytes_per_site()*sizeh+sizeof(double);            
    this->getComm().SetFileView(displacement * idxvec);

    std::ifstream in;
    if (this->getComm().IamRoot()) {
      in.open(fname.c_str());
    }
    in.ignore(displacement*idxvec);
    if (!evnersc.read_header(in)) {
      throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();


    this->getComm().initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.header_size(), global, local, READ);
    
    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++)
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        if (evnersc.end_of_buffer()) {
            this->getComm().readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
            evnersc.process_read_data();
        }
        gVect3<floatT> ret = evnersc.template get<floatT>();
        gSite site = GInd::getSite(x, y, z, t);
        spinorAccessor.setElement(GInd::getSiteMu(site, 0), ret);
    }

    // this->getComm().closeIOBinary();

    //    if (!evnersc.checksums_match()) {
    //       throw std::runtime_error(stdLogger.fatal("Error checksum!"));
    //    }
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

