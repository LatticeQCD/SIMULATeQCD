#include "new_eigenpairs.h"
#include "../base/IO/evnersc.h"
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
    rootLogger.info("readconf_evnersc: Reading NERSC configuration ", fname);
    evNerscFormat<HaloDepth> evnersc(this->getComm());
    // typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;
    int nvec = NStacks;
    double lambda;
    float vec31[8];
    // float vec32[8];
    int sizeh=48*48*48*8/2;
    if (this->getComm().IamRoot()) {
        in.open(fname.c_str());
        for(int i=0; i<nvec-1; i++){
            in.read( (char*) &lambda, sizeof(double) );
            printf("lambda[%d]=%le\n",i,lambda);
            in.read( (char*) &vec31, sizeof(float)*8 );
            in.ignore( (sizeh-2)*sizeof(float)*8 );
            // in.read( (char*) &vec32, sizeof(float)*8 );
            printf("cvect3[%d]={{(%e,%e)(%e,%e)(%e,%e)}\n",i,vec31[0],vec31[1],vec31[2],vec31[3],vec31[4],vec31[5]);
            // printf("   ... {(%e,%e)(%e,%e)(%e,%e)}}\n",vec32[0],vec32[1],vec32[2],vec32[3],vec32[4],vec32[5]);
            printf("\n");
        }
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

