#include "eigenpairs.h"
#include "../base/IO/evnersc.h"
// #include "../base/IO/nersc.h"
// #include "../base/latticeParameters.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../modules/dslash/dslash.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::read_evnersc(const int &numVecIn, const std::string &fname) 
{   
    numVec = numVecIn;
    lambda_vect.reserve(numVec);
    double lambda_temp;
    if(onDevice) {
        Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> vector_host(this->getComm());
        for (int n = 0; n < numVec; n++) {
            spinors.emplace_back(this->getComm());
            read_evnersc_host(vector_host.getAccessor(), n, lambda_temp, fname);
            spinors[n] = vector_host;
            lambda_vect[n] = lambda_temp;
        }
    } 
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::read_evnersc_host(Vect3arrayAcc<floatT> spinorAccessor, int idxvec, double &lambda, const std::string &fname)
{
    evNerscFormat<HaloDepthSpin> evnersc(this->getComm());
    typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    int sizeh=GInd::getLatData().sizeh;
    int displacement=evnersc.bytes_per_site()*sizeh+sizeof(double);            
    this->getComm().SetFileView(displacement * idxvec);

    std::ifstream in;
    if (this->getComm().IamRoot()) {
      in.open(fname.c_str());
    }
    in.ignore(displacement*idxvec);

    if (!evnersc.read_header(in, lambda)) {
      throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();


    this->getComm().initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.header_size(), global, local, READ);

    // for (int m = 0; m < sizeh; m++)  {
    //     if (true)  {
    //         sitexyzt coord = GInd.indexToCoord(m);
    //         gSite site = GInd::getSite(coord);
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++)
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        if ((x+y+z+t)%2==0){
            gSite site = GInd::getSite(x,y,z,t);

            if (evnersc.end_of_buffer()) {
                this->getComm().readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
                evnersc.process_read_data();
            }
            Vect3<floatT> ret = evnersc.template get<floatT>();
            spinorAccessor.setElement(GInd::getSiteMu(site, 0), ret);
    
        }
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::tester(CommunicationBase &commBase, Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge) 
{    
    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
    // smearing.SmearAll();

    HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

    for (int i = 0; i < numVec; i++) {
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorIn = spinors[i];
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorIn.getComm());
        
        floatT lambda = lambda_vect[i];
        rootLogger.info("lambda=", lambda);
        
        vr = spinorIn;
        
        dslash.applyMdaggM(vr, spinorIn, true);

        vr.template axpyThisB<64>(lambda, spinorIn);
        rootLogger.info("norm(Ax-µx)**2=", vr.realdotProduct(vr));
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::start_vector(double mass, Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn) {
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorIn.getComm());
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(spinorIn.getComm());
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> va(spinorIn.getComm());

    va = spinorIn;

    double lambda;
    COMPLEX(double) faktor_double;
    COMPLEX(floatT) faktor_compat;

    for (int i = 0; i < numVec; i++) {
        spinorEv = spinors[i];
        lambda = mass*mass + lambda_vect[i];


        faktor_double =  spinorEv.dotProduct(spinorIn);

        faktor_double /= lambda;

        faktor_compat = GPUcomplex<floatT>(real(faktor_double), imag(faktor_double));

        vr.template axpyThisB<64>(faktor_compat, spinorEv);
    }
    spinorOut = vr;
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::start_vector_tester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS) {
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorStart.getComm());
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> va(spinorRHS.getComm());
    va = spinorRHS;

    dslash.applyMdaggM(vr, spinorStart, true);
    
    COMPLEX(double) sum0 = vr.dotProduct(vr)-va.dotProduct(vr);
    rootLogger.info("start_vector_tester0=", sum0);

    COMPLEX(double) sum1 = va.dotProduct(vr);
    
    for (int i =0; i < numVec; i++) {
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorEv = spinors[i];
        vr = spinorEv;
        sum1 -= va.dotProduct(vr) * vr.dotProduct(va);    
    }
    rootLogger.info("start_vector_tester1=", sum1);
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
returnEigen<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::returnEigen(const eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &spinorIn) :
        _gAcc(spinorIn.getAccessor()) {
}

#define EIGEN_INIT_PLHHSN(floatT,LO,HaloDepth, HaloDepthSpin,STACKS)\
template class eigenpairs<floatT,false,LO,HaloDepth, HaloDepthSpin,STACKS>;\
template struct returnEigen<floatT,false,LO,HaloDepth, HaloDepthSpin,STACKS>;
INIT_PLHHSN(EIGEN_INIT_PLHHSN)

#define EIGEN_INIT_PLHHSN_HALF(floatT,LO,HaloDepth, HaloDepthSpin,STACKS)\
template class eigenpairs<floatT,true,LO,HaloDepth, HaloDepthSpin,STACKS>;\
template struct returnEigen<floatT,true,LO,HaloDepth, HaloDepthSpin,STACKS>;
INIT_PLHHSN_HALF(EIGEN_INIT_PLHHSN_HALF)

