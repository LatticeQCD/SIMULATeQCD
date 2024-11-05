#include "eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/IO/nersc.h"
#include "../base/latticeParameters.h"
#include "../modules/hisq/hisqSmearing.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::read_evnersc(const std::string &fname) 
{   
    int nvec = param.num_toread_vectors();
    lambda_vect.reserve(nvec);
    double lambda_temp;
    if(onDevice) {
        Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> vector_host(this->getComm());
        for (int n = 0; n < nvec; n++) {
            spinors.emplace_back(this->getComm());
            read_evnersc_host(vector_host.getAccessor(), n, lambda_temp, fname);
            spinors[n] = vector_host;
            lambda_vect[n] = lambda_temp;
        }
    } 
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::read_evnersc_host(Vect3arrayAcc<floatT> spinorAccessor, int idxvec, double &lambda, const std::string &fname)
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


// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
// void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::tester(int argc, char **argv) 
// {    
//     CommunicationBase commBase(&argc, &argv);
//     // const size_t HaloDepthGauge = 2;
//     // gauge.readconf_nersc(param.GaugefileName());
//     // gauge.updateAll();

//     // Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared(comm);
//     // // Gaugefield<floatT, onDevice, HaloDepthGauge, U3R18> gauge_Naik(comm);
//     // // HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
//     // // smearing.SmearAll();
//     // // HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

//     // for (int i = 0; i < param.num_toread_vectors(); i++) {
//     //     // Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorIn = spinors[i];
//     //     // Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorIn.getComm());
        
//     //     // floatT lambda = lambda_vect[i];
//     //     // rootLogger.info("lambda=", lambda);
//     //     // // rootLogger.info("lambda**2=", lambda * lambda);
        
//     //     // vr = spinorIn;
//     //     // // rootLogger.info("norm(x)**2=", vr.realdotProduct(vr));
        
//     //     // dslash.applyMdaggM(vr, spinorIn, true);
//     //     // // rootLogger.info("norm(Ax)**2=", vr.realdotProduct(vr));

//     //     // vr.template axpyThisB<64>(lambda, spinorIn);
//     //     // rootLogger.info("norm(Ax-µx)**2=", vr.realdotProduct(vr));
//     // }
// }


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::start_vector(double mass, Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn) {
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorIn.getComm());
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(spinorIn.getComm());

    double lambda;
    COMPLEX(double) faktor_double;
    COMPLEX(floatT) faktor_compat;

    for (int i = 0; i < param.num_toread_vectors(); i++) {
        spinorEv = spinors[i];
        lambda = mass*mass - lambda_vect[i];

        faktor_double =  spinorEv.dotProduct(spinorIn);
        faktor_double  /= lambda;
        faktor_compat = GPUcomplex<floatT>(real(faktor_double), imag(faktor_double));
        vr.template axpyThisB<64>(faktor_compat, spinorEv);
    }
    spinorOut = vr;
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::start_vector_tester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS) {
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorStart.getComm());
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> va(spinorRHS.getComm());
    va = spinorRHS;

    dslash.applyMdaggM(vr, spinorStart, true);
    
    COMPLEX(double) sum0 = vr.dotProduct(vr)-va.dotProduct(vr);
    rootLogger.info("start_vector_tester0=", sum0);

    COMPLEX(double) sum1 = va.dotProduct(vr);
    
    for (int i =0; i < param.num_toread_vectors(); i++) {
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorEv = spinors[i];
        vr = spinorEv;
        sum1 -= va.dotProduct(vr) * vr.dotProduct(va);    
    }
    rootLogger.info("start_vector_tester1=", sum1);
}


template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, size_t Nstacks>
returnEigen<floatT, onDevice, LatLayout, HaloDepthSpin, Nstacks>::returnEigen(const eigenpairs<floatT, onDevice, LatLayout, HaloDepthSpin, Nstacks> &spinorIn) :
        _gAcc(spinorIn.getAccessor()) {
}

#define EIGEN_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class eigenpairs<floatT,false,LO,HALOSPIN,STACKS>;\
template struct returnEigen<floatT,false,LO,HALOSPIN,STACKS>;\


INIT_PLHSN(EIGEN_INIT_PLHSN)

#define EIGEN_INIT_PLHSN_HALF(floatT,LO,HALOSPIN,STACKS)\
template class eigenpairs<floatT,true,LO,HALOSPIN,STACKS>;\
template struct returnEigen<floatT,true,LO,HALOSPIN,STACKS>;\

INIT_PLHSN_HALF(EIGEN_INIT_PLHSN_HALF)

