#include "eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/math/random.h"
// #include "../modules/hisq/hisqSmearing.h"
#include <fstream>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::fillRandom(const int &num_vec_in) {
    lambda_vec.clear();
    spinor_vec.clear();
    vector_len = num_vec_in;
    lambda_vec.resize(vector_len);

    grnd_state<false> h_rand;
    h_rand.make_rng_state(1234);

    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor_host(this->getComm());

    if constexpr (LatticeLayout == Layout::All) {
        static std::mt19937 gen(1234);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int n = 0; n < vector_len; n++) {
            spinor_host.gauss(h_rand.state);

            spinor_vec.emplace_back(this->getComm());
            spinor_vec[n] = spinor_host;
            lambda_vec[n][0] = dist(gen);
            lambda_vec[n][1] = dist(gen);
        }
    }   else {
        static std::mt19937 gen(1234);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int n = 0; n < vector_len; n++) {
            spinor_host.gauss(h_rand.state);

            spinor_vec.emplace_back(this->getComm());
            spinor_vec[n] = spinor_host;
            lambda_vec[n] = dist(gen);
        }
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::writeEigenpairsFile(const std::string &fname, int diskprec, Endianness en) 
{   
    if (onDevice) {
        writeEigenpairsFileHost(fname, diskprec, en);
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::writeEigenpairsFileHost(const std::string &fname, int diskprec, Endianness en) 
{   
    EigenFormat<HaloDepthSpin> evnersc(this->getComm());

    if (this->getComm().IamRoot()) {
        std::ofstream out(fname.c_str());
        if (!out.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname.c_str()));
        }
        if (!evnersc.template write_header<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>(diskprec, en, out)) {
            throw std::runtime_error(stdLogger.fatal("Error writing header of ", fname.c_str()));
        }
        out.close();
    }

    // typedef GIndexer<All, HaloDepthSpin> GInd;
    // LatticeDimensions global = GInd::getLatData().globalLattice();
    // LatticeDimensions local = GInd::getLatData().localLattice();

    // this->getComm().initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.header_size(), global, local, WRITE);
    
    // Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor(this->getComm());
    // Spinorfield<floatT, false, Even, HaloDepthSpin, NStacks> spinor_even(this->getComm());
    // Spinorfield<floatT, false, Odd, HaloDepthSpin, NStacks> spinor_odd(this->getComm());
    // Spinorfield<floatT, false, All, HaloDepthSpin, NStacks> spinor_host(this->getComm());
    // SpinorfieldAll<floatT, false, HaloDepthSpin, NStacks> spinor_split(this->getComm());
    // Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    // if constexpr (LatticeLayout == Layout::All) {
    //     // TODO
    // }   else {
    //     for (int n = 0; n < vector_len; n+=2) {
            
    //         floatT tmp = lambda_vec[n];
    //         evnersc.put_scalar(tmp);
    //         tmp = lambda_vec[n+1];
    //         evnersc.put_scalar(tmp);

    //         spinor = spinor_vec[n];
    //         spinor_even.iterateOverBulk(
    //             returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, NStacks>(spinor)
    //         );
    //         spinor_split.even = spinor_even;

    //         spinor = spinor_vec[n+1];
    //         spinor_odd.iterateOverBulk(
    //             returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, NStacks>(spinor)
    //         );
    //         spinor_split.odd = spinor_odd;

    //         spinor_host = spinor_split;

    //         for (size_t t = 0; t < GInd::getLatData().lt; t++)
    //         for (size_t z = 0; z < GInd::getLatData().lz; z++)
    //         for (size_t y = 0; y < GInd::getLatData().ly; y++)
    //         for (size_t x = 0; x < GInd::getLatData().lx; x++) {
    //             gSite site = GInd::getSite(x,y,z,t);
    //             Vect3<floatT> tmp = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
    //             evnersc.put_vector(tmp);

    //             if (evnersc.end_of_buffer()) {
    //                 evnersc.process_write_data();
    //                 this->getComm().writeBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
    //                 rootLogger.info("Writing spinor", n, " data to file ", fname.c_str());
    //             }
    //         }
    //     }
    // }
    // this->getComm().closeIOBinary();
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEigenpairsFile(const std::string &fname) 
{
    if(onDevice) {
        readEigenpairsFileHost(fname);
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEigenpairsFileHost(const std::string &fname) 
{
    EigenFormat<HaloDepthSpin> evnersc(this->getComm());

    if (this->getComm().IamRoot()) {
        std::ifstream in(fname.c_str());
        if (!in.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname.c_str()));
        }
        if (!evnersc.read_header(in)) {
            throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
        }
        in.close();
    }
}

// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
// void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEvNersc(const std::string &fname, const int &num_vec_in) 
// {   
//     lambda_vec.clear();
//     spinor_vec.clear();
//     vector_len = num_vec_in;
//     lambda_vec.resize(vector_len);

//     Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor_host(this->getComm());
//     double lambda_host;

//     if constexpr (LatticeLayout == Layout::All) {
//         // TODO
//     }   else {
//         for (int n = 0; n < vector_len; n++) {
//             spinor_vec.emplace_back(this->getComm());
//             readEvNerscHost(spinor_host.getAccessor(), lambda_host, fname, n);
//             spinor_vec[n] = spinor_host;
//             lambda_vec[n] = lambda_host;
//         }
//     }
// }

// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
// void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEvNerscHost(Vect3arrayAcc<floatT> spinor_accessor, double &lambda, const std::string &fname, int vector_idx)
// {
//     EigenFormat<HaloDepthSpin> evnersc(this->getComm());
//     typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

//     int sizeh=GInd::getLatData().sizeh;
//     int displacement_local=(evnersc.bytes_per_site()*sizeh+sizeof(double))*vector_idx;
//     this->getComm().SetFileView(displacement_local);

//     std::ifstream in;
//     if (this->getComm().IamRoot()) {
//       in.open(fname.c_str());
//     }
//     // in.ignore(displacement_local);

//     if (!evnersc.read_double(in, lambda)) {
//       throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
//     }

//     LatticeDimensions global = GInd::getLatData().globalLattice();
//     LatticeDimensions local = GInd::getLatData().localLattice();


//     this->getComm().initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.displacement(), global, local, READ);

//     // for (int m = 0; m < sizeh; m++)  {
//     //     if (true)  {
//     //         sitexyzt coord = GInd.indexToCoord(m);
//     //         gSite site = GInd::getSite(coord);
//     for (size_t t = 0; t < GInd::getLatData().lt; t++)
//     for (size_t z = 0; z < GInd::getLatData().lz; z++)
//     for (size_t y = 0; y < GInd::getLatData().ly; y++)
//     for (size_t x = 0; x < GInd::getLatData().lx; x++) {
//         if ((x+y+z+t)%2==0){
//             gSite site = GInd::getSite(x,y,z,t);

//             if (evnersc.end_of_buffer()) {
//                 this->getComm().readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
//                 evnersc.process_read_data();
//             }
//             Vect3<floatT> ret = evnersc.template get<floatT>();
//             spinor_accessor.setElement(GInd::getSiteMu(site, 0), ret);
//         }
//     }
// }


// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
// void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::tester(CommunicationBase &commBase, Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge) 
// {    
//     Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared(commBase);
//     Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik(commBase);
//     HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
//     // smearing.SmearAll();

//     HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);


//     if constexpr (LatticeLayout == Layout::All) {
//         // TODO
//     }   else {
//         for (int i = 0; i < vector_len; i++) {
//             Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorIn = spinor_vec[i];
//             Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorIn.getComm());
            
//             floatT lambda = lambda_vec[i];
//             rootLogger.info("lambda=", lambda);
            
//             vr = spinorIn;
            
//             dslash.applyMdaggM(vr, spinorIn, true);

//             vr.template axpyThisB<64>(lambda, spinorIn);
//             rootLogger.info("norm(Ax-µx)**2=", vr.realdotProduct(vr));
//         }
//     }
// }


// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
// void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVector(double mass, Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn) {
//     Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorSum(spinorIn.getComm());
//     Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(spinorIn.getComm());

//     double lambda;
//     COMPLEX(double) factorDouble;
//     COMPLEX(floatT) factorCompat;
    

//     if constexpr (LatticeLayout == Layout::All) {
//         // TODO
//     }   else {
//         for (int i = 0; i < vector_len; i++) {
//             spinorEv = spinor_vec[i];
//             lambda = mass*mass + lambda_vec[i];

//             factorDouble =  spinorEv.dotProduct(spinorIn);

//             factorDouble /= lambda;

//             factorCompat = GPUcomplex<floatT>(real(factorDouble), imag(factorDouble));

//             spinorSum.template axpyThisB<64>(factorCompat, spinorEv);
//         }
//         spinorOut = spinorSum;
//     }
// }


// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
// void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVectorTester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS) {
//     Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(spinorStart.getComm());
//     Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> va(spinorRHS.getComm());
//     va = spinorRHS;

//     dslash.applyMdaggM(vr, spinorStart, true);
    
//     COMPLEX(double) sum0 = vr.dotProduct(vr)-va.dotProduct(vr);
//     rootLogger.info("start_vector_tester0=", sum0);

//     COMPLEX(double) sum1 = va.dotProduct(vr);
    
//     for (int i =0; i < vector_len; i++) {
//         Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorEv = spinor_vec[i];
//         vr = spinorEv;
//         sum1 -= va.dotProduct(vr) * vr.dotProduct(va);    
//     }
//     rootLogger.info("start_vector_tester1=", sum1);
// }


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
returnEigen<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::returnEigen(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &spinorIn) :
        _gAcc(spinorIn.getAccessor()) {
}

#define EIGEN_INIT_PLHHSN(floatT,LO,HaloDepth, HaloDepthSpin,STACKS)\
template class Eigenpairs<floatT,false,LO,HaloDepth, HaloDepthSpin,STACKS>;\
template struct returnEigen<floatT,false,LO,HaloDepth, HaloDepthSpin,STACKS>;
INIT_PLHHSN(EIGEN_INIT_PLHHSN)

#define EIGEN_INIT_PLHHSN_HALF(floatT,LO,HaloDepth, HaloDepthSpin,STACKS)\
template class Eigenpairs<floatT,true,LO,HaloDepth, HaloDepthSpin,STACKS>;\
template struct returnEigen<floatT,true,LO,HaloDepth, HaloDepthSpin,STACKS>;
INIT_PLHHSN_HALF(EIGEN_INIT_PLHHSN_HALF)

