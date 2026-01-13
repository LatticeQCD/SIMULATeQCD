#include "eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/math/random.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../modules/dslash/dslash.h"
#include <fstream>

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::fillRandom(const int &num_vec_in) {
    // Setup
    lambda_vec.clear();
    spinor_vec.clear();
    spinor_count = num_vec_in;
    spinor_vec.reserve(spinor_count);
    lambda_vec.reserve(spinor_count);
    CommunicationBase &commBase = this->getComm();

    // Allocate vectors
    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor_host(commBase);

    // Initialize q with random Gaussian
    grnd_state<false> h_rand;
    h_rand.make_rng_state(1234);

    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int n = 0; n < spinor_count; n++) {
            spinor_host.gauss(h_rand.state);
            // spinor_vec.emplace_back(spinor_host);
            spinor_vec.emplace_back(commBase);
            spinor_vec[n] = spinor_host;
            lambda_vec.emplace_back(static_cast<double>(get_rand<floatT>(h_rand.state)));
        }
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::writeEigenpairsSequential(const std::string &fname, int diskprec, Endianness en) 
{    
    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    CommunicationBase &commBase = this->getComm();

    EigenFormat<HaloDepthSpin> evnersc(commBase);

    int even_len = spinor_count - (spinor_count % 2);

    std::ofstream out;
    if (commBase.IamRoot()) {
        out.open(fname.c_str());
        if (!out.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname));
        }
    }
    // Write header
    if (!evnersc.template write_header<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>(diskprec, spinor_count, en, out)) {
        rootLogger.error("Unable to write EVNERSC file: ", fname);
        return;
    }
    // Write Eigenvalues
    if (commBase.IamRoot()) {
        if constexpr (LatticeLayout == Layout::All) {
            // TODO
        }   else {
            // Write eigenvalues
            for (int i = 0; i < even_len; ++i) {
                out.write(reinterpret_cast<const char*>(&lambda_vec[i]), sizeof(double));
            }
        }

        out.close();
    }

    size_t spinor_size = GInd::getLatData().globvol4 * evnersc.bytes_per_site();
    size_t displacement = sizeof(double) * spinor_count + evnersc.header_size();
    size_t file_size = spinor_size * spinor_count + displacement;

    commBase.initIOBinary(fname, file_size, evnersc.bytes_per_site(), displacement, global, local, WRITE);
    
    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, NStacks> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, NStacks> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, NStacks> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, NStacks> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    for (int n = 0; n < even_len; n+=2) {
        spinor = spinor_vec[n];
        spinor_even.iterateOverBulk(
            returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, NStacks>(spinor)
        );
        spinor_split.even = spinor_even;

        spinor = spinor_vec[n+1];
        spinor_odd.iterateOverBulk(
            returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, NStacks>(spinor)
        );
        spinor_split.odd = spinor_odd;

        spinor_host = spinor_split;

        for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
        for (size_t x = 0; x < GInd::getLatData().lx; x++) {
            gSite site = GInd::getSite(x,y,z,t);
            Vect3<floatT> tmp = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            evnersc.put_vector(tmp);

            if (evnersc.end_of_buffer()) {
                evnersc.process_write_data();
                commBase.writeBinary(evnersc.buf_ptr(), GInd::getLatData().vol4);
            }
        }
    }
    commBase.closeIOBinary();
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEigenpairsSequential(const std::string &fname) 
{
    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    CommunicationBase &commBase = this->getComm();

    EigenFormat<HaloDepthSpin> evnersc(commBase);

    std::ifstream in;
    in.open(fname.c_str());

    if (commBase.IamRoot()) {
        if (!in.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname));
        }
    }
    // Read header
    if (!evnersc.read_header(in)) {
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }
    spinor_count = evnersc.spinor_count();
    int even_len = spinor_count - (spinor_count % 2);
    rootLogger.info("Reading ", fname, " with ", spinor_count, " spinors");
    // Read Eigenvalues
    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        lambda_vec.reserve(spinor_count);
        if (commBase.IamRoot()) {
            lambda_vec.clear();
            for (int i = 0; i < even_len; ++i) {
                double lambda;
                in.read(reinterpret_cast<char*>(&lambda), sizeof(double));
                lambda_vec.emplace_back(lambda);
            }
        }
            

        if (!commBase.single()) {
            commBase.root2all(lambda_vec);
        }
    }

    in.close();


    size_t displacement = sizeof(double) * spinor_count + evnersc.header_size();

    commBase.initIOBinary(fname, 0, evnersc.bytes_per_site(), displacement, global, local, READ);

    spinor_vec.clear();

    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, NStacks> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, NStacks> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, NStacks> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, NStacks> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    for (int n = 0; n < even_len; n+=2) {
        for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
        for (size_t x = 0; x < GInd::getLatData().lx; x++) {
            if (evnersc.end_of_buffer()) {
                rootLogger.info("Reading spinor", n, " data from file ", fname.c_str());
                commBase.readBinary(evnersc.buf_ptr(), GInd::getLatData().vol4);
                evnersc.process_read_data();
            }

            Vect3<floatT> tmp = evnersc.template get_vector<floatT>();
            gSite site = GInd::getSite(x,y,z,t);
            spinor_accessor.setElement(GInd::getSiteMu(site, 0), tmp);
        }

        spinor_split = spinor_host;

        spinor_even = spinor_split.even;
        spinor.iterateOverBulk(
            returnSpinor<floatT, false, Even, HaloDepthSpin, NStacks>(spinor_even)
        );
        spinor_vec.emplace_back(commBase);
        spinor_vec[n] = spinor;

        spinor_odd = spinor_split.odd;
        spinor.iterateOverBulk(
            returnSpinor<floatT, false, Odd, HaloDepthSpin, NStacks>(spinor_odd)
        );
        spinor_vec.emplace_back(commBase);
        spinor_vec[n+1] = spinor;
    }
    commBase.closeIOBinary();
}



template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::tester(Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge) 
{    
    CommunicationBase &commBase = this->getComm();
    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);

    HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);


    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int i = 0; i < spinor_count; i++) {
            Spinor_t &spinorIn = spinor_vec[i];
            Spinor_t vr(commBase);
            
            floatT lambda = lambda_vec[i];
            rootLogger.info("tester:lambda=", lambda);
            
            vr = spinorIn;
            // spinorIn.updateAll();
            // vr.updateAll();
            
            Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor_host(commBase);
            spinor_host = spinorIn;
            Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

            gSite site = GInd::getSite(0,0,0,0);
            Vect3<floatT> vec31 = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            rootLogger.info("tester:Spinor element at 0,0,0,0", vec31.getElement0(), vec31.getElement1(), vec31.getElement2());
            
            site = GInd::getSite(
                GInd::getLatData().lx-1,
                GInd::getLatData().ly-1,
                GInd::getLatData().lz-1,
                GInd::getLatData().lt-1
            );
            vec31 = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            rootLogger.info("tester:Spinor element at last:", vec31.getElement0(), vec31.getElement1(), vec31.getElement2());
            
            // this->updateAll(All);
            dslash.applyMdaggM(vr, spinorIn, true);


            vr.template axpyThisB<64>(lambda, spinorIn);
            rootLogger.info("tester:norm(Ax-µx)**2=", vr.realdotProduct(vr));
        }
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVector(double mass, Spinor_t& spinorOut, const Spinor_t& spinorIn) {
    CommunicationBase &commBase = this->getComm();
    Spinor_t spinorSum(commBase);
    Spinor_t spinorEv(commBase);

    double lambda;
    COMPLEX(double) factorDouble;
    COMPLEX(floatT) factorCompat;
    

    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int i = 0; i < spinor_count; i++) {
            spinorEv = spinor_vec[i];
            lambda = mass*mass + lambda_vec[i];

            factorDouble =  spinorEv.dotProduct(spinorIn);

            factorDouble /= lambda;

            factorCompat = GPUcomplex<floatT>(real(factorDouble), imag(factorDouble));

            spinorSum.template axpyThisB<64>(factorCompat, spinorEv);
        }
        spinorOut = spinorSum;
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVectorTester(LinearOperator<Spinor_t>& dslash, const Spinor_t& spinorStart, const Spinor_t& spinorRHS) {
    CommunicationBase &commBase = this->getComm();



    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int i = 0; i < spinor_count; i++) {
            Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorIn = spinor_vec[i];
            Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(commBase);
            
            floatT lambda = lambda_vec[i];
            rootLogger.info("startVectorTester:lambda=", lambda);
            
            vr = spinorIn;
            
            Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor_host(commBase);
            spinor_host = spinorIn;
            Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

            gSite site = GInd::getSite(0,0,0,0);
            Vect3<floatT> vec31 = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            rootLogger.info("startVectorTester:Spinor element at 0,0,0,0", vec31.getElement0(), vec31.getElement1(), vec31.getElement2());
            
            site = GInd::getSite(
                GInd::getLatData().lx-1,
                GInd::getLatData().ly-1,
                GInd::getLatData().lz-1,
                GInd::getLatData().lt-1
            );
            vec31 = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            rootLogger.info("startVectorTester:Spinor element at last:", vec31.getElement0(), vec31.getElement1(), vec31.getElement2());
            
            dslash.applyMdaggM(vr, spinorIn, true);


            vr.template axpyThisB<64>(lambda, spinorIn);
            rootLogger.info("startVectorTester:norm(Ax-µx)**2=", vr.realdotProduct(vr));
        }
    }

    Spinor_t vr(commBase);
    Spinor_t va(commBase);
    va = spinorRHS;

    dslash.applyMdaggM(vr, spinorStart, true);
    
    COMPLEX(double) sum0 = vr.dotProduct(vr)-va.dotProduct(vr);
    rootLogger.info("startVectorTester:0=", sum0);

    COMPLEX(double) sum1 = va.dotProduct(vr);
    
    for (int i =0; i < spinor_count; i++) {
        Spinor_t &spinorEv = spinor_vec[i];
        vr = spinorEv;
        sum1 -= va.dotProduct(vr) * vr.dotProduct(va);    
    }
    rootLogger.info("startVectorTester:1=", sum1);
}


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

