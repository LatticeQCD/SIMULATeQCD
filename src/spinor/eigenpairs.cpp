#include "eigenpairs.h"
#include "../base/IO/evnersc.h"
#include "../base/math/random.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../modules/dslash/dslash.h"
#include "../modules/inverter/lanczos.h"
#include <fstream>
// #define BLOCKSIZE 64


// Write eigenpairs to file in EVNERSC format. The diskprec parameter specifies the precision to use when writing the eigenvalues and eigenvectors to disk, and can be either 32 or 64. The en parameter specifies the endianness to use when writing the eigenvalues and eigenvectors to disk, and can be either ENDIAN_LITTLE, ENDIAN_BIG, or ENDIAN_AUTO.
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::writeEigenpairsToFile(const std::string &fname, int diskprec, Endianness en) 
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
            for (int i = 0; i < spinor_count; ++i) {
                out.write(reinterpret_cast<const char*>(&lambda_vec[i]), sizeof(LambdaType));
            }
        }

        out.close();
    }

    size_t spinor_size = GInd::getLatData().globvol4 * evnersc.bytes_per_site(); // The size in bytes of a single spinor.
    size_t displacement = sizeof(LambdaType) * spinor_count + evnersc.header_size(); // The displacement in the file where the spinor data starts, which is after the header and the eigenvalues.
    size_t file_size = spinor_size * spinor_count + displacement; // Total file size is the size of the header plus the size of the eigenvalues plus the size of all the spinors.

    commBase.initIOBinary(fname, file_size, evnersc.bytes_per_site(), displacement, global, local, WRITE); // Initialize binary I/O for writing the spinor data to the file, with the correct displacement to account for the header and eigenvalues.

    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, 1> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, 1> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, 1> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, 1> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    for (int n = 0; n < spinor_count; n+=2) {
        spinor = spinor_vec[n];
        spinor_even.iterateOverBulk(returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, 1>(spinor)); // This will fill spinor_even with the even sites of the current spinor.
        spinor_split.even = spinor_even; // Now spinor_split contains the even sites of the n-th eigenvector, and the odd sites are still uninitialized.

        // If the number of spinors is odd, the last odd spinor will be a copy of the last even spinor. 
        if (n+1 < even_len) {
            spinor = spinor_vec[n+1];
        } else {
            spinor = spinor_vec[n]; 
        }

        spinor_odd.iterateOverBulk(returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, 1>(spinor)); // This will fill spinor_odd with the odd sites of the current spinor. If n+1 < even_len, this will be the (n+1)-th eigenvector, otherwise it will be a copy of the n-th eigenvector.
        spinor_split.odd = spinor_odd; // Now spinor_split contains the full spinor for the n-th eigenvector, and if n+1 < even_len, it contains the full spinor for the (n+1)-th eigenvector as well.

        spinor_host = spinor_split; // Now spinor_host contains the full spinor for the n-th eigenvector, and if n+1 < even_len, it contains the full spinor for the (n+1)-th eigenvector as well.

        for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
        for (size_t x = 0; x < GInd::getLatData().lx; x++) {
            gSite site = GInd::getSite(x,y,z,t);
            Vect3<floatT> tmp = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
            evnersc.put_vector(tmp); // This function should be called for each vector element of the spinor to be written to the file. It takes care of packing the vector data into the buffer in the correct format for EVNERSC, and also handles endianness conversion if necessary. After calling this function for all vector elements of the spinor, the buffer will be ready to be written to the file.

            if (evnersc.end_of_buffer()) {
                evnersc.process_write_data(); // This function should be called before writing to the file when the buffer is full. It processes the data in the buffer and prepares it for writing to the file.
                commBase.writeBinary(evnersc.buf_ptr(), GInd::getLatData().vol4); // This function writes the data in the buffer to the file. It should be called after evnersc.process_write_data() when the buffer is full, and also after the loop to write any remaining data in the buffer to the file.
            }
        }
    }
    commBase.closeIOBinary();
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEigenpairsFromFile(const std::string &fname) 
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
        lambda_vec.clear();
        lambda_vec.resize(spinor_count);
        if (commBase.IamRoot()) {
            for (int i = 0; i < spinor_count; ++i) {
                LambdaType lambda;
                in.read(reinterpret_cast<char*>(&lambda), sizeof(LambdaType));
                lambda_vec[i] = lambda;
            }
        }
            

        if (!commBase.single()) {
            commBase.root2all(lambda_vec);
        }
    }

    in.close();


    size_t displacement = sizeof(LambdaType) * spinor_count + evnersc.header_size();

    commBase.initIOBinary(fname, 0, evnersc.bytes_per_site(), displacement, global, local, READ);

    spinor_vec.clear();

    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, 1> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, 1> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, 1> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, 1> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    for (int n = 0; n < spinor_count; n+=2) {
        for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
        for (size_t x = 0; x < GInd::getLatData().lx; x++) {
            if (evnersc.end_of_buffer()) {
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
            returnSpinor<floatT, false, Even, HaloDepthSpin, 1>(spinor_even)
        );
        spinor_vec.emplace_back(commBase);
        spinor_vec[n] = spinor;

        // If the number of spinors is odd, the last odd spinor will be omitted.
        if (n+1 < even_len) {
            spinor_odd = spinor_split.odd;
            spinor.iterateOverBulk(
                returnSpinor<floatT, false, Odd, HaloDepthSpin, 1>(spinor_odd)
            );
            spinor_vec.emplace_back(commBase);
            spinor_vec[n+1] = spinor;
        }
    }
    commBase.closeIOBinary();
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::checkEigenValueEquation(LinearOperator<Spinor_external> &op, double mass, double tol) {
    CommunicationBase &commBase = this->getComm();
    
    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        if (onDevice) {
            for (int n = 0; n < spinor_count; n++) {

                // get eigenvector and eigenvalue
                Spinor_external spinorEv(commBase);
                double lambda;
                getEigenPair(spinorEv, lambda, n);

                // compute m^2 + lambda for this eigenpair
                SimpleArray<double, NStacks> minusMassLambdaArray(- mass*mass - lambda);


                // compute M†Mv
                Spinor_external spinorMdMx(commBase);
                op.applyMdaggM(spinorMdMx, spinorEv, true);

                // compute (M†M - (m^2+λ))v and its norm to check the eigenvalue equation
                SimpleArray<double, NStacks> norm2(0.0);
                spinorMdMx.axpyThisLoopd(minusMassLambdaArray, spinorEv, NStacks);
                norm2 = spinorMdMx.realdotProductStacked(spinorMdMx);

                if (commBase.IamRoot()) {
                    if (max(norm2) > tol*tol) {
                        rootLogger.error("Eigenpair ", n, " does not satisfy the eigenvalue equation within tolerance: ||M†Mv - (m^2+λ)v|| = ", sqrt(max(norm2)), " > ", tol);
                    } else {
                        rootLogger.info("Eigenpair ", n, " satisfies the eigenvalue equation within tolerance: ||M†Mv - (m^2+λ)v|| = ", sqrt(max(norm2)));
                    }
                }

            }
        }
    }
}


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
    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1> spinor_host(commBase);

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
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::lanczos(LinearOperator<Spinor_external> &op, const int &num_vec_in) {
    spinor_vec.clear();
    lambda_vec.clear();

    CommunicationBase &commBase = this->getComm();

    if constexpr (LatticeLayout == Layout::All) {
        throw std::runtime_error(stdLogger.fatal("Eigenpairs::lanczos is currently implemented only for Even/Odd layout"));
    } else {
        TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::compute(
            commBase,
            op,
            num_vec_in,
            spinor_vec,
            lambda_vec
        );

        spinor_count = static_cast<int>(spinor_vec.size());
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::lanczos(
    LinearOperator<Spinor_external> &op,
    const int &num_vec_in,
    const TRLanRestartParams &params) {
    // This overload is intentionally thin: Eigenpairs owns storage and file I/O,
    // while TRLanSpinorSolver owns the numerical algorithm. Passing params here
    // enables thick restart and either polynomial filter without changing
    // existing callers of lanczos(op, num_vec_in).
    spinor_vec.clear();
    lambda_vec.clear();

    CommunicationBase &commBase = this->getComm();

    if constexpr (LatticeLayout == Layout::All) {
        throw std::runtime_error(stdLogger.fatal("Eigenpairs::lanczos is currently implemented only for Even/Odd layout"));
    } else {
        TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::compute(
            commBase,
            op,
            num_vec_in,
            spinor_vec,
            lambda_vec,
            params
        );

        spinor_count = static_cast<int>(spinor_vec.size());
    }
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
