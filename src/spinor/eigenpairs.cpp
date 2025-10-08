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
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::lanczos(Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge, const int &num_vec_in, const int &max_iter) {
    // Setup
    spinor_count = num_vec_in;
    lambda_vec.resize(spinor_count);
    // spinor_vec.resize(spinor_count);

    CommunicationBase &commBase = this->getComm();

    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);

    HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

    // Allocate vectors
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q_prev(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q_next(commBase);

    std::vector<std::vector<floatT>> H(max_iter, std::vector<floatT>(max_iter));
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>> lanczos_vecs;

    std::vector<floatT> alpha(max_iter+1, 0.0);
    std::vector<floatT> beta(max_iter+1, 0.0);

    // Initialize q with random Gaussian
    grnd_state<false> h_rand;
    h_rand.make_rng_state(1234);
    q.gauss(h_rand.state);

    // Normalize q
    floatT norm = sqrt(q.realdotProduct(q));
    q *= (1.0 / norm);

    lanczos_vecs.emplace_back(commBase);
    lanczos_vecs[0] = q;

    // Main Lanczos loop
    for (int k = 0; k < max_iter; ++k) {
        // Apply operator (e.g., D†D)
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> Aq(commBase);
        dslash.applyMdaggM(Aq, q, true);

        // Orthogonalize
        if (k > 0) {
            Aq.axpyThisB(-beta[k], q_prev);
        }
        alpha[k] = Aq.realdotProduct(q);
        Aq.axpyThisB(-alpha[k], q);

        // Compute beta
        beta[k+1] = sqrt(Aq.realdotProduct(Aq));
        if (beta[k+1] < 1e-12) break; // Converged

        // Prepare next vector
        q_prev = q;
        q = Aq;
        q *= (1.0 / beta[k+1]);

        lanczos_vecs.emplace_back(commBase);
        lanczos_vecs[k+1] = q;
    }

    // Build tridiagonal matrix T from alpha/beta
    for (int i = 0; i <= max_iter; ++i) {
        H[i][i] = alpha[i];
        if (i > 0) {
            H[i-1][i] = beta[i];
            H[i][i-1] = beta[i];
        }
    }

    // // QR algorithm to compute eigenvalues
    // int max_iter_qr = 1000; // Maximum number of iterations for QR algorithm
    // floatT tolerance = 1e-8;
    // std::vector<std::vector<floatT>> Q(max_iter+2, std::vector<floatT>(max_iter+1));
    // std::vector<std::vector<floatT>> R(max_iter+1, std::vector<floatT>(max_iter+2));

    // for (int k = 0; k < max_iter + 2; ++k) {
    //     Q[k][k] = 1.0;
    // }

    // for (int iter = 0; iter < max_iter_qr; ++iter) {
    //     std::copy(H.begin(), H.end(), R.begin());
    //     QRDecomposition(max_iter+1, R, Q);
    //     std::copy(Q.begin(), Q.end(), H.begin());

    //     // Check convergence
    //     floatT max_off_diagonal = 0.0;
    //     for (int i = 0; i < max_iter; ++i) {
    //         max_off_diagonal = std::max(max_off_diagonal, std::abs(H[i][i+1]));
    //     }
    //     if (max_off_diagonal < tolerance) break;
    // }

    // auto tmp = H[0][0];

    // // Extract eigenvalues and eigenvectors from H
    // for (int i = 0; i <= max_iter; ++i) {
    //     tmp = H[i][i];
    //     // lambda_vec[i] = tmp;
    //     // Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> eigenvector(commBase);
    //     // for (int j = 0; j <= max_iter; ++j) {
    //     //     eigenvector.axpyThisB(Q[j][i], lanczos_vecs[j]);
    //     // }
    //     // spinor_vec.emplace_back(commBase);
    //     // spinor_vec[i] = eigenvector;
    // }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::QRDecomposition(int n, std::vector<std::vector<floatT>>& A, std::vector<std::vector<floatT>>& Q) {
    for (int i = 0; i < n; ++i) {
        // Compute Householder vector
        floatT alpha = 0.0;
        for (int j = i; j < n; ++j) alpha += A[j][i] * A[j][i];
        alpha = std::sqrt(alpha);
        if (A[i][i] > 0) alpha *= -1;

        Q[i][i] = 1.0;
        for (int j = i+1; j < n; ++j) Q[i][j] = A[j][i] / alpha;

        // Apply Householder transformation
        floatT beta = 2.0 * alpha * alpha;
        for (int j = i; j < n; ++j) {
            A[i][j] = 0.0;
            for (int k = i+1; k < n; ++k) {
                A[j][k] -= Q[i][j] * Q[k][i] * beta * A[i][k];
                A[k][j] -= Q[i][k] * Q[j][i] * beta * A[i][j];
            }
        }
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::writeEigenpairsAlternating(const std::string &fname, int diskprec, Endianness en) 
{    
    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    CommunicationBase &commBase = this->getComm();

    EigenFormat<HaloDepthSpin> evnersc(commBase);

    if (commBase.IamRoot()) {
        std::ofstream out(fname.c_str());
        if (!out.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname));
        }
        if (!evnersc.template write_header<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>(diskprec, spinor_count, en, out)) {
            throw std::runtime_error(stdLogger.fatal("Error writing header of " + std::string(fname)));
        }
        out.close();
    }

    
    size_t spinor_size = GInd::getLatData().vol4 * evnersc.bytes_per_site();
    size_t file_size = (spinor_size + 16) * spinor_count + evnersc.header_size();

    commBase.initIOBinary(fname, file_size, evnersc.bytes_per_site(), evnersc.header_size(), global, local, WRITE);
    
    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, NStacks> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, NStacks> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, NStacks> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, NStacks> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();

    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        int even_len = spinor_count - (spinor_count % 2);
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
                if (t+z+y+x==0) {
                    double lambda = lambda_vec[n];
                    evnersc.put_scalar(lambda);
                    lambda = lambda_vec[n+1];
                    evnersc.put_scalar(lambda);
                    rootLogger.debug("Writing site ", x, " ", y, " ", z, " ", t);
                }

                gSite site = GInd::getSite(x,y,z,t);
                Vect3<floatT> tmp = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
                evnersc.put_vector(tmp);

                if (evnersc.end_of_buffer()) {
                    evnersc.process_write_data();
                    commBase.writeBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
                    rootLogger.info("Writing spinor", n, " data to file ", fname.c_str());
                }
            }
        }
        if (spinor_count % 2 != 0) {
            int n = spinor_count - 1;
            spinor = spinor_vec[n];
            spinor_even.iterateOverBulk(
                returnSpinor<floatT, false, LatticeLayout, HaloDepthSpin, NStacks>(spinor)
            );
            spinor_split.even = spinor_even;

            // No odd partner, so clear or skip odd
            spinor_split.odd = spinor_odd;

            spinor_host = spinor_split;

            for (size_t t = 0; t < GInd::getLatData().lt; t++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
            for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                if (t+z+y+x==0) {
                    double lambda = lambda_vec[n];
                    evnersc.put_scalar(lambda);
                    rootLogger.debug("Reading site ", x, " ", y, " ", z, " ", t);
                }

                gSite site = GInd::getSite(x,y,z,t);
                Vect3<floatT> tmp = spinor_accessor.getElement(GInd::getSiteMu(site, 0));
                evnersc.put_vector(tmp);

                if (evnersc.end_of_buffer()) {
                    evnersc.process_write_data();
                    commBase.writeBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
                    rootLogger.info("Writing spinor", n, " data to file ", fname.c_str());
                }
            }
        }
    }
    commBase.closeIOBinary();
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

    if (commBase.IamRoot()) {
        std::ofstream out(fname.c_str());
        if (!out.is_open()) {
            throw std::runtime_error(stdLogger.fatal("Could not open file ", fname));
        }
        
        // Write header
        if (!evnersc.template write_header<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>(diskprec, spinor_count, en, out)) {
            throw std::runtime_error(stdLogger.fatal("Error writing header of " + std::string(fname)));
        }

        // Write eigenvalues
        for (size_t i = 0; i < even_len; ++i) {
            out.write(reinterpret_cast<const char*>(&lambda_vec[i]), sizeof(floatT));
        }

        out.close();
    }

    size_t spinor_size = GInd::getLatData().vol4 * evnersc.bytes_per_site();
    size_t displacement = 16 * spinor_count + evnersc.header_size();
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
                commBase.writeBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
                rootLogger.info("Writing spinor", n, " data to file ", fname.c_str());
            }
        }
    }
    commBase.closeIOBinary();
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::readEigenpairsAlternating(const std::string &fname) 
{
    CommunicationBase &commBase = this->getComm();

    EigenFormat<HaloDepthSpin> evnersc(commBase);
    typedef GIndexer<All, HaloDepthSpin> GInd;

    rootLogger.info("Reading eigenpairs from file ", fname.c_str());

    std::ifstream in;
    if (commBase.IamRoot()) {
        in.open(fname.c_str());
    }
    if (!evnersc.read_header(in)) {
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }
    rootLogger.info("Reading eigenpairs from file ", fname.c_str());
    spinor_count = evnersc.spinor_count();

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    commBase.initIOBinary(fname, 0, evnersc.bytes_per_site(), evnersc.header_size(), global, local, READ);

    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinor(commBase);
    Spinorfield<floatT, false, Even, HaloDepthSpin, NStacks> spinor_even(commBase);
    Spinorfield<floatT, false, Odd, HaloDepthSpin, NStacks> spinor_odd(commBase);
    Spinorfield<floatT, false, All, HaloDepthSpin, NStacks> spinor_host(commBase);
    SpinorfieldAll<floatT, false, HaloDepthSpin, NStacks> spinor_split(commBase);
    Vect3arrayAcc<floatT> spinor_accessor = spinor_host.getAccessor();
    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {

        lambda_vec.clear();
        spinor_vec.clear();
        lambda_vec.resize(spinor_count);

        for (int n = 0; n < spinor_count; n+=2) {

            for (size_t t = 0; t < GInd::getLatData().lt; t++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
            for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                if (evnersc.end_of_buffer()) {
                    rootLogger.info("Reading spinor", n, " data from file ", fname.c_str());
                    commBase.readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
                    evnersc.process_read_data();
                }
                if (t+z+y+x==0) {
                    double lambda = evnersc.template get_scalar<double>();
                    lambda_vec[n] = lambda;
                    lambda = evnersc.template get_scalar<double>();
                    lambda_vec[n+1] = lambda;
                    rootLogger.debug("Reading site ", x, " ", y, " ", z, " ", t);
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
    if (commBase.IamRoot()) {
        in.open(fname.c_str());
    }
    if (!evnersc.read_header(in)) {
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }
    in.close();

    int spinor_count = evnersc.spinor_count();
    int even_len = spinor_count - (spinor_count % 2);


    size_t displacement = 16 * spinor_count + evnersc.header_size();

    commBase.initIOBinary(fname, 0, evnersc.bytes_per_site(), displacement, global, local, READ);

    spinor_vec.clear();
    lambda_vec.reserve(even_len);

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
                commBase.readBinary(evnersc.buf_ptr(), evnersc.buf_size() / evnersc.bytes_per_site());
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
    // smearing.SmearAll();

    HisqDSlash<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);


    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int i = 0; i < spinor_count; i++) {
            Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorIn = spinor_vec[i];
            Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(commBase);
            
            floatT lambda = lambda_vec[i];
            rootLogger.info("lambda=", lambda);
            
            vr = spinorIn;
            
            dslash.applyMdaggM(vr, spinorIn, true);

            vr.template axpyThisB<64>(lambda, spinorIn);
            rootLogger.info("norm(Ax-µx)**2=", vr.realdotProduct(vr));
        }
    }
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVector(double mass, Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn) {
    CommunicationBase &commBase = this->getComm();
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorSum(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(commBase);

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
void Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::startVectorTester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS) {
    CommunicationBase &commBase = this->getComm();

    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vr(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> va(commBase);
    va = spinorRHS;

    dslash.applyMdaggM(vr, spinorStart, true);
    
    COMPLEX(double) sum0 = vr.dotProduct(vr)-va.dotProduct(vr);
    rootLogger.info("start_vector_tester0=", sum0);

    COMPLEX(double) sum1 = va.dotProduct(vr);
    
    for (int i =0; i < spinor_count; i++) {
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &spinorEv = spinor_vec[i];
        vr = spinorEv;
        sum1 -= va.dotProduct(vr) * vr.dotProduct(va);    
    }
    rootLogger.info("start_vector_tester1=", sum1);
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

