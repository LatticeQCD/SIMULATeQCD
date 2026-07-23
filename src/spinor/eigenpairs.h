#pragma once

#include "../base/latticeContainer.h"
#include "../base/IO/evnersc.h"
#include "../modules/inverter/lanczos.h"
#include "spinorfield.h"

/// Abstract base class for all kind of linear operators that shall enter the inversion
template <typename Vector>
class LinearOperator{
public:
    virtual void applyMdaggM(Vector&, const Vector&, bool update = true) = 0;
};


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class Eigenpairs : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepthSpin>
{
    using Spinor_external = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>; // Use the same template parameters as Eigenpairs for the external Spinorfield type
    using Spinor_internal = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>; // Use single stack for internal Spinorfield type
    using LambdaType = typename std::conditional<LatticeLayout == All, std::array<double, 2>, double>::type; // Use pairs of doubles for lambda_vec when LatticeLayout == All, otherwise use single double
protected:
    Spinor_internal _spinor_lattice;

private:
    // Disable copy constructor and copy assignment operator to prevent accidental copying of Eigenpairs objects
    Eigenpairs(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;
    Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> & operator=(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;

    // Store eigenvectors and eigenvalues in separate vectors. The eigenvectors are stored as Spinor_internal objects, which use a single stack, while the eigenvalues are stored in a vector of LambdaType, which can be either double or std::array<double, 2> depending on the LatticeLayout.
    std::vector<Spinor_internal> spinor_vec;
    std::vector<LambdaType> lambda_vec;

    // Store the number of eigenvectors currently stored in the spinor_vec and lambda_vec vectors. This is used to keep track of how many eigenvectors and eigenvalues have been computed and stored, and to ensure that the correct number of eigenvectors and eigenvalues are written to disk when the writeEigenpairsSequential function is called.
    int spinor_count = 0;


public:
    // typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    explicit Eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepthSpin>(comm),
            _spinor_lattice(comm) { }

    void writeEigenpairsToFile(const std::string &fname, int diskprec, Endianness en); // Write eigenpairs to file in EVNERSC format. The diskprec parameter specifies the precision to use when writing the eigenvalues and eigenvectors to disk, and can be either 32 or 64. The en parameter specifies the endianness to use when writing the eigenvalues and eigenvectors to disk, and can be either ENDIAN_LITTLE, ENDIAN_BIG, or ENDIAN_AUTO.
    void readEigenpairsFromFile(const std::string &fname); // Read eigenpairs from file in EVNERSC format. This function reads the eigenvalues and eigenvectors from the specified file and stores them in the lambda_vec and spinor_vec vectors, respectively. The function assumes that the file is in EVNERSC format and that the eigenvalues and eigenvectors are stored in the same order as they were written by the writeEigenpairsToFile function.

    void getEigenPair(Spinor_external &spinorOut, LambdaType &lambdaOut, int index) const {
        for (size_t i = 0; i < NStacks; i++) {
            spinorOut.copyFromStackToStack(spinor_vec[index], i, 0);
        }
        lambdaOut = lambda_vec[index];
    }

    void getEigenSpinor(Spinor_external &spinorOut, int index) const {
        for (size_t i = 0; i < NStacks; i++) {
            spinorOut.copyFromStackToStack(spinor_vec[index], i, 0);
        }
    }

    LambdaType getEigenValue(int index) const {
        return lambda_vec[index];
    }

    void clearEigenpairs() {
        spinor_vec.clear();
        lambda_vec.clear();
        spinor_count = 0;
    }

    int SpinorCount() const {
        return spinor_count;
    }

    void checkEigenValueEquation(LinearOperator<Spinor_external> &op, double mass, double tol);

    void fillRandom(const int &num_vec_in);

    void lanczos(LinearOperator<Spinor_external> &op, const int &num_vec_in);
    // Restart/filter-aware Lanczos entry point. Use this overload to pass
    // Denscode-style m_lan/k_lan controls and Chebyshev filter parameters.
    void lanczos(LinearOperator<Spinor_external> &op, const int &num_vec_in, const TRLanRestartParams &params);
    
    virtual Vect3arrayAcc<floatT> getAccessor() const;
};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
inline Vect3arrayAcc<floatT> Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::getAccessor() const {
    return (_spinor_lattice.getAccessor());
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Nstacks>
struct returnEigen {
    Vect3arrayAcc<floatT> _gAcc;

    explicit returnEigen(const Eigenpairs<floatT, onDevice, LatLayout, HaloDepthGauge, HaloDepthSpin, Nstacks> &spinorIn);
    __host__ __device__ Vect3<floatT> operator()(gSiteStack site);
};
