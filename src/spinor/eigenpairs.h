#pragma once

#include "../base/latticeContainer.h"
#include "../base/IO/evnersc.h"
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

    using Spinor_external = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>;
    using Spinor_internal = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>;
    using LambdaType = typename std::conditional<LatticeLayout == All, std::array<double, 2>, double>::type;
protected:
    Spinor_internal _spinor_lattice;

private:

    Eigenpairs(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;
    std::vector<Spinor_internal> spinor_vec;
    std::vector<LambdaType> lambda_vec;
    int spinor_count = 0;


public:
    typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    // Use pairs of doubles for lambda_vec when LatticeLayout == All, otherwise use single double



    explicit Eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepthSpin>(comm),
            _spinor_lattice(comm) { }

    void writeEigenpairsSequential(const std::string &fname, int diskprec, Endianness en);
    void readEigenpairsSequential(const std::string &fname);

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

    void getEigenValue(LambdaType &lambdaOut, int index) const {
        lambdaOut = lambda_vec[index];
    }

    void clearEigenpairs() {
        spinor_vec.clear();
        lambda_vec.clear();
        spinor_count = 0;
    }

    int SpinorCount() const {
        return spinor_count;
    }

    void fillRandom(const int &num_vec_in);
    
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
