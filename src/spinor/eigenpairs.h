#pragma once

#include "../base/latticeContainer.h"
#include "../modules/inverter/inverter.h"
#include "../base/IO/evnersc.h"
#include "spinorfield.h"


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class Eigenpairs : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepthSpin>
{

    using Spinor_t = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>;

protected:
    Spinor_t _spinor_lattice;

private:

    Eigenpairs(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    // Use pairs of doubles for lambda_vec when LatticeLayout == All, otherwise use single double
    std::vector<Spinor_t> spinor_vec;
    using LambdaType = typename std::conditional<LatticeLayout == All, std::array<double, 2>, double>::type;
    std::vector<LambdaType> lambda_vec;

    int spinor_count = 0;


    explicit Eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepthSpin>(comm),
            _spinor_lattice(comm) { }

    void fillRandom(const int &num_vec_in);

    void writeEigenpairsSequential(const std::string &fname, int diskprec, Endianness en);
    void readEigenpairsSequential(const std::string &fname);

    void tester(Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge);
    void startVector(double mass,  Spinor_t& spinorOut, const Spinor_t& spinorIn);
    void startVectorTester(LinearOperator<Spinor_t>& dslash, const Spinor_t& spinorStart, const Spinor_t& spinorRHS);

    
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
