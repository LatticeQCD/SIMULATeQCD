#pragma once

#include "../base/latticeContainer.h"
#include "../modules/inverter/inverter.h"
#include "../base/IO/evnersc.h"
#include "spinorfield.h"


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class Eigenpairs : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepthSpin>
{
protected:
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> _spinor_lattice;

private:

    Eigenpairs(const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    // Use pairs of doubles for lambda_vec when LatticeLayout == All, otherwise use single double
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>> spinor_vec;
    using LambdaType = typename std::conditional<LatticeLayout == All, std::array<double, 2>, double>::type;
        std::vector<LambdaType> lambda_vec;

    int vector_len;


    explicit Eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepthSpin>(comm),
            _spinor_lattice(comm) { }

    void fillRandom(const int &num_vec_in);

    void writeEigenpairsFile(const std::string &fname, int diskprec, Endianness en);
    void writeEigenpairsFileHost(const std::string &fname, int diskprec, Endianness en);

    void readEigenpairsFile(const std::string &fname);
    void readEigenpairsFileHost(const std::string &fname);

    void tester(CommunicationBase &commBase, Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge);
    void startVector(double mass,  Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn);
    void startVectorTester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS);

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
