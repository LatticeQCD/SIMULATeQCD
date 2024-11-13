#pragma once

#include "../base/latticeContainer.h"
#include "../modules/inverter/inverter.h"
#include "spinorfield.h"

/// Abstract base class for all kind of linear operators that shall enter the inversion
// template <typename Vector>
// class LinearOperator{
// public:
//     virtual void applyMdaggM(Vector&, const Vector&, bool update = true) = 0;
// };

// class eigenpairsParameters : public LatticeParameters {
//     public:
//         DynamicParameter<long> operator_ids;
//         DynamicParameter<double> valence_masses;

//         Parameter<int> num_random_vectors;
//         Parameter<int> num_toread_vectors;
//         Parameter<int> seed;

//         // Dslash related values
//         Parameter<double> mu_f;
//         Parameter<bool> use_naik_epsilon;
//         Parameter<double> residue;
//         Parameter<int> cgMax;

//         Parameter<std::string> eigen_file;
//         Parameter<std::string> output_file;
//         Parameter<std::string> collected_output_file;

//         eigenpairsParameters() {
//             add(operator_ids, "operator_ids");
//             add(valence_masses, "valence_masses");
//             add(num_random_vectors, "num_random_vectors");
//             add(num_toread_vectors, "num_toread_vectors");
//             addDefault(seed, "seed", 0);
//             addOptional(eigen_file, "eigen_file");
//             addOptional(output_file, "output_file");
//             addOptional(collected_output_file, "collected_output_file");

//             addDefault(mu_f, "mu0", 0.0);
//             add(use_naik_epsilon, "use_naik_epsilon"); // No default to make this very clear in usage!
//             addDefault(residue, "residue", 1e-12);
//             addDefault(cgMax, "cgMax", 20000);
//         }
// };


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class eigenpairs : public SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepthSpin>
{
protected:
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> _spinor_lattice;
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>> spinors;

private:

    eigenpairs(const eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &) = delete;


public:
    typedef GIndexer<LatticeLayout, HaloDepthSpin> GInd;

    std::vector<double> lambda_vect;
    int numVec;


    explicit eigenpairs(CommunicationBase &comm) :
            SiteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepthSpin>(comm),
            _spinor_lattice(comm) { }

    void read_evnersc(const int &numVecIn, const std::string &fname);
    void read_evnersc_host(Vect3arrayAcc<floatT> Vect3arrayAcc, int idxvec, double &lambda, const std::string &fname);
    void tester(CommunicationBase &commBase, Gaugefield<floatT,onDevice,HaloDepthGauge,R18> &gauge);
    void start_vector(double mass,  Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorOut, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn);
    void start_vector_tester(LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS);

    virtual Vect3arrayAcc<floatT> getAccessor() const;
};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
inline Vect3arrayAcc<floatT> eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks>::getAccessor() const {
    return (_spinor_lattice.getAccessor());
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Nstacks>
struct returnEigen {
    Vect3arrayAcc<floatT> _gAcc;

    explicit returnEigen(const eigenpairs<floatT, onDevice, LatLayout, HaloDepthGauge, HaloDepthSpin, Nstacks> &spinorIn);
    __host__ __device__ Vect3<floatT> operator()(gSiteStack site);
};
