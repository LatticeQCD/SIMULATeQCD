#pragma once

#include "../../spinor/spinorfield.h"

template <typename Vector>
class LinearOperator;

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
class TRLanSpinorSolver {
public:
  using Spinor_internal = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>;
  using Spinor_external = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>;

  static void compute(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      int requestedEigenpairs,
      std::vector<Spinor_internal> &eigenvectors,
      std::vector<double> &eigenvalues,
      int krylovDim = -1,
      double breakdownTol = 1e-12,
      unsigned int seed = 1234);

private:
  static void applyMdaggMSingle(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in);

  static void fullReorthogonalize(Spinor_internal &vec, std::vector<Spinor_internal> &basis, int nBasis);

  static void normalizeOrThrow(Spinor_internal &vec, const char *errorMsg);

  static double pythag(double a, double b);

  static void tqli(std::vector<double> &d, std::vector<double> &e, int n, std::vector<std::vector<double>> &z);

  static void sortEigenpairsAscending(std::vector<double> &d, std::vector<std::vector<double>> &z, int n);

  static void solveProjectedSystem(
      const std::vector<Spinor_internal> &basis,
      const std::vector<double> &alpha,
      const std::vector<double> &beta,
      int n,
      int requestedEigenpairs,
      std::vector<Spinor_internal> &eigenvectors,
      std::vector<double> &eigenvalues,
      CommunicationBase &comm);
};