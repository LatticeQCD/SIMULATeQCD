#pragma once

#include "../../spinor/spinorfield.h"
#include <vector>

template <typename Vector>
class LinearOperator;

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
class TRLanSpinorSolver {
public:
  using SpinorSingle = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>;
  using SpinorExternal = Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>;

  static void compute(
      CommunicationBase &comm,
      LinearOperator<SpinorExternal> &op,
      int requestedEigenpairs,
      std::vector<SpinorSingle> &eigenvectors,
      std::vector<double> &eigenvalues,
      int krylovDim = -1,
      double breakdownTol = 1e-12,
      unsigned int seed = 1234);

private:
  static void applyMdaggMSingle(
      CommunicationBase &comm,
      LinearOperator<SpinorExternal> &op,
      SpinorSingle &out,
      const SpinorSingle &in);

  static void fullReorthogonalize(SpinorSingle &vec, std::vector<SpinorSingle> &basis, int nBasis);

  static void normalizeOrThrow(SpinorSingle &vec, const char *errorMsg);

  static double pythag(double a, double b);

  static void tqli(std::vector<double> &d, std::vector<double> &e, int n, std::vector<std::vector<double>> &z);

  static void sortEigenpairsAscending(std::vector<double> &d, std::vector<std::vector<double>> &z, int n);

  static void solveProjectedSystem(
      const std::vector<SpinorSingle> &basis,
      const std::vector<double> &alpha,
      const std::vector<double> &beta,
      int n,
      int requestedEigenpairs,
      std::vector<SpinorSingle> &eigenvectors,
      std::vector<double> &eigenvalues,
      CommunicationBase &comm);
};