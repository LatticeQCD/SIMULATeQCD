#pragma once

#include "lanczosKernels.h"

#include <cstdint>
#include <vector>

template <typename Vector>
class LinearOperator;

struct TRLanChebyshevFilterParams {
  bool enabled = false;
  int order = 0;
  double lowerBound = 0.0;
  double upperBound = 0.0;
};

// Densecode applies
//
//   [ beta * (I + alpha * DoeDeo / order) ]^order .
//
// The affine mapping from SIMULATeQCD's raw operator A to Densecode's DoeDeo
// operator D is configured as D = operatorScale * (A - operatorShift).
struct TRLanExponentialFilterParams {
  bool enabled = false;
  int order = 0;
  double alpha = 0.0;
  double beta = 1.0;
  // Additive scalar contained in SIMULATeQCD's operator but absent from
  // Densecode's DoeDeo filter.
  double operatorShift = 0.0;
  // Use +1 when A already has Densecode's massless DoeDeo sign. Use -1 when
  // A - operatorShift is the positive operator -DoeDeo.
  double operatorScale = 1.0;
};

enum class TRLanConvergenceCriterion {
  MaximumScaledPerMode,
  DensecodeAggregatePhysical
};

inline const char *trlanConvergenceCriterionName(
    const TRLanConvergenceCriterion criterion) {
  switch (criterion) {
    case TRLanConvergenceCriterion::MaximumScaledPerMode:
      return "maximum_scaled_per_mode";
    case TRLanConvergenceCriterion::DensecodeAggregatePhysical:
      return "densecode_aggregate_physical";
  }
  return "unknown";
}

struct TRLanRestartParams {
  // Densecode's m_lan and retained k_lan dimensions.
  int krylovDim = -1;
  int thickRestartDim = -1;
  int maxRestarts = 200;

  // Physical residual requirement for the unfiltered eigenproblem.
  double residualTol = 1e-10;
  TRLanConvergenceCriterion convergenceCriterion =
      TRLanConvergenceCriterion::MaximumScaledPerMode;
  double breakdownTol = 1e-12;
  unsigned int seed = 1234;

  TRLanChebyshevFilterParams chebyshev;
  TRLanExponentialFilterParams exponential;

  // One batched classical Gram-Schmidt pass plus this many correction passes
  // are applied to every new vector. Two passes are robust for filtered runs.
  int reorthogonalizationPasses = 2;

  // Physical residuals are always checked when the inexpensive Lanczos residual
  // estimate indicates convergence and on the final cycle. A positive interval
  // additionally forces a physical check every N restart cycles.
  int physicalCheckInterval = 5;

  bool failOnNoConvergence = true;

  // Diagnostic output is opt-in so production runs retain the existing
  // operator count, collective count, and logging volume.
  bool convergenceDiagnostics = false;
  bool mpiConsistencyDiagnostics = false;
};

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
class TRLanSpinorSolver {
public:
  using Spinor_internal =
      Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>;
  using Spinor_external =
      Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>;

  static void compute(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      int requestedEigenpairs,
      std::vector<Spinor_internal> &eigenvectors,
      std::vector<double> &eigenvalues,
      int krylovDim = -1,
      double breakdownTol = 1e-12,
      unsigned int seed = 1234);

  static void compute(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      int requestedEigenpairs,
      std::vector<Spinor_internal> &eigenvectors,
      std::vector<double> &eigenvalues,
      const TRLanRestartParams &params);

private:
  using Basis =
      trlan_detail::Basis<
          floatT, onDevice, LatticeLayout, HaloDepthSpin>;

  struct OperatorWorkspace {
    Spinor_external stackedInput;
    Spinor_external stackedOutput;
    Spinor_internal stage0;
    Spinor_internal stage1;
    Spinor_internal stage2;
    Spinor_internal basisVector;
    Spinor_internal filteredOutput;
    uint64_t operatorApplications;

    explicit OperatorWorkspace(CommunicationBase &comm)
        : stackedInput(comm, "TRLan_stackedInput"),
          stackedOutput(comm, "TRLan_stackedOutput"),
          stage0(comm, "TRLan_filterStage0"),
          stage1(comm, "TRLan_filterStage1"),
          stage2(comm, "TRLan_filterStage2"),
          basisVector(comm, "TRLan_basisVector"),
          filteredOutput(comm, "TRLan_filteredOutput"),
          operatorApplications(0) {}
  };

  static void validateParams(
      const TRLanRestartParams &params,
      int requestedEigenpairs,
      int krylovDim,
      int thickRestartDim);

  static void applyMdaggMSingle(
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      OperatorWorkspace &workspace);

  static void applyFilteredOperator(
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      const TRLanRestartParams &params,
      OperatorWorkspace &workspace);

  static void applyChebyshevFilter(
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      const TRLanChebyshevFilterParams &filter,
      OperatorWorkspace &workspace);

  static void applyExponentialFilter(
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      const TRLanExponentialFilterParams &filter,
      OperatorWorkspace &workspace);

  static int extendLanczosFactorization(
      LinearOperator<Spinor_external> &op,
      Basis &basis,
      int firstColumn,
      int targetDimension,
      double breakdownTol,
      int reorthogonalizationPasses,
      const TRLanRestartParams &params,
      OperatorWorkspace &workspace,
      Spinor_internal &terminalVector,
      double &terminalBeta,
      std::vector<std::vector<double>> &projected);

  static void normalizeOrThrow(
      Spinor_internal &vector,
      double breakdownTol,
      const char *errorMessage);

  static void diagonalizeProjected(
      const std::vector<std::vector<double>> &projected,
      int dimension,
      std::vector<double> &eigenvalues,
      std::vector<std::vector<double>> &eigenvectors);

  static std::vector<int> targetOrder(
      const std::vector<double> &filteredEigenvalues,
      const TRLanRestartParams &params);

  static double correctedExponentialEigenvalue(
      double filteredEigenvalue,
      const TRLanExponentialFilterParams &filter);

  static double exponentialDerivativeMagnitude(
      double filteredEigenvalue,
      const TRLanExponentialFilterParams &filter);

  static bool inexpensiveConvergenceGate(
      const std::vector<double> &filteredEigenvalues,
      const std::vector<double> &filteredResiduals,
      int requestedEigenpairs,
      double residualTol,
      const TRLanRestartParams &params);

  static double physicalResidual(
      LinearOperator<Spinor_external> &op,
      Spinor_internal &vector,
      double physicalEigenvalue,
      bool densecodePositiveConvention,
      double operatorShift,
      double operatorScale,
      double &rayleighQuotient,
      double *vectorNorm,
      OperatorWorkspace &workspace);

  static void householderTridiagonalize(
      std::vector<std::vector<double>> &matrix,
      int dimension,
      std::vector<double> &diagonal,
      std::vector<double> &offDiagonal);

  static double pythag(double a, double b);

  static void tridiagonalQL(
      std::vector<double> &diagonal,
      std::vector<double> &offDiagonal,
      int dimension,
      std::vector<std::vector<double>> &eigenvectors);
};
