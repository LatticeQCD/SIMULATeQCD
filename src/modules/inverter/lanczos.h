#pragma once

#include "../../spinor/spinorfield.h"

template <typename Vector>
class LinearOperator;

// Optional polynomial filter used inside the Lanczos projection.
// For low-mode searches, lowerBound should be the cutoff above the desired low
// modes, while upperBound must safely bound the top of the full spectrum. The
// unwanted interval [lowerBound, upperBound] is mapped to [-1, 1], where
// Chebyshev polynomials remain bounded. Desired low modes below lowerBound map
// to x < -1 and are amplified in polynomial magnitude.
struct TRLanChebyshevFilterParams {
  // Switches the polynomial filter on. The default keeps old behavior.
  bool enabled = false;
  // Degree of the Chebyshev polynomial. Denscode calls this
  // tschebyscheff_order; if enabled is true, this must be positive.
  int order = 0;
  // Low-mode cutoff: eigenvalues below this are the desired modes, while the
  // interval from lowerBound to upperBound is treated as unwanted bulk spectrum.
  double lowerBound = 0.0;
  // Safe upper bound on the full spectrum. Underestimating this can amplify
  // high modes as well as low modes.
  double upperBound = 0.0;
};

// Controls the thick-restarted Lanczos run. The defaults preserve the previous
// public API while allowing callers to choose Denscode-like m_lan/k_lan values.
struct TRLanRestartParams {
  // Maximum Krylov basis size before a restart. This corresponds to Denscode's
  // m_lan. If negative, a conservative size is chosen from requestedEigenpairs.
  int krylovDim = -1;
  // Number of Ritz vectors retained at each restart. This is Denscode's thick
  // retained block k1/k_lan idea. It must be at least requestedEigenpairs and
  // leave room for a restart residual and one newly generated direction.
  int thickRestartDim = -1;
  // Safety cap for repeated restarts if the requested residual tolerance is not
  // reached.
  int maxRestarts = 200;
  // Relative physical residual tolerance for the requested low modes:
  // ||MdaggM v - lambda v|| <= residualTol * max(1, |lambda|).
  double residualTol = 1e-10;
  // Treat a newly generated basis vector with norm below this value as a
  // Lanczos breakdown and stop extending the current space.
  double breakdownTol = 1e-12;
  // Seed for the Gaussian starting vector, matching the old compute overload.
  unsigned int seed = 1234;
  // Optional Chebyshev filter parameters.
  TRLanChebyshevFilterParams chebyshev;
  // If true, hitting maxRestarts before the requested physical residual is
  // reached is an error. The legacy overload disables this to preserve the old
  // "return best Ritz pairs from one projection" behavior.
  bool failOnNoConvergence = true;
};

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

  // Restart-aware entry point. This overload is what new callers should use
  // when they want thick restart and/or the Chebyshev filter.
  static void compute(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      int requestedEigenpairs,
      std::vector<Spinor_internal> &eigenvectors,
      std::vector<double> &eigenvalues,
      const TRLanRestartParams &params);

private:
  static void applyMdaggMSingle(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in);

  // Apply either the original MdaggM operator or the filtered polynomial,
  // depending on filter.enabled. This keeps the restart code independent of the
  // operator choice.
  static void applyFilteredOperator(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      const TRLanChebyshevFilterParams &filter);

  // Evaluate the Chebyshev polynomial filter with the three-term recurrence.
  static void applyChebyshevFilter(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &out,
      const Spinor_internal &in,
      const TRLanChebyshevFilterParams &filter);

  // Convenience helpers for operations that are repeatedly needed when building
  // Ritz combinations and residuals.
  static void zero(Spinor_internal &vec);

  static void axpyReal(Spinor_internal &out, double coeff, const Spinor_internal &in);

  static void fullReorthogonalize(Spinor_internal &vec, std::vector<Spinor_internal> &basis, int nBasis);

  static void normalizeOrThrow(Spinor_internal &vec, const char *errorMsg);

  static void validateParams(const TRLanRestartParams &params, int requestedEigenpairs, int krylovDim);

  // Extend the current basis until targetDim is reached or a breakdown occurs.
  // After a restart, basis already contains retained Ritz vectors.
  static void buildKrylovSpace(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      std::vector<Spinor_internal> &basis,
      int targetDim,
      double breakdownTol,
      const TRLanChebyshevFilterParams &filter);

  // Build the dense projected matrix H_ij = <q_i, A q_j> using the current
  // basis and whichever operator A is active.
  static void projectedMatrix(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      std::vector<Spinor_internal> &basis,
      const TRLanChebyshevFilterParams &filter,
      std::vector<std::vector<double>> &matrix);

  // Diagonalize the small dense projected matrix. The eigenvectors z are used
  // to form Ritz vectors in the full spinor space.
  static void diagonalizeSymmetricDense(
      const std::vector<std::vector<double>> &matrix,
      std::vector<double> &d,
      std::vector<std::vector<double>> &z);

  // Sort projected eigenpairs by algebraic value. This is correct for the
  // unfiltered MdaggM problem.
  static void sortDenseEigenpairsAscending(std::vector<double> &d, std::vector<std::vector<double>> &z);

  // Sort filtered projected eigenpairs by largest polynomial magnitude. Low
  // physical modes can appear at either algebraic end of T_n(A'), depending on
  // polynomial parity, so magnitude is safer than always sorting ascending.
  static void sortDenseEigenpairsByMagnitudeDescending(std::vector<double> &d, std::vector<std::vector<double>> &z);

  // Convert projected eigenvectors into physical Ritz spinors:
  // r_ev = sum_j z[j][ev] q_j.
  static void buildRitzVectors(
      const std::vector<Spinor_internal> &basis,
      const std::vector<std::vector<double>> &z,
      int nKeep,
      CommunicationBase &comm,
      std::vector<Spinor_internal> &ritzVectors);

  // Compute ||MdaggM v - lambda v|| and return lambda through the reference.
  // This physical residual is the only convergence certificate, even when the
  // basis was built with a Chebyshev filter.
  static double physicalResidual(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &vec,
      double &lambda);

  // Return <v, MdaggM v>/<v,v>. When filtering is enabled this converts the
  // final vector quality back to the original physical operator.
  static double rayleighQuotient(
      CommunicationBase &comm,
      LinearOperator<Spinor_external> &op,
      Spinor_internal &vec);

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
