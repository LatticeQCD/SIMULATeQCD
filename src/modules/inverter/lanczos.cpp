#include "lanczos.h"

#include "../../define.h"
#include "../../base/math/random.h"
#include "../../spinor/eigenpairs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::compute(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	int requestedEigenpairs,
	std::vector<Spinor_internal> &eigenvectors,
	std::vector<double> &eigenvalues,
	int krylovDim,
	double breakdownTol,
	unsigned int seed)
{
	// Preserve the old API by translating the original arguments into the new
	// restart parameter object. With the default settings this behaves like the
	// previous non-restarted solver from a caller's point of view.
	TRLanRestartParams params;
	params.krylovDim = krylovDim;
	params.breakdownTol = breakdownTol;
	params.seed = seed;
	params.maxRestarts = 0;
	params.failOnNoConvergence = false;
	compute(comm, op, requestedEigenpairs, eigenvectors, eigenvalues, params);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::compute(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	int requestedEigenpairs,
	std::vector<Spinor_internal> &eigenvectors,
	std::vector<double> &eigenvalues,
	const TRLanRestartParams &params)
{
	if constexpr (LatticeLayout == Layout::All) {
		throw std::runtime_error(stdLogger.fatal("TRLanSpinorSolver currently supports only Even/Odd layout"));
	}

	if (requestedEigenpairs <= 0) {
		eigenvectors.clear();
		eigenvalues.clear();
		return;
	}

	// m is the maximum Krylov space dimension before a restart. Denscode calls
	// the same idea m_lan. A larger m gives better Ritz approximations per cycle
	// but costs more spinor memory.
	const int m = (params.krylovDim > 0) ? std::max(params.krylovDim, requestedEigenpairs + 2) : std::max(2 * requestedEigenpairs + 8, requestedEigenpairs + 2);
	validateParams(params, requestedEigenpairs, m);

	double residualTol = params.residualTol;
	double breakdownTol = params.breakdownTol;
	if constexpr (std::is_same_v<floatT, float>) {
		// Single-precision spinor storage cannot generally certify double-like
		// residuals. These floors keep default restarted runs from chasing
		// unattainable tolerances forever.
		residualTol = std::max(residualTol, 1e-6);
		breakdownTol = std::max(breakdownTol, 1e-7);
	}

	// thickDim is how many Ritz vectors survive a restart. Keeping the requested
	// low modes plus guard vectors stabilizes clustered or nearly degenerate low
	// spectra. The m-2 cap leaves room for a residual direction and one fresh
	// Krylov vector after the restart.
	const int guard = std::max(4, requestedEigenpairs / 2);
	const int thickDim = (params.thickRestartDim > 0) ? params.thickRestartDim : std::min(m - 2, requestedEigenpairs + guard);
	if (thickDim < requestedEigenpairs) {
		throw std::runtime_error(stdLogger.fatal("thickRestartDim must keep at least the requested eigenpairs"));
	}

	// q stores the current orthonormal basis. It starts with one normalized
	// Gaussian vector and is later replaced by the retained Ritz vectors.
	std::vector<Spinor_internal> q;
	q.reserve(m);

	grnd_state<false> h_rand;
	h_rand.make_rng_state(params.seed);

	q.emplace_back(comm);
	q[0].gauss(h_rand.state);
	normalizeOrThrow(q[0], "Could not normalize start vector in Lanczos");

	std::vector<double> ritzValues;
	std::vector<std::vector<double>> ritzCoefficients;
	std::vector<Spinor_internal> ritzVectors;
	std::vector<double> physicalEigenvalues;
	std::vector<double> physicalResiduals;
	std::vector<double> correctedEigenvalues;
	std::vector<int> candidateOrder;

	// Each cycle extends q to size m, projects the operator into that subspace,
	// diagonalizes the projected matrix, and either stops or thick-restarts with
	// the best Ritz vectors. This is the same flow as Denscode's AlgTRLan::run.
	double maxPhysicalResidual = std::numeric_limits<double>::infinity();
	bool converged = false;
	bool stoppedByBreakdown = false;
	int restart = 0;
	for (; restart <= params.maxRestarts; ++restart) {
		// Extend the current basis. On the first cycle q has one random vector;
		// after a restart it contains the retained Ritz block plus possibly one
		// normalized residual direction.
		buildKrylovSpace(comm, op, q, m, breakdownTol, params.chebyshev, params.exponential);

		// Build and diagonalize the dense Rayleigh-Ritz matrix H = Q^dag A Q.
		// Denscode diagonalizes the tridiagonal/dense T_lan matrix here.
		std::vector<std::vector<double>> projected;
		projectedMatrix(comm, op, q, params.chebyshev, params.exponential, projected);
		diagonalizeSymmetricDense(projected, ritzValues, ritzCoefficients);
		if (params.exponential.enabled) {
			sortDenseEigenpairsDescending(ritzValues, ritzCoefficients);
		} else if (params.chebyshev.enabled) {
			sortDenseEigenpairsByMagnitudeDescending(ritzValues, ritzCoefficients);
		} else {
			sortDenseEigenpairsAscending(ritzValues, ritzCoefficients);
		}

		// Convert the retained projected eigenvectors into full spinor Ritz
		// vectors. These are the vectors that become the thick restart basis.
		buildRitzVectors(q, ritzCoefficients, std::min(thickDim, static_cast<int>(q.size())), comm, ritzVectors);

		// Check convergence only with an MdaggM residual. For the exponential path
		// this uses the inverse-transformed Ritz value that will be returned; for
		// the other paths it uses the physical Rayleigh quotient.
		const int nCandidates = static_cast<int>(ritzVectors.size());
		physicalEigenvalues.assign(nCandidates, 0.0);
		physicalResiduals.assign(nCandidates, 0.0);
		correctedEigenvalues.assign(nCandidates, 0.0);
		candidateOrder.resize(nCandidates);
		for (int i = 0; i < nCandidates; ++i) {
			candidateOrder[i] = i;
			correctedEigenvalues[i] = params.exponential.enabled
				? correctExponentialEigenvalue(ritzValues[i], params.exponential)
				: 0.0;
			if (params.exponential.enabled) {
				physicalResiduals[i] = physicalResidualForEigenvalue(
					comm,
					op,
					ritzVectors[i],
					correctedEigenvalues[i],
					physicalEigenvalues[i]);
			} else {
				physicalResiduals[i] =
					physicalResidual(comm, op, ritzVectors[i], physicalEigenvalues[i]);
				correctedEigenvalues[i] = physicalEigenvalues[i];
			}
		}
		std::sort(candidateOrder.begin(), candidateOrder.end(), [&](int lhs, int rhs) {
			return correctedEigenvalues[lhs] < correctedEigenvalues[rhs];
		});

		maxPhysicalResidual = 0.0;
		const int nRequestedAvailable = std::min(requestedEigenpairs, nCandidates);
		for (int i = 0; i < nRequestedAvailable; ++i) {
			const int idx = candidateOrder[i];
			const double allowed = residualTol * std::max(1.0, std::fabs(correctedEigenvalues[idx]));
			const double scaledResidual = physicalResiduals[idx] / std::max(allowed, std::numeric_limits<double>::min());
			maxPhysicalResidual = std::max(maxPhysicalResidual, scaledResidual);
		}
		converged = (nRequestedAvailable == requestedEigenpairs && maxPhysicalResidual <= 1.0);

		// Thick restart, following the Denscode strategy: keep a block of Ritz
		// vectors, discard the rest of the Krylov basis, then extend again.
		// Denscode stores this retained block in R_lan and multiplies by the
		// eigenvector matrix of the projected problem; here the same operation is
		// expressed directly as Spinorfield linear combinations.
		stoppedByBreakdown = (static_cast<int>(q.size()) <= thickDim);
		if (converged || restart == params.maxRestarts || stoppedByBreakdown) {
			break;
		}

		// Pick a residual direction to continue the Krylov expansion after the
		// restart. The retained Ritz vectors alone span the already-known part;
		// appending the largest unconverged residual gives the next cycle a
		// direction in which those Ritz pairs still need correction.
		Spinor_internal restartResidual(comm);
		restartResidual = ritzVectors[0];
		zero(restartResidual);
		double restartResidualNorm = 0.0;
		const int nResiduals = std::min(requestedEigenpairs, static_cast<int>(ritzVectors.size()));
		for (int ev = 0; ev < nResiduals; ++ev) {
			const int idx = candidateOrder.empty() ? ev : candidateOrder[ev];
			Spinor_internal residual(comm);
			applyFilteredOperator(
				comm, op, residual, ritzVectors[idx], params.chebyshev, params.exponential);
			axpyReal(residual, -ritzValues[idx], ritzVectors[idx]);
			const double residual2 = residual.realdotProduct(residual);
			if (!std::isfinite(residual2) || residual2 < 0.0) {
				throw std::runtime_error(stdLogger.fatal(
					"Filtered Ritz residual is not finite"));
			}
			const double norm = std::sqrt(residual2);
			if (norm > restartResidualNorm) {
				restartResidualNorm = norm;
				restartResidual = residual;
			}
		}

		// Replace the old Krylov basis by the thick retained block. This is the
		// direct Spinorfield equivalent of Denscode's multiplication
		// R_lan <- R_lan * T_lan followed by MGS reorthogonalization.
		q.clear();
		q.reserve(m);
		for (int i = 0; i < std::min(thickDim, static_cast<int>(ritzVectors.size())); ++i) {
			q.emplace_back(comm);
			q.back() = ritzVectors[i];
		}

		// If the residual direction is linearly independent of the retained block,
		// normalize it and append it. If it is too small, the next buildKrylovSpace
		// call will simply try to extend from the retained block.
		if (restartResidualNorm > breakdownTol && static_cast<int>(q.size()) < m) {
			fullReorthogonalize(restartResidual, q, static_cast<int>(q.size()));
			const double n2 = restartResidual.realdotProduct(restartResidual);
			if (!std::isfinite(n2) || n2 < 0.0) {
				throw std::runtime_error(stdLogger.fatal(
					"Thick-restart residual direction has a non-finite norm"));
			}
			if (n2 > breakdownTol * breakdownTol) {
				q.emplace_back(comm);
				q.back() = restartResidual;
				q.back() *= COMPLEX(floatT)(static_cast<floatT>(1.0 / std::sqrt(n2)), static_cast<floatT>(0.0));
			}
		}
	}

	if (!converged && params.failOnNoConvergence) {
		throw std::runtime_error(stdLogger.fatal(
			"Thick restarted Lanczos did not reach the requested physical residual before stopping"));
	}

	const int nKeep = std::min(requestedEigenpairs, static_cast<int>(ritzVectors.size()));
	eigenvectors.clear();
	eigenvalues.clear();
	eigenvectors.reserve(nKeep);
	eigenvalues.reserve(nKeep);

	// Store only the requested number of eigenpairs. The exponential path reports
	// the inverse-transformed Ritz value, matching Densecode's EVPolyCorrect before
	// WriteEV. The raw and Chebyshev paths report the physical Rayleigh quotient.
	for (int ev = 0; ev < nKeep; ++ev) {
		const int idx = candidateOrder.empty() ? ev : candidateOrder[ev];
		eigenvectors.emplace_back(comm);
		eigenvectors.back() = ritzVectors[idx];
		if (params.exponential.enabled) {
			eigenvalues.push_back(correctedEigenvalues[idx]);
		} else {
			eigenvalues.push_back(
				physicalEigenvalues.empty()
					? rayleighQuotient(comm, op, eigenvectors.back())
					: physicalEigenvalues[idx]);
		}
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::applyMdaggMSingle(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &out,
	const Spinor_internal &in)
{
	Spinor_external inStacked(comm);
	Spinor_external outStacked(comm);

	// LinearOperator works on the external NStacks spinor type. The Lanczos
	// basis stores single-stack spinors, so copy the input into each stack before
	// applying MdaggM and then read back stack zero.
	for (size_t stack = 0; stack < NStacks; ++stack) {
		inStacked.copyFromStackToStack(in, stack, 0);
	}

	op.applyMdaggM(outStacked, inStacked, true);
	out.copyFromStackToStack(outStacked, 0, 0);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::applyFilteredOperator(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &out,
	const Spinor_internal &in,
	const TRLanChebyshevFilterParams &chebyshev,
	const TRLanExponentialFilterParams &exponential)
{
	// Keep all Lanczos and restart code calling one function. This avoids having
	// separate filtered/unfiltered code paths that can drift apart.
	if (exponential.enabled) {
		applyExponentialFilter(comm, op, out, in, exponential);
	} else if (chebyshev.enabled) {
		applyChebyshevFilter(comm, op, out, in, chebyshev);
	} else {
		applyMdaggMSingle(comm, op, out, in);
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::applyChebyshevFilter(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &out,
	const Spinor_internal &in,
	const TRLanChebyshevFilterParams &filter)
{
	if (filter.upperBound <= filter.lowerBound) {
		throw std::runtime_error(stdLogger.fatal("Chebyshev filter requires upperBound > lowerBound"));
	}

	const double halfWidth = 0.5 * (filter.upperBound - filter.lowerBound);
	const double center = 0.5 * (filter.upperBound + filter.lowerBound);
	const double invHalfWidth = 1.0 / halfWidth;

	// Apply A' = (A - center I)/halfWidth. This maps the configured spectral
	// window to [-1,1], which is the domain where Chebyshev polynomials are
	// bounded and numerically useful as filters.
	auto applyScaledOperator = [&](Spinor_internal &scaledOut, const Spinor_internal &vec) {
		applyMdaggMSingle(comm, op, scaledOut, vec);
		scaledOut *= COMPLEX(floatT)(static_cast<floatT>(invHalfWidth), static_cast<floatT>(0.0));
		axpyReal(scaledOut, -center * invHalfWidth, vec);
	};

	// T_0(A') v = v. The scaled operator A' maps the filtered spectral
	// interval [lowerBound, upperBound] to [-1, 1], matching the standard
	// Chebyshev recurrence T_n(x)=2*x*T_{n-1}(x)-T_{n-2}(x).
	if (filter.order == 0) {
		out = in;
		return;
	}

	Spinor_internal t0(comm);
	Spinor_internal t1(comm);
	// Store T_0(A')v and T_1(A')v. The recurrence below only needs the previous
	// two vectors, so this avoids keeping all polynomial stages in memory.
	t0 = in;
	applyScaledOperator(t1, in);

	if (filter.order == 1) {
		out = t1;
		return;
	}

	Spinor_internal scaled(comm);
	Spinor_internal next(comm);
	for (int degree = 2; degree <= filter.order; ++degree) {
		// Compute T_degree(A')v = 2 A' T_{degree-1}(A')v
		//                         - T_{degree-2}(A')v.
		applyScaledOperator(scaled, t1);
		next = scaled;
		next *= COMPLEX(floatT)(static_cast<floatT>(2.0), static_cast<floatT>(0.0));
		axpyReal(next, -1.0, t0);
		t0 = t1;
		t1 = next;
	}

	out = t1;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::applyExponentialFilter(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &out,
	const Spinor_internal &in,
	const TRLanExponentialFilterParams &filter)
{
	// Densecode's ExpInit/ExpIter repeatedly apply
	//
	//   beta * I + alpha * beta / order * DoeDeo.
	//
	// Densecode's printed/stored eigenvalue convention is positive mu,
	// while SIMULATeQCD's raw applyMdaggM convention gives lambda_SIM = -mu
	// for the same low mode. Therefore the Densecode factor
	//     beta * (1 - alpha * mu / order)
	// becomes, in SIMULATeQCD variables,
	//     beta * (1 + alpha * lambda_SIM / order).
	const double mdaggMCoeff =
		(filter.alpha / static_cast<double>(filter.order)) * filter.beta;

	Spinor_internal current(comm);
	Spinor_internal applied(comm);
	Spinor_internal next(comm);
	current = in;

	for (int stage = 0; stage < filter.order; ++stage) {
		applyMdaggMSingle(comm, op, applied, current);
		next = current;
		next *= COMPLEX(floatT)(static_cast<floatT>(filter.beta), static_cast<floatT>(0.0));
		axpyReal(next, mdaggMCoeff, applied);
		current = next;
	}

	out = current;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::correctExponentialEigenvalue(
	double filteredEigenvalue,
	const TRLanExponentialFilterParams &filter)
{
	if (!std::isfinite(filteredEigenvalue)) {
		throw std::runtime_error(stdLogger.fatal(
			"Cannot correct a non-finite exponential-filtered Ritz value"));
	}

	double root = 0.0;
	if (filteredEigenvalue < 0.0) {
		if (filter.order % 2 == 0) {
			throw std::runtime_error(stdLogger.fatal(
				"Negative Ritz value cannot be inverted through an even-order exponential filter"));
		}
		root = -std::pow(-filteredEigenvalue, 1.0 / static_cast<double>(filter.order));
	} else {
		root = std::pow(filteredEigenvalue, 1.0 / static_cast<double>(filter.order));
	}

	// This is Densecode's EVPolyCorrect formula, with theta represented by
	// filteredEigenvalue:
	//   lambda = n * (beta - theta^(1/n)) / (alpha * beta).
	const double corrected =
		static_cast<double>(filter.order) * (filter.beta - root)
		/ (filter.alpha * filter.beta);
	if (!std::isfinite(corrected)) {
		throw std::runtime_error(stdLogger.fatal(
			"Exponential-filtered Ritz value correction produced a non-finite eigenvalue"));
	}
	return corrected;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::zero(Spinor_internal &vec)
{
	vec *= COMPLEX(floatT)(static_cast<floatT>(0.0), static_cast<floatT>(0.0));
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::axpyReal(
	Spinor_internal &out,
	double coeff,
	const Spinor_internal &in)
{
	// axpyThis has both real and complex overloads in SIMulateQCD. This helper
	// makes the real-valued Ritz coefficient case explicit at call sites.
	out.axpyThis(static_cast<floatT>(coeff), in);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::fullReorthogonalize(
	Spinor_internal &vec,
	std::vector<Spinor_internal> &basis,
	int nBasis)
{
	for (int pass = 0; pass < 2; ++pass) {
		for (int i = 0; i < nBasis; ++i) {
			// Remove the component along every existing basis vector. Two passes
			// of modified Gram-Schmidt are cheap compared with a filtered
			// operator application and are much safer after a high-order filter.
			const COMPLEX(double) proj = basis[i].dotProduct(vec);
			vec.axpyThis(COMPLEX(floatT)(static_cast<floatT>(-proj.cREAL), static_cast<floatT>(-proj.cIMAG)), basis[i]);
		}
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::normalizeOrThrow(
	Spinor_internal &vec,
	const char *errorMsg)
{
	const double n2 = vec.realdotProduct(vec);
	if (!std::isfinite(n2) || n2 <= std::numeric_limits<double>::epsilon()) {
		throw std::runtime_error(stdLogger.fatal(errorMsg));
	}

	const double invNorm = 1.0 / std::sqrt(n2);
	vec *= COMPLEX(floatT)(static_cast<floatT>(invNorm), static_cast<floatT>(0.0));
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::validateParams(
	const TRLanRestartParams &params,
	int requestedEigenpairs,
	int krylovDim)
{
	if (params.maxRestarts < 0) {
		throw std::runtime_error(stdLogger.fatal("maxRestarts must be non-negative"));
	}
	if (params.residualTol <= 0.0 || !std::isfinite(params.residualTol)) {
		throw std::runtime_error(stdLogger.fatal("residualTol must be positive and finite"));
	}
	if (params.breakdownTol <= 0.0 || !std::isfinite(params.breakdownTol)) {
		throw std::runtime_error(stdLogger.fatal("breakdownTol must be positive and finite"));
	}
	if (params.thickRestartDim > 0 && params.thickRestartDim < requestedEigenpairs) {
		throw std::runtime_error(stdLogger.fatal("thickRestartDim must be at least requestedEigenpairs"));
	}
	if (params.thickRestartDim > 0 && params.thickRestartDim > krylovDim - 2) {
		throw std::runtime_error(stdLogger.fatal("thickRestartDim must be at most krylovDim - 2"));
	}
	if (params.chebyshev.enabled && params.exponential.enabled) {
		throw std::runtime_error(stdLogger.fatal(
			"Chebyshev and exponential Lanczos filters are mutually exclusive"));
	}
	if (params.chebyshev.enabled) {
		if (params.chebyshev.order <= 0) {
			throw std::runtime_error(stdLogger.fatal("Enabled Chebyshev filter requires order > 0"));
		}
		if (!std::isfinite(params.chebyshev.lowerBound) || !std::isfinite(params.chebyshev.upperBound)) {
			throw std::runtime_error(stdLogger.fatal("Chebyshev bounds must be finite"));
		}
		if (params.chebyshev.upperBound <= params.chebyshev.lowerBound) {
			throw std::runtime_error(stdLogger.fatal("Chebyshev filter requires upperBound > lowerBound"));
		}
	}
	if (params.exponential.enabled) {
		if (params.exponential.order <= 0) {
			throw std::runtime_error(stdLogger.fatal(
				"Enabled exponential filter requires order > 0"));
		}
		if (!std::isfinite(params.exponential.alpha) || params.exponential.alpha <= 0.0) {
			throw std::runtime_error(stdLogger.fatal(
				"Exponential filter alpha must be positive and finite"));
		}
		if (!std::isfinite(params.exponential.beta) || params.exponential.beta <= 0.0) {
			throw std::runtime_error(stdLogger.fatal(
				"Exponential filter beta must be positive and finite"));
		}
		const double stageCoefficient =
			(params.exponential.alpha / static_cast<double>(params.exponential.order))
			* params.exponential.beta;
		if (!std::isfinite(stageCoefficient)) {
			throw std::runtime_error(stdLogger.fatal(
				"Exponential filter stage coefficient is not finite"));
		}
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::buildKrylovSpace(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	std::vector<Spinor_internal> &basis,
	int targetDim,
	double breakdownTol,
	const TRLanChebyshevFilterParams &chebyshev,
	const TRLanExponentialFilterParams &exponential)
{
	while (static_cast<int>(basis.size()) < targetDim) {
		const int j = static_cast<int>(basis.size()) - 1;
		Spinor_internal w(comm);
		// Apply A q_j, where A is MdaggM or the selected polynomial transform.
		applyFilteredOperator(comm, op, w, basis[j], chebyshev, exponential);

		// Full reorthogonalization is intentionally retained for restart
		// robustness. After a thick restart the first basis vectors are Ritz
		// combinations, so relying on a three-term recurrence alone is fragile.
		fullReorthogonalize(w, basis, j + 1);

		const double normW2 = w.realdotProduct(w);
		if (!std::isfinite(normW2) || normW2 < 0.0) {
			throw std::runtime_error(stdLogger.fatal(
				"Lanczos operator application produced a non-finite vector norm"));
		}
		const double normW = std::sqrt(normW2);
		if (normW < breakdownTol) {
			// A tiny norm means the Krylov space has stopped growing. In exact
			// arithmetic this is a happy breakdown; numerically it is safest to
			// stop extending this cycle.
			break;
		}

		// Normalize the new direction and append it to the basis.
		basis.emplace_back(comm);
		basis.back() = w;
		basis.back() *= COMPLEX(floatT)(static_cast<floatT>(1.0 / normW), static_cast<floatT>(0.0));
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::projectedMatrix(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	std::vector<Spinor_internal> &basis,
	const TRLanChebyshevFilterParams &chebyshev,
	const TRLanExponentialFilterParams &exponential,
	std::vector<std::vector<double>> &matrix)
{
	const int n = static_cast<int>(basis.size());
	matrix.assign(n, std::vector<double>(n, 0.0));

	for (int col = 0; col < n; ++col) {
		Spinor_internal w(comm);
		// Compute A q_col once, then take dot products with all previous basis
		// vectors. Symmetry fills the lower triangle without another operator
		// application.
			applyFilteredOperator(comm, op, w, basis[col], chebyshev, exponential);
			for (int row = 0; row <= col; ++row) {
				const COMPLEX(double) elem = basis[row].dotProduct(w);
				if (!std::isfinite(elem.cREAL)) {
					throw std::runtime_error(stdLogger.fatal(
						"Lanczos projected matrix contains a non-finite value"));
				}
				matrix[row][col] = elem.cREAL;
			matrix[col][row] = elem.cREAL;
		}
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::diagonalizeSymmetricDense(
	const std::vector<std::vector<double>> &matrix,
	std::vector<double> &d,
	std::vector<std::vector<double>> &z)
{
	const int n = static_cast<int>(matrix.size());
	// Copy the projected matrix because the Jacobi sweeps overwrite it while
	// driving off-diagonal entries to zero.
	std::vector<std::vector<double>> a = matrix;
	z.assign(n, std::vector<double>(n, 0.0));
	for (int i = 0; i < n; ++i) {
		z[i][i] = 1.0;
	}
	if (n <= 1) {
		d.assign(n, 0.0);
		if (n == 1) {
			d[0] = a[0][0];
		}
		return;
	}

	double matrixNorm2 = 0.0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			matrixNorm2 += a[i][j] * a[i][j];
		}
	}
	const double offdiagTol = 1e-12 * std::max(1.0, std::sqrt(matrixNorm2));
	bool converged = false;
	const int maxSweeps = std::max(50, 10 * n * n);
	for (int sweep = 0; sweep < maxSweeps; ++sweep) {
		// Find the largest remaining off-diagonal entry. A Jacobi rotation on
		// this pair reduces the Frobenius norm of the off-diagonal part.
		int p = 0;
		int q = 1;
		double maxOffdiag = 0.0;
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				const double value = std::fabs(a[i][j]);
				if (value > maxOffdiag) {
					maxOffdiag = value;
					p = i;
					q = j;
				}
			}
		}

		if (maxOffdiag <= offdiagTol) {
			converged = true;
			break;
		}

		// Compute a stable plane rotation that diagonalizes the p/q 2x2 block.
		const double tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
		const double t = ((tau >= 0.0) ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
		const double c = 1.0 / std::sqrt(1.0 + t * t);
		const double s = t * c;
		const double app = a[p][p];
		const double aqq = a[q][q];
		const double apq = a[p][q];

		a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
		a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
		a[p][q] = 0.0;
		a[q][p] = 0.0;

		// Rotate the matrix columns/rows p and q, keeping a symmetric storage
		// layout.
		for (int k = 0; k < n; ++k) {
			if (k == p || k == q) {
				continue;
			}
			const double akp = a[k][p];
			const double akq = a[k][q];
			a[k][p] = c * akp - s * akq;
			a[p][k] = a[k][p];
			a[k][q] = s * akp + c * akq;
			a[q][k] = a[k][q];
		}

		// Accumulate the same rotation into z. At convergence, each column z[:,i]
		// is an eigenvector of the projected matrix.
		for (int k = 0; k < n; ++k) {
			const double zkp = z[k][p];
			const double zkq = z[k][q];
			z[k][p] = c * zkp - s * zkq;
			z[k][q] = s * zkp + c * zkq;
		}
	}

	if (!converged) {
		double offdiagNorm2 = 0.0;
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				offdiagNorm2 += 2.0 * a[i][j] * a[i][j];
			}
		}
		if (std::sqrt(offdiagNorm2) > offdiagTol) {
			throw std::runtime_error(stdLogger.fatal("Dense projected eigensolver did not converge"));
		}
	}

	d.assign(n, 0.0);
	for (int i = 0; i < n; ++i) {
		d[i] = a[i][i];
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::sortDenseEigenpairsAscending(
	std::vector<double> &d,
	std::vector<std::vector<double>> &z)
{
	const int n = static_cast<int>(d.size());
	std::vector<int> order(n);
	for (int i = 0; i < n; ++i) {
		order[i] = i;
	}

	std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
		return d[lhs] < d[rhs];
	});

	// Reorder both eigenvalues and eigenvector columns together so the Ritz
	// vector construction still uses matching pairs.
	std::vector<double> dSorted(n, 0.0);
	std::vector<std::vector<double>> zSorted(n, std::vector<double>(n, 0.0));
	for (int col = 0; col < n; ++col) {
		const int src = order[col];
		dSorted[col] = d[src];
		for (int row = 0; row < n; ++row) {
			zSorted[row][col] = z[row][src];
		}
	}

	d.swap(dSorted);
	z.swap(zSorted);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::sortDenseEigenpairsDescending(
	std::vector<double> &d,
	std::vector<std::vector<double>> &z)
{
	const int n = static_cast<int>(d.size());
	std::vector<int> order(n);
	std::iota(order.begin(), order.end(), 0);

	std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
		return d[lhs] > d[rhs];
	});

	std::vector<double> dSorted(n, 0.0);
	std::vector<std::vector<double>> zSorted(n, std::vector<double>(n, 0.0));
	for (int col = 0; col < n; ++col) {
		const int src = order[col];
		dSorted[col] = d[src];
		for (int row = 0; row < n; ++row) {
			zSorted[row][col] = z[row][src];
		}
	}

	d.swap(dSorted);
	z.swap(zSorted);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::sortDenseEigenpairsByMagnitudeDescending(
	std::vector<double> &d,
	std::vector<std::vector<double>> &z)
{
	const int n = static_cast<int>(d.size());
	std::vector<int> order(n);
	for (int i = 0; i < n; ++i) {
		order[i] = i;
	}

	std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
		return std::fabs(d[lhs]) > std::fabs(d[rhs]);
	});

	// For Chebyshev filtering the desired low modes are amplified outside the
	// mapped unwanted interval, but their algebraic sign depends on polynomial
	// parity. Sorting by magnitude keeps both odd and even orders targeted at
	// the amplified subspace.
	std::vector<double> dSorted(n, 0.0);
	std::vector<std::vector<double>> zSorted(n, std::vector<double>(n, 0.0));
	for (int col = 0; col < n; ++col) {
		const int src = order[col];
		dSorted[col] = d[src];
		for (int row = 0; row < n; ++row) {
			zSorted[row][col] = z[row][src];
		}
	}

	d.swap(dSorted);
	z.swap(zSorted);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::buildRitzVectors(
	const std::vector<Spinor_internal> &basis,
	const std::vector<std::vector<double>> &z,
	int nKeep,
	CommunicationBase &comm,
	std::vector<Spinor_internal> &ritzVectors)
{
	ritzVectors.clear();
	ritzVectors.reserve(nKeep);
	const int n = static_cast<int>(basis.size());

	for (int ev = 0; ev < nKeep; ++ev) {
		Spinor_internal ritz(comm);
		ritz = basis[0];
		zero(ritz);

		// Form r_ev = Q z_ev as a linear combination of the Lanczos basis. This
		// is the high-level equivalent of Denscode's trlanMMLaunch matrix-vector
		// block multiplication.
		for (int j = 0; j < n; ++j) {
			const double coeff = z[j][ev];
			if (std::fabs(coeff) < std::numeric_limits<double>::epsilon()) {
				continue;
			}
			axpyReal(ritz, coeff, basis[j]);
		}

		normalizeOrThrow(ritz, "Could not normalize Ritz vector in thick restarted Lanczos");
		ritzVectors.emplace_back(comm);
		ritzVectors.back() = ritz;
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::physicalResidual(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &vec,
	double &lambda)
{
	Spinor_internal applied(comm);
	// Always use the original operator here, even if the Chebyshev filter was
	// used internally. This is the only residual that certifies the physical
	// eigenpair.
	applyMdaggMSingle(comm, op, applied, vec);
	const COMPLEX(double) numerator = vec.dotProduct(applied);
	const double denominator = vec.realdotProduct(vec);
	if (!std::isfinite(numerator.cREAL) || !std::isfinite(denominator)
		|| denominator <= std::numeric_limits<double>::epsilon()) {
		throw std::runtime_error(stdLogger.fatal("Cannot compute residual of zero spinor"));
	}
	lambda = numerator.cREAL / denominator;
	axpyReal(applied, -lambda, vec);
	const double residual2 = applied.realdotProduct(applied);
	if (!std::isfinite(residual2) || residual2 < 0.0) {
		throw std::runtime_error(stdLogger.fatal(
			"Physical eigenpair residual is not finite"));
	}
	return std::sqrt(residual2);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::physicalResidualForEigenvalue(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &vec,
	double lambda,
	double &rayleigh)
{
	Spinor_internal applied(comm);
	applyMdaggMSingle(comm, op, applied, vec);
	const COMPLEX(double) numerator = vec.dotProduct(applied);
	const double denominator = vec.realdotProduct(vec);
	if (!std::isfinite(numerator.cREAL) || !std::isfinite(denominator)
		|| denominator <= std::numeric_limits<double>::epsilon()) {
		throw std::runtime_error(stdLogger.fatal("Cannot compute residual of zero spinor"));
	}

	rayleigh = numerator.cREAL / denominator;

	// In the Densecode exponential path, lambda is the positive stored
	// eigenvalue mu, while SIMULATeQCD's raw operator convention gives
	// A_SIM v = -mu v for these modes. Therefore check A_SIM v + mu v.
	axpyReal(applied, lambda, vec);
	const double residual2 = applied.realdotProduct(applied);
	if (!std::isfinite(residual2) || residual2 < 0.0) {
		throw std::runtime_error(stdLogger.fatal(
			"Corrected physical eigenpair residual is not finite"));
	}
	return std::sqrt(residual2);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::rayleighQuotient(
	CommunicationBase &comm,
	LinearOperator<Spinor_external> &op,
	Spinor_internal &vec)
{
	double lambda = 0.0;
	physicalResidual(comm, op, vec, lambda);
	return lambda;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::pythag(double a, double b)
{
	const double absa = std::fabs(a);
	const double absb = std::fabs(b);
	if (absa > absb) {
		return absa * std::sqrt(1.0 + (absb / absa) * (absb / absa));
	}
	return (absb == 0.0 ? 0.0 : absb * std::sqrt(1.0 + (absa / absb) * (absa / absb)));
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::tqli(
	std::vector<double> &d,
	std::vector<double> &e,
	int n,
	std::vector<std::vector<double>> &z)
{
	for (int i = 2; i <= n; ++i) {
		e[i - 1] = e[i];
	}
	e[n] = 0.0;

	for (int l = 1; l <= n; ++l) {
		int iter = 0;
		int m;
		do {
			for (m = l; m <= n - 1; ++m) {
				const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
				if (std::fabs(e[m]) + dd == dd) {
					break;
				}
			}

			if (m != l) {
				if (iter++ >= 60) {
					throw std::runtime_error(stdLogger.fatal("tqli did not converge in Lanczos"));
				}

				double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
				double r = pythag(g, 1.0);
				g = d[m] - d[l] + e[l] / (g + ((g >= 0.0) ? std::fabs(r) : -std::fabs(r)));

				double s = 1.0;
				double c = 1.0;
				double p = 0.0;

				for (int i = m - 1; i >= l; --i) {
					const double f = s * e[i];
					const double b = c * e[i];
					r = pythag(f, g);
					e[i + 1] = r;

					if (r == 0.0) {
						d[i + 1] -= p;
						e[m] = 0.0;
						break;
					}

					s = f / r;
					c = g / r;
					g = d[i + 1] - p;
					r = (d[i] - g) * s + 2.0 * c * b;
					p = s * r;
					d[i + 1] = g + p;
					g = c * r - b;

					for (int k = 1; k <= n; ++k) {
						const double zk_i1 = z[k][i + 1];
						const double zk_i = z[k][i];
						z[k][i + 1] = s * zk_i + c * zk_i1;
						z[k][i] = c * zk_i - s * zk_i1;
					}
				}

				if (r == 0.0 && m - 1 >= l) {
					continue;
				}

				d[l] -= p;
				e[l] = g;
				e[m] = 0.0;
			}
		} while (m != l);
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::sortEigenpairsAscending(
	std::vector<double> &d,
	std::vector<std::vector<double>> &z,
	int n)
{
	std::vector<int> order(n);
	for (int i = 0; i < n; ++i) {
		order[i] = i + 1;
	}

	std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
		return d[lhs] < d[rhs];
	});

	std::vector<double> dSorted(n + 1, 0.0);
	std::vector<std::vector<double>> zSorted(n + 1, std::vector<double>(n + 1, 0.0));

	for (int col = 1; col <= n; ++col) {
		const int srcCol = order[col - 1];
		dSorted[col] = d[srcCol];
		for (int row = 1; row <= n; ++row) {
			zSorted[row][col] = z[row][srcCol];
		}
	}

	d.swap(dSorted);
	z.swap(zSorted);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::solveProjectedSystem(
	const std::vector<Spinor_internal> &basis,
	const std::vector<double> &alpha,
	const std::vector<double> &beta,
	int n,
	int requestedEigenpairs,
	std::vector<Spinor_internal> &eigenvectors,
	std::vector<double> &eigenvalues,
	CommunicationBase &comm)
{
	std::vector<double> d(n + 1, 0.0);
	std::vector<double> e(n + 1, 0.0);
	std::vector<std::vector<double>> z(n + 1, std::vector<double>(n + 1, 0.0));

	for (int i = 1; i <= n; ++i) {
		d[i] = alpha[i - 1];
		if (i < n) {
			e[i + 1] = beta[i - 1];
		}
		z[i][i] = 1.0;
	}

	tqli(d, e, n, z);
	sortEigenpairsAscending(d, z, n);

	const int nKeep = std::min(requestedEigenpairs, n);
	eigenvectors.clear();
	eigenvalues.clear();
	eigenvectors.reserve(nKeep);
	eigenvalues.reserve(nKeep);

	for (int ev = 0; ev < nKeep; ++ev) {
		Spinor_internal ritz(comm);
		ritz = basis[0];
		ritz *= COMPLEX(floatT)(static_cast<floatT>(0.0), static_cast<floatT>(0.0));

		for (int j = 0; j < n; ++j) {
			const double coeff = z[j + 1][ev + 1];
			if (std::fabs(coeff) < std::numeric_limits<double>::epsilon()) {
				continue;
			}
			ritz.axpyThis(static_cast<floatT>(coeff), basis[j]);
		}

		normalizeOrThrow(ritz, "Could not normalize Ritz vector in Lanczos");

		eigenvectors.emplace_back(comm);
		eigenvectors.back() = ritz;
		eigenvalues.push_back(d[ev + 1]);
	}
}

#define LANCZOS_INIT_PLHHSN(floatT,LO,HaloDepth,HaloDepthSpin,STACKS) \
template class TRLanSpinorSolver<floatT, false, LO, HaloDepthSpin, STACKS>; \
template class TRLanSpinorSolver<floatT, true,  LO, HaloDepthSpin, STACKS>;
INIT_PLHHSN(LANCZOS_INIT_PLHHSN)
