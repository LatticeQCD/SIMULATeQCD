#include "lanczos.h"

#include "../../define.h"
#include "../../base/math/random.h"
#include "../../spinor/eigenpairs.h"


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
	if constexpr (LatticeLayout == Layout::All) {
		throw std::runtime_error(stdLogger.fatal("TRLanSpinorSolver currently supports only Even/Odd layout"));
	}

	if (requestedEigenpairs <= 0) {
		eigenvectors.clear();
		eigenvalues.clear();
		return;
	}

	const int m = (krylovDim > 0) ? std::max(krylovDim, requestedEigenpairs + 2) : std::max(2 * requestedEigenpairs + 8, requestedEigenpairs + 2);

	std::vector<Spinor_internal> q;
	q.reserve(m);

	std::vector<double> alpha(m, 0.0);
	std::vector<double> beta(m, 0.0);

	grnd_state<false> h_rand;
	h_rand.make_rng_state(seed);

	q.emplace_back(comm);
	q[0].gauss(h_rand.state);
	normalizeOrThrow(q[0], "Could not normalize start vector in Lanczos");

	int usedDim = 1;
	for (int j = 0; j < m; ++j) {
		Spinor_internal w(comm);
		applyMdaggMSingle(comm, op, w, q[j]);

		if (j > 0) {
			w.axpyThis(static_cast<floatT>(-beta[j - 1]), q[j - 1]);
		}

		const COMPLEX(double) qAw = q[j].dotProduct(w);
		alpha[j] = qAw.cREAL;
		w.axpyThis(COMPLEX(floatT)(static_cast<floatT>(-qAw.cREAL), static_cast<floatT>(-qAw.cIMAG)), q[j]);

		fullReorthogonalize(w, q, j + 1);

		const double normW = std::sqrt(std::max(w.realdotProduct(w), 0.0));
		beta[j] = normW;

		usedDim = j + 1;
		if (j + 1 >= m || normW < breakdownTol) {
			break;
		}

		q.emplace_back(comm);
		q[j + 1] = w;
		q[j + 1] *= COMPLEX(floatT)(static_cast<floatT>(1.0 / normW), static_cast<floatT>(0.0));
	}

	solveProjectedSystem(q, alpha, beta, usedDim, requestedEigenpairs, eigenvectors, eigenvalues, comm);
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

	for (size_t stack = 0; stack < NStacks; ++stack) {
		inStacked.copyFromStackToStack(in, stack, 0);
	}

	op.applyMdaggM(outStacked, inStacked, true);
	out.copyFromStackToStack(outStacked, 0, 0);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::fullReorthogonalize(
	Spinor_internal &vec,
	std::vector<Spinor_internal> &basis,
	int nBasis)
{
	for (int i = 0; i < nBasis; ++i) {
		const COMPLEX(double) proj = basis[i].dotProduct(vec);
		vec.axpyThis(COMPLEX(floatT)(static_cast<floatT>(-proj.cREAL), static_cast<floatT>(-proj.cIMAG)), basis[i]);
	}
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>::normalizeOrThrow(
	Spinor_internal &vec,
	const char *errorMsg)
{
	const double n2 = vec.realdotProduct(vec);
	if (n2 <= std::numeric_limits<double>::epsilon()) {
		throw std::runtime_error(stdLogger.fatal(errorMsg));
	}

	const double invNorm = 1.0 / std::sqrt(n2);
	vec *= COMPLEX(floatT)(static_cast<floatT>(invNorm), static_cast<floatT>(0.0));
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
