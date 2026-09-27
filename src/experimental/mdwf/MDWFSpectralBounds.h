/*
 * Test-only spectral-bound estimates for Hermitian positive MDWF operators
 * (typically M^\dagger M), used to choose the interval [lambda_low, lambda_high]
 * of rational approximations (AlgRemez).
 *
 * mdwfLanczosExtremes runs plain Lanczos with the coupled 5D inner product of
 * an MDWFCoupledSolverAdapter-like adapter (apply, dotProduct5D, norm2) and
 * records the extreme Ritz values of the Lanczos tridiagonal matrix T_k at
 * requested step counts k. The lowest Ritz value converges to lambda_min from
 * above and the highest to lambda_max from below, so the true spectrum can
 * extend slightly beyond the reported range: choose the approximation
 * interval with a safety margin. No reorthogonalization is done; loss of
 * orthogonality only creates duplicate ("ghost") Ritz values inside the
 * spectrum and does not move the extremes outside it.
 *
 * The extreme eigenvalues of T_k are computed on the host by Sturm-sequence
 * bisection. Single rank or multi rank as far as the adapter's reductions go.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Number of eigenvalues below x of the symmetric tridiagonal matrix with diagonal a and off-diagonal b.
inline int mdwfTridiagonalCountBelow(const std::vector<double> &a, const std::vector<double> &b, double x) {
    int count = 0;
    double d = 1.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double off = (i == 0) ? 0.0 : b[i - 1] * b[i - 1] / d;
        d = a[i] - x - off;
        if (d == 0.0) {
            d = -std::numeric_limits<double>::epsilon() * (std::abs(a[i]) + std::abs(x) + 1.0);
        }
        if (d < 0.0) {
            count++;
        }
    }
    return count;
}

// k-th smallest eigenvalue (k = 1 ... n) of the symmetric tridiagonal matrix by bisection.
inline double mdwfTridiagonalEigenvalue(const std::vector<double> &a, const std::vector<double> &b, int k) {
    double lo = std::numeric_limits<double>::max();
    double hi = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < a.size(); i++) {
        const double radius = (i > 0 ? std::abs(b[i - 1]) : 0.0) + (i + 1 < a.size() ? std::abs(b[i]) : 0.0);
        lo = std::min(lo, a[i] - radius);
        hi = std::max(hi, a[i] + radius);
    }
    for (int iter = 0; iter < 200 && hi - lo > 1e-15 * std::max(1.0, std::abs(hi)); iter++) {
        const double mid = 0.5 * (lo + hi);
        if (mdwfTridiagonalCountBelow(a, b, mid) >= k) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return 0.5 * (lo + hi);
}

struct MDWFLanczosCheckpoint {
    int steps;
    double lambda_min;
    double lambda_max;
};

struct MDWFLanczosResult {
    std::vector<MDWFLanczosCheckpoint> checkpoints;
    int completed_steps;
    bool breakdown;
};

template<class Adapter, class Spinor>
MDWFLanczosResult mdwfLanczosExtremes(Adapter &adapter, const Spinor &start, int maxSteps,
                                      const std::vector<int> &checkpointSteps, const std::string &name) {
    Spinor previous(start.getComm(), name + "_lz_previous");
    Spinor current(start.getComm(), name + "_lz_current");
    Spinor work(start.getComm(), name + "_lz_work");

    std::vector<double> alpha;
    std::vector<double> beta;
    MDWFLanczosResult result{{}, 0, false};

    current = start;
    const double startNorm = std::sqrt(adapter.norm2(current));
    current *= COMPLEX(double)(1.0 / startNorm, 0.0);
    previous = 0.0 * current;

    auto record = [&](int steps) {
        const std::vector<double> offDiagonal(beta.begin(), beta.begin() + (steps - 1));
        const std::vector<double> diagonal(alpha.begin(), alpha.begin() + steps);
        result.checkpoints.push_back({steps, mdwfTridiagonalEigenvalue(diagonal, offDiagonal, 1),
                                      mdwfTridiagonalEigenvalue(diagonal, offDiagonal, steps)});
    };

    for (int step = 0; step < maxSteps; step++) {
        current.updateAll();
        adapter.apply(work, current, false);
        if (step > 0) {
            work.template axpyThisB<64>(-beta[step - 1], previous);
        }
        const double a = real<double>(adapter.dotProduct5D(current, work));
        alpha.push_back(a);
        work.template axpyThisB<64>(-a, current);
        const double b = std::sqrt(adapter.norm2(work));
        beta.push_back(b);
        result.completed_steps = step + 1;

        const bool isCheckpoint = std::find(checkpointSteps.begin(), checkpointSteps.end(), step + 1)
                                  != checkpointSteps.end();
        const bool breakdown = !(b > 1e-14 * std::max(1.0, std::abs(a)));
        if (isCheckpoint || breakdown || step + 1 == maxSteps) {
            if (result.checkpoints.empty() || result.checkpoints.back().steps != step + 1) {
                record(step + 1);
            }
        }
        if (breakdown) {
            result.breakdown = true;
            break;
        }

        previous = current;
        current = work;
        current *= COMPLEX(double)(1.0 / b, 0.0);
    }
    return result;
}
