/*
 * Minimal coupled-5D MDWF multishift-CG scaffold.
 *
 * This is intentionally isolated from the existing inverter and RHMC modules.
 * It exposes a multishift interface for systems
 *
 *     (A + sigma_i) x_i = b
 *
 * while preserving the coupled 5D inner product where Spinorfield stacks are
 * the physical fifth dimension Ls.
 *
 * Two strategies:
 *   Independent (default, validated first): each shift solved with its own
 *     coupled-5D CG; bit-reproducible against single-shift solves.
 *   Simultaneous: one Krylov sequence for all shifts (Jegerlehner's
 *     multishift CG; the recurrence of MILC's multi-mass CG in the sign
 *     convention used here). The base CG runs on the smallest shift sigma_0,
 *     which converges slowest; shift i follows with s_i = sigma_i - sigma_0 >= 0
 *     through
 *       zeta_{k+1} = zeta_k zeta_{k-1} a_{k-1}
 *                    / (a_k b_{k-1} (zeta_{k-1} - zeta_k) + zeta_{k-1} a_{k-1} (1 + s_i a_k)),
 *       a_k^i = a_k zeta_{k+1} / zeta_k,  x_i += a_k^i p_i,
 *       b_k^i = b_k (zeta_{k+1} / zeta_k)^2,  p_i = zeta_{k+1} r_{k+1} + b_k^i p_i,
 *     with base CG step a_k and direction coefficient b_k. The residual of
 *     shift i is zeta_i r; a shift is frozen once zeta_i^2 |r|^2 meets the
 *     target. Cost is about one base solve; memory is 2 * shifts + 3 spinors.
 *     Agrees with Independent to solver precision, not bit for bit.
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFCoupledSolverAdapter.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


/*
 * Fused shift update of the simultaneous multishift CG for up to MaxShifts shifts per launch:
 *   x_i += a_i d_i,   d_i = z_i r + b_i d_i,
 * one pass over r and over each x_i, d_i (separate axpy / scale / axpy kernels: 8 passes per shift plus one r read).
 */
template<class Spinor>
struct MDWFMultiShiftUpdateKernel {
    using T = MDWFSpinorTraits<Spinor>;
    using floatT = typename T::floatT;
    using Acc = VectArrayAcc<floatT, T::elems>;
    static constexpr size_t MaxShifts = 8;   // 2 * 8 accessors of 12 pointers: kernel parameters stay below 4 KB

    Acc _r;
    Acc _x[MaxShifts];
    Acc _d[MaxShifts];
    floatT _a[MaxShifts], _b[MaxShifts], _z[MaxShifts];
    int _n;

    explicit MDWFMultiShiftUpdateKernel(const Spinor &r) : _r(r.getAccessor()), _n(0) {}

    void add(Spinor &x, Spinor &d, double a, double b, double z) {
        _x[_n] = x.getAccessor();
        _d[_n] = d.getAccessor();
        _a[_n] = static_cast<floatT>(a);
        _b[_n] = static_cast<floatT>(b);
        _z[_n] = static_cast<floatT>(z);
        _n++;
    }

    bool full() const {
        return _n == static_cast<int>(MaxShifts);
    }

    __host__ __device__ void operator()(gSiteStack site) {
        const Vect<floatT, T::elems> r = _r.getElement(site);
        for (int i = 0; i < _n; i++) {
            const Vect<floatT, T::elems> d = _d[i].getElement(site);
            Vect<floatT, T::elems> x = _x[i].getElement(site);
            x += _a[i] * d;
            _x[i].setElement(site, x);
            _d[i].setElement(site, _z[i] * r + _b[i] * d);
        }
    }

    void launch() {
        if (_n == 0) {
            return;
        }
        typedef GIndexer<T::layout, T::halo> GInd;
        const size_t sites = (T::layout == All) ? GInd::getLatData().vol4 : GInd::getLatData().sizeh;
        static_assert(T::stacks <= DEFAULT_NBLOCKS, "fused multishift update: one block row per stack");
        CalcGSiteStack<T::layout, T::halo> calcSiteStack;
        iterateFunctorNoReturn<true, (DEFAULT_NBLOCKS / T::stacks) * T::stacks>(*this, calcSiteStack, sites, T::stacks);
        _n = 0;
    }
};

template<class floatT>
struct MDWFCoupledMultiShiftCGResult {
    int iterations;
    double residue;
    bool converged;
};

template<class floatT>
struct MDWFCoupledMultiShiftCGResults {
    std::vector<MDWFCoupledMultiShiftCGResult<floatT>> shifts;

    bool converged() const {
        for (const auto &result : shifts) {
            if (!result.converged) {
                return false;
            }
        }
        return true;
    }
};

enum class MDWFMultiShiftStrategy {
    Independent,
    Simultaneous
};

template<class floatT, class CoupledAdapter, size_t BlockSize = 64>
class MDWFCoupledMultiShiftCG {
public:
    using Spinor = typename CoupledAdapter::Spinor;

    explicit MDWFCoupledMultiShiftCG(MDWFMultiShiftStrategy strategy = MDWFMultiShiftStrategy::Independent)
        : _strategy(strategy) {}

    MDWFCoupledMultiShiftCGResults<floatT> invert(CoupledAdapter &adapter,
                                                  const std::vector<Spinor *> &spinor_out,
                                                  const Spinor &spinor_in,
                                                  const std::vector<floatT> &sigma,
                                                  int max_iter,
                                                  double precision,
                                                  bool update = true) const {
        if (spinor_out.size() != sigma.size() || sigma.empty()) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF coupled multishift CG requires matching nonempty solution and shift vectors"));
        }

        for (size_t shift = 0; shift < sigma.size(); shift++) {
            if (spinor_out[shift] == nullptr) {
                throw std::runtime_error(stdLogger.fatal("MDWF coupled multishift CG received a null solution pointer"));
            }
            if (!std::isfinite(static_cast<double>(sigma[shift])) || sigma[shift] < static_cast<floatT>(0.0)) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF coupled multishift CG requires finite nonnegative shifts, got sigma = ", sigma[shift]));
            }
        }

        if (_strategy == MDWFMultiShiftStrategy::Simultaneous) {
            return invertSimultaneous(adapter, spinor_out, spinor_in, sigma, max_iter, precision, update);
        }

        MDWFCoupledMultiShiftCGResults<floatT> results;
        results.shifts.reserve(sigma.size());
        for (size_t shift = 0; shift < sigma.size(); shift++) {
            results.shifts.push_back(invertShift(
                adapter, *spinor_out[shift], spinor_in, sigma[shift], max_iter, precision, update));
        }
        return results;
    }

    MDWFMultiShiftStrategy strategy() const {
        return _strategy;
    }

private:
    MDWFMultiShiftStrategy _strategy;

    MDWFCoupledMultiShiftCGResults<floatT> invertSimultaneous(CoupledAdapter &adapter,
                                                              const std::vector<Spinor *> &spinor_out,
                                                              const Spinor &spinor_in,
                                                              const std::vector<floatT> &sigma,
                                                              int max_iter,
                                                              double precision,
                                                              bool update) const {
        const size_t nShift = sigma.size();
        const double sigma0 = static_cast<double>(*std::min_element(sigma.begin(), sigma.end()));

        Spinor residual(spinor_in.getComm(), "MDWF_mscg_residual");
        Spinor search(spinor_in.getComm(), "MDWF_mscg_search");
        Spinor applied(spinor_in.getComm(), "MDWF_mscg_applied");
        std::vector<std::unique_ptr<Spinor>> directions;
        directions.reserve(nShift);
        for (size_t i = 0; i < nShift; i++) {
            directions.emplace_back(new Spinor(spinor_in.getComm(), "MDWF_mscg_dir_" + std::to_string(i)));
        }

        residual = spinor_in;
        search = spinor_in;
        for (size_t i = 0; i < nShift; i++) {
            *spinor_out[i] = static_cast<floatT>(0.0) * spinor_in;
            *directions[i] = spinor_in;
        }

        std::vector<double> shiftOffset(nShift);
        std::vector<double> zeta(nShift, 1.0);
        std::vector<double> zetaOld(nShift, 1.0);
        std::vector<bool> active(nShift, true);
        MDWFCoupledMultiShiftCGResults<floatT> results;
        results.shifts.assign(nShift, {0, 0.0, false});
        for (size_t i = 0; i < nShift; i++) {
            shiftOffset[i] = static_cast<double>(sigma[i]) - sigma0;
        }

        const double sourceNorm = adapter.norm2(residual);
        const double normScale = std::max(sourceNorm, 1.0);
        const double targetNorm = precision * precision * normScale;
        double residualNorm = sourceNorm;

        auto markConverged = [&](size_t i, int iterations) {
            active[i] = false;
            results.shifts[i] = {iterations, std::sqrt(zeta[i] * zeta[i] * residualNorm / normScale), true};
        };
        for (size_t i = 0; i < nShift; i++) {
            if (residualNorm <= targetNorm) {
                markConverged(i, 0);
            }
        }

        double aOld = 1.0;
        double bOld = 0.0;
        int iteration = 0;
        for (; iteration < max_iter && std::find(active.begin(), active.end(), true) != active.end(); iteration++) {
            mdwfPrepareInput(adapter, search);
            adapter.apply(applied, search, false);
            applied.template axpyThisB<BlockSize>(static_cast<floatT>(sigma0), search);

            const double denominator = real<double>(adapter.dotProduct5D(search, applied));
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF simultaneous multishift CG encountered a nonpositive search denominator"));
            }
            const double a = residualNorm / denominator;

            std::vector<double> zetaNew(nShift, 0.0);
            std::vector<double> aShift(nShift, 0.0);
            for (size_t i = 0; i < nShift; i++) {
                if (!active[i]) {
                    continue;
                }
                zetaNew[i] = zeta[i] * zetaOld[i] * aOld
                             / (a * bOld * (zetaOld[i] - zeta[i]) + zetaOld[i] * aOld * (1.0 + shiftOffset[i] * a));
                aShift[i] = a * zetaNew[i] / zeta[i];
            }

            residual.template axpyThisB<BlockSize>(static_cast<floatT>(-a), applied);
            const double nextResidualNorm = adapter.norm2(residual);
            const double b = nextResidualNorm / residualNorm;

            // x_i += aShift_i d_i (old d_i), then d_i = zetaNew_i r + bShift_i d_i, fused over the active shifts.
            MDWFMultiShiftUpdateKernel<Spinor> update(residual);
            for (size_t i = 0; i < nShift; i++) {
                if (!active[i]) {
                    continue;
                }
                const double ratio = zetaNew[i] / zeta[i];
                const double bShift = b * ratio * ratio;
                update.add(*spinor_out[i], *directions[i], aShift[i], bShift, zetaNew[i]);
                if (update.full()) {
                    update.launch();
                }
                zetaOld[i] = zeta[i];
                zeta[i] = zetaNew[i];
            }
            update.launch();
            search *= COMPLEX(floatT)(static_cast<floatT>(b), 0.0);
            search += residual;

            residualNorm = nextResidualNorm;
            aOld = a;
            bOld = b;
            for (size_t i = 0; i < nShift; i++) {
                if (active[i] && zeta[i] * zeta[i] * residualNorm <= targetNorm) {
                    markConverged(i, iteration + 1);
                }
            }
        }

        for (size_t i = 0; i < nShift; i++) {
            if (active[i]) {
                results.shifts[i] = {iteration, std::sqrt(zeta[i] * zeta[i] * residualNorm / normScale), false};
            }
            if (update) {
                mdwfUpdateHalo(*spinor_out[i]);
            }
        }
        return results;
    }
    MDWFCoupledMultiShiftCGResult<floatT> invertShift(CoupledAdapter &adapter,
                                                     Spinor &spinor_out,
                                                     const Spinor &spinor_in,
                                                     floatT sigma,
                                                     int max_iter,
                                                     double precision,
                                                     bool update) const {
        Spinor residual(spinor_in.getComm(), "MDWF_coupled_multishift_cg_residual");
        Spinor search(spinor_in.getComm(), "MDWF_coupled_multishift_cg_search");
        Spinor operator_search(spinor_in.getComm(), "MDWF_coupled_multishift_cg_operator_search");

        spinor_out = static_cast<floatT>(0.0) * spinor_in;
        residual = spinor_in;
        search = residual;

        const double source_norm = adapter.norm2(residual);
        const double target_norm = precision * precision * std::max(source_norm, 1.0);
        double residual_norm = source_norm;

        if (residual_norm <= target_norm) {
            if (update) {
                mdwfUpdateHalo(spinor_out);
            }
            return {0, std::sqrt(residual_norm / std::max(source_norm, 1.0)), true};
        }

        for (int iteration = 0; iteration < max_iter; iteration++) {
            mdwfPrepareInput(adapter, search);
            adapter.apply(operator_search, search, false);
            operator_search.template axpyThisB<BlockSize>(sigma, search);

            const double denominator = real<double>(adapter.dotProduct5D(search, operator_search));
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF coupled multishift CG encountered a nonpositive search denominator"));
            }

            const floatT alpha = static_cast<floatT>(residual_norm / denominator);
            spinor_out.template axpyThisB<BlockSize>(alpha, search);
            residual.template axpyThisB<BlockSize>(-alpha, operator_search);

            const double next_residual_norm = adapter.norm2(residual);
            if (next_residual_norm <= target_norm) {
                if (update) {
                    mdwfUpdateHalo(spinor_out);
                }
                return {iteration + 1,
                        std::sqrt(next_residual_norm / std::max(source_norm, 1.0)),
                        true};
            }

            const floatT beta = static_cast<floatT>(next_residual_norm / residual_norm);
            search *= COMPLEX(floatT)(beta, 0.0);
            search += residual;
            residual_norm = next_residual_norm;
        }

        if (update) {
            mdwfUpdateHalo(spinor_out);
        }
        return {max_iter, std::sqrt(residual_norm / std::max(source_norm, 1.0)), false};
    }
};
