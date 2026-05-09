/*
 * MDWF fermion-force workspace scaffold.
 *
 * This helper prepares the shifted fields needed by a future MDWF force:
 *
 *     chi_i = (M^\dagger M + sigma_i)^(-1) phi
 *     eta_i = M chi_i
 *
 * It intentionally does not accumulate gauge force, update momenta, call
 * RHMC/HMC, touch HISQ, or define force sign conventions.
 */

#pragma once

#include "MDWFCoupledCG.h"
#include "MDWFRationalOperator.h"
#include "MDWFShiftedNormalOperator.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template<class floatT>
struct MDWFFermionForceShiftWorkspaceInfo {
    floatT numerator;
    floatT shift;
    int iterations;
    double residue;
    bool converged;
};

template<class floatT,
         size_t HaloDepthGauge,
         size_t HaloDepthSpin,
         size_t Ls,
         class NormalOperatorT,
         class ForwardOperatorT,
         size_t BlockSize = 64>
class MDWFFermionForceWorkspace {
public:
    using Spinor = typename NormalOperatorT::Spinor;
    using ForwardSpinor = typename ForwardOperatorT::Spinor;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperatorT, floatT, BlockSize>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<floatT, HaloDepthGauge, HaloDepthSpin, Ls, ShiftedOperator>;

private:
    std::vector<std::unique_ptr<Spinor>> _chi;
    std::vector<std::unique_ptr<Spinor>> _eta;
    std::vector<MDWFFermionForceShiftWorkspaceInfo<floatT>> _shift_info;

public:
    MDWFFermionForceWorkspace() {
        static_assert(std::is_same<Spinor, ForwardSpinor>::value,
                      "MDWF force workspace requires matching normal and forward spinor types");
    }

    void prepare(NormalOperatorT &normal_operator,
                 ForwardOperatorT &forward_operator,
                 Spinor &phi,
                 const MDWFRationalCoefficients<floatT> &coefficients,
                 int max_iter,
                 double precision,
                 const std::string &name = "MDWF_fermion_force_workspace") {

        coefficients.validate();
        if (max_iter <= 0 || precision <= 0.0 || !std::isfinite(precision)) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF fermion force workspace requires positive max_iter and precision"));
        }

        _chi.clear();
        _eta.clear();
        _shift_info.clear();
        _chi.reserve(coefficients.shift.size());
        _eta.reserve(coefficients.shift.size());
        _shift_info.reserve(coefficients.shift.size());

        for (size_t term = 0; term < coefficients.shift.size(); term++) {
            _chi.emplace_back(new Spinor(phi.getComm(), name + "_chi_" + std::to_string(term)));
            _eta.emplace_back(new Spinor(phi.getComm(), name + "_eta_" + std::to_string(term)));

            ShiftedOperator shifted_operator(normal_operator, coefficients.shift[term],
                                             name + "_shifted_" + std::to_string(term));
            ShiftedAdapter shifted_adapter(shifted_operator);
            MDWFCoupledCG<floatT, ShiftedAdapter, BlockSize> cg;
            MDWFCoupledCGResult<floatT> solve_result
                = cg.invert(shifted_adapter, *_chi[term], phi, max_iter, precision, true);

            forward_operator.apply(*_eta[term], *_chi[term], true);

            _shift_info.push_back({
                coefficients.numerator[term],
                coefficients.shift[term],
                solve_result.iterations,
                solve_result.residue,
                solve_result.converged
            });
        }
    }

    size_t size() const {
        return _chi.size();
    }

    const Spinor &chi(size_t term) const {
        return *_chi.at(term);
    }

    const Spinor &eta(size_t term) const {
        return *_eta.at(term);
    }

    const std::vector<MDWFFermionForceShiftWorkspaceInfo<floatT>> &shiftInfo() const {
        return _shift_info;
    }

    bool converged() const {
        for (const auto &info : _shift_info) {
            if (!info.converged) {
                return false;
            }
        }
        return true;
    }
};
