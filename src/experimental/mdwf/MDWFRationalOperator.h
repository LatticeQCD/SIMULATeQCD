/*
 * Explicit coupled-5D MDWF rational-operator scaffold.
 *
 * This wrapper applies
 *
 *     y = c0 x + sum_i numerator_i (A + shift_i)^(-1) x
 *
 * using an already supplied coupled-5D solver adapter and the isolated
 * MDWFCoupledMultiShiftCG interface.  The coefficients are passed explicitly:
 * this file does not define determinant powers, pseudofermion conventions,
 * RHMC/HMC integration, force terms, or rational-approximation generation.
 */

#pragma once

#include "MDWFCoupledMultiShiftCG.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT>
struct MDWFRationalCoefficients {
    floatT constant;
    std::vector<floatT> numerator;
    std::vector<floatT> shift;

    void validate() const {
        if (numerator.size() != shift.size()) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational operator requires matching numerator and shift vectors"));
        }
        if (!std::isfinite(static_cast<double>(constant))) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational operator received a nonfinite constant coefficient"));
        }
        for (size_t term = 0; term < shift.size(); term++) {
            if (!std::isfinite(static_cast<double>(numerator[term]))
                || !std::isfinite(static_cast<double>(shift[term]))
                || shift[term] < static_cast<floatT>(0.0)) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF rational operator requires finite numerators and finite nonnegative shifts"));
            }
        }
    }
};

template<class floatT, class CoupledAdapter, size_t BlockSize = 64>
class MDWFRationalOperator {
public:
    using Spinor = typename CoupledAdapter::Spinor;

private:
    MDWFRationalCoefficients<floatT> _coefficients;
    int _max_iter;
    double _precision;
    std::string _name;

public:
    MDWFRationalOperator(MDWFRationalCoefficients<floatT> coefficients,
                         int max_iter,
                         double precision,
                         std::string name = "MDWF_rational_operator")
        : _coefficients(coefficients),
          _max_iter(max_iter),
          _precision(precision),
          _name(name) {
        _coefficients.validate();
        if (_max_iter <= 0 || _precision <= 0.0 || !std::isfinite(_precision)) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational operator requires positive max_iter and precision"));
        }
    }

    MDWFCoupledMultiShiftCGResults<floatT> apply(CoupledAdapter &adapter,
                                                Spinor &spinor_out,
                                                const Spinor &spinor_in,
                                                bool update = true) const {
        spinor_out = _coefficients.constant * spinor_in;

        if (_coefficients.shift.empty()) {
            if (update) {
                spinor_out.updateAll();
            }
            return {};
        }

        std::vector<std::unique_ptr<Spinor>> shifted_solutions;
        std::vector<Spinor *> shifted_solution_ptrs;
        shifted_solutions.reserve(_coefficients.shift.size());
        shifted_solution_ptrs.reserve(_coefficients.shift.size());

        for (size_t term = 0; term < _coefficients.shift.size(); term++) {
            shifted_solutions.emplace_back(new Spinor(
                spinor_in.getComm(), _name + "_shift_solution_" + std::to_string(term)));
            shifted_solution_ptrs.push_back(shifted_solutions.back().get());
        }

        MDWFCoupledMultiShiftCG<floatT, CoupledAdapter, BlockSize> multishift_cg;
        MDWFCoupledMultiShiftCGResults<floatT> results = multishift_cg.invert(
            adapter, shifted_solution_ptrs, spinor_in, _coefficients.shift, _max_iter, _precision, true);

        for (size_t term = 0; term < _coefficients.numerator.size(); term++) {
            spinor_out.template axpyThisB<BlockSize>(_coefficients.numerator[term], *shifted_solutions[term]);
        }

        if (update) {
            spinor_out.updateAll();
        }
        return results;
    }

    const MDWFRationalCoefficients<floatT> &coefficients() const {
        return _coefficients;
    }
};
