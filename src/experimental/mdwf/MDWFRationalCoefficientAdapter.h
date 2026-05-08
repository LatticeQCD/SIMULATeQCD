/*
 * MDWF rational-coefficient adapter scaffold.
 *
 * This file maps an explicitly supplied partial-fraction rational form
 *
 *     r(x) = constant + sum_i numerator_i / (x + denominator_i)
 *
 * into the MDWFRationalCoefficients representation used by
 * MDWFRationalOperator.  It intentionally does not parse RHMC files, assign
 * determinant powers, define Hasenbusch factors, or choose heatbath/action/force
 * semantics.  Those meanings must be supplied by a future MDWF parameter layer.
 */

#pragma once

#include "MDWFRationalOperator.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

enum class MDWFRationalCoefficientRole {
    Heatbath,
    Action,
    Force
};

inline std::string mdwfRationalCoefficientRoleName(MDWFRationalCoefficientRole role) {
    switch (role) {
    case MDWFRationalCoefficientRole::Heatbath:
        return "heatbath";
    case MDWFRationalCoefficientRole::Action:
        return "action";
    case MDWFRationalCoefficientRole::Force:
        return "force";
    }
    return "unknown";
}

template<class floatT>
struct MDWFExplicitRationalInput {
    std::string name;
    MDWFRationalCoefficientRole role;
    floatT constant;
    std::vector<floatT> numerator;
    std::vector<floatT> denominator;
};

template<class floatT>
MDWFRationalCoefficients<floatT> makeMDWFRationalCoefficients(const MDWFExplicitRationalInput<floatT> &input) {
    if (input.numerator.size() != input.denominator.size()) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational coefficient adapter requires matching numerator and denominator vectors for ",
            input.name));
    }
    if (!std::isfinite(static_cast<double>(input.constant))) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational coefficient adapter received a nonfinite constant for ", input.name));
    }

    MDWFRationalCoefficients<floatT> coefficients;
    coefficients.constant = input.constant;
    coefficients.numerator = input.numerator;
    coefficients.shift = input.denominator;

    coefficients.validate();
    return coefficients;
}

template<class floatT>
floatT evaluateMDWFRationalScalar(floatT x, const MDWFRationalCoefficients<floatT> &coefficients) {
    if (!std::isfinite(static_cast<double>(x))) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational scalar evaluation received a nonfinite argument"));
    }

    coefficients.validate();

    floatT result = coefficients.constant;
    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        const floatT denominator = x + coefficients.shift[term];
        if (denominator == static_cast<floatT>(0.0)) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational scalar evaluation encountered a zero denominator"));
        }
        result += coefficients.numerator[term] / denominator;
    }
    return result;
}
