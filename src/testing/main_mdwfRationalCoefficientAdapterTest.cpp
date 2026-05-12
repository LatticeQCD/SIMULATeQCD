/*
 * MDWF rational-coefficient adapter smoke test.
 *
 * This test uses a tiny explicit partial-fraction coefficient set and verifies
 * that it maps into MDWFRationalCoefficients without assigning RHMC determinant
 * powers or force/action semantics implicitly.  It avoids intentionally
 * triggering fatal validation logs, and does not construct spinors, apply
 * MDWF, call CG, or touch RHMC/HMC/force code.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void requireClose(double value, double expected, double tolerance, const std::string &label) {
    const double diff = std::abs(value - expected);
    if (diff > tolerance) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational coefficient adapter test failed for ", label,
            ": value = ", value,
            ", expected = ", expected,
            ", diff = ", diff));
    }
}

void runMDWFRationalCoefficientAdapterTest() {
    MDWFExplicitRationalInput<double> explicitInput{
        "tiny_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.25,
        {0.5, -0.125, 0.75},
        {0.0, 0.1, 0.3}
    };

    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(explicitInput);

    if (mdwfRationalCoefficientRoleName(MDWFRationalCoefficientRole::Heatbath) != "heatbath"
        || mdwfRationalCoefficientRoleName(explicitInput.role) != "action"
        || mdwfRationalCoefficientRoleName(MDWFRationalCoefficientRole::Force) != "force"
        || coefficients.numerator.size() != 3
        || coefficients.shift.size() != 3) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational coefficient adapter test failed basic metadata check"));
    }

    requireClose(coefficients.constant, 0.25, 1e-15, "constant");
    requireClose(coefficients.numerator[0], 0.5, 1e-15, "numerator[0]");
    requireClose(coefficients.numerator[1], -0.125, 1e-15, "numerator[1]");
    requireClose(coefficients.numerator[2], 0.75, 1e-15, "numerator[2]");
    requireClose(coefficients.shift[0], 0.0, 1e-15, "shift[0]");
    requireClose(coefficients.shift[1], 0.1, 1e-15, "shift[1]");
    requireClose(coefficients.shift[2], 0.3, 1e-15, "shift[2]");

    const double x = 2.0;
    const double expectedValue = 0.25 + 0.5 / (x + 0.0) - 0.125 / (x + 0.1) + 0.75 / (x + 0.3);
    requireClose(evaluateMDWFRationalScalar(x, coefficients), expectedValue, 1e-15, "scalar evaluation");

    rootLogger.info("MDWF rational coefficient adapter smoke test passed with terms = ",
                    coefficients.shift.size(),
                    ", value_at_2 = ", evaluateMDWFRationalScalar(x, coefficients));
    std::cout << "MDWF rational coefficient adapter smoke test passed with terms = "
              << coefficients.shift.size()
              << ", value_at_2 = " << evaluateMDWFRationalScalar(x, coefficients)
              << std::endl;
}

int main() {
    try {
        stdLogger.setVerbosity(INFO);
        runMDWFRationalCoefficientAdapterTest();
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
