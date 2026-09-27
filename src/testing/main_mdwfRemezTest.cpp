/*
 * MDWF Remez interface test (plain host C++, no GPU/MPI).
 *
 * Checks mdwfRemezPower (MDWFRemez.h, on top of the unchanged AlgRemez in
 * src/tools/rational_approx):
 *
 *   1. Reproduction of ratApprox output: x^(+-3/8) on [5.565063e-3, 5] (the
 *      file's printed bounds), order 14, 50 digits, must match r_inv_1f_*
 *      (power) and r_1f_* (inverse) from the HISQ file
 *      in.rational_b6330ms0746mls240 (constant, first/last residue and pole) to
 *      relative 1e-6. A first run with lambda_low = 0.0746^2 differed by 1e-6
 *      to 2e-5, the size of that 1.7e-5 relative bound difference.
 *   2. MDWF-like interval [1e-4, 200]: for x^(+-1/4) and x^(+-1/2) (order 14,
 *      100 digits) the sampled maximum relative error over 2000 log-spaced
 *      points is <= 1.01 * the AlgRemez error + 1e-15, the AlgRemez error is
 *      below 1e-6, all poles are positive, residues of negative powers are
 *      positive, and residues of positive powers are negative.
 *   3. Consistency of heatbath and action functions: r_{+1/4}(x)^2 * r_{-1/2}(x)
 *      = 1 on the grid within 2 * err(1/4) + err(1/2) + 1e-14.
 *   4. Diagnostic only: relative error at lambda_low/2 and 2*lambda_high, to
 *      show how quickly the approximation degrades outside its interval.
 */

#include "../experimental/mdwf/MDWFRemez.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

bool checkClose(const char *label, double value, double reference, double tolerance) {
    const double rel = std::abs(value - reference) / std::max(std::abs(reference), 1e-300);
    const bool ok = rel <= tolerance;
    std::printf("MDWF Remez test anchor %-22s value = % .16e reference = % .16e relDiff = %.3e %s\n",
                label, value, reference, rel, ok ? "ok" : "FAIL");
    return ok;
}

double maxSampledRelError(const MDWFRemezPartialFractions &pf, double exponent, double low, double high,
                          int samples) {
    double maxErr = 0.0;
    for (int i = 0; i < samples; i++) {
        const double x = low * std::pow(high / low, static_cast<double>(i) / static_cast<double>(samples - 1));
        const double f = std::pow(x, exponent);
        maxErr = std::max(maxErr, std::abs(mdwfRemezEvaluate(pf, x) - f) / f);
    }
    return maxErr;
}

bool checkSigns(const MDWFRemezPartialFractions &pf, bool negativePower, const char *label) {
    bool polesPositive = true;
    bool residueSigns = true;
    for (size_t i = 0; i < pf.poles.size(); i++) {
        polesPositive = polesPositive && pf.poles[i] > 0.0;
        residueSigns = residueSigns && (negativePower ? pf.residues[i] > 0.0 : pf.residues[i] < 0.0);
    }
    const bool constantOk = negativePower ? pf.constant >= 0.0 : pf.constant > 0.0;
    std::printf("MDWF Remez test signs %-14s poles positive = %d, residue signs = %d, constant sign = %d\n",
                label, polesPositive, residueSigns, constantOk);
    return polesPositive && residueSigns && constantOk;
}

}  // namespace

int main() {
    bool allPassed = true;

    // --- Part 1: reproduce ratApprox (HISQ strange-quark file). ---
    // The file's header gives the bounds as [5.565063e-03, 5.000000e+00]; its lower bound is slightly
    // below m_s^2 = 0.0746^2 = 5.56516e-3. The printed 7 significant digits leave <~1e-7 relative
    // uncertainty in the coefficients, hence the tolerance.
    const MDWFRemezApproximation anchor = mdwfRemezPower(3, 8, 5.565063e-3, 5.0, 14, 50);
    std::cout << mdwfRemezDescribe(anchor);
    const double tol = 1e-6;
    bool anchorPassed = true;
    anchorPassed &= checkClose("r_inv_1f_const", anchor.power.constant, 9.5060139692630425e+00, tol);
    anchorPassed &= checkClose("r_inv_1f_num[0]", anchor.power.residues[0], -2.1485790149854497e-05, tol);
    anchorPassed &= checkClose("r_inv_1f_num[13]", anchor.power.residues[13], -9.0861919587392981e+02, tol);
    anchorPassed &= checkClose("r_inv_1f_den[0]", anchor.power.poles[0], 5.5727652690748205e-04, tol);
    anchorPassed &= checkClose("r_inv_1f_den[13]", anchor.power.poles[13], 1.3889809157959124e+02, tol);
    anchorPassed &= checkClose("r_1f_const", anchor.inverse.constant, 1.0519656327388349e-01, tol);
    anchorPassed &= checkClose("r_1f_num[0]", anchor.inverse.residues[0], 5.0207017669607560e-03, tol);
    anchorPassed &= checkClose("r_1f_num[13]", anchor.inverse.residues[13], 7.3753885239137427e+00, tol);
    anchorPassed &= checkClose("r_1f_den[0]", anchor.inverse.poles[0], 2.0032900800552442e-04, tol);
    anchorPassed &= checkClose("r_1f_den[13]", anchor.inverse.poles[13], 4.9930897061845030e+01, tol);
    std::printf("MDWF Remez test anchor: AlgRemez error = %.6e (file: 3.795363e-13), passed = %d\n",
                anchor.max_relative_error, anchorPassed);
    allPassed &= anchorPassed;

    // --- Part 2: MDWF-like interval. ---
    const double low = 1e-4;
    const double high = 200.0;
    const int order = 14;
    const int digits = 100;
    const int samples = 2000;
    const MDWFRemezApproximation quarter = mdwfRemezPower(1, 4, low, high, order, digits);
    const MDWFRemezApproximation half = mdwfRemezPower(1, 2, low, high, order, digits);
    std::cout << mdwfRemezDescribe(quarter) << mdwfRemezDescribe(half);

    struct Case {
        const char *label;
        const MDWFRemezPartialFractions *pf;
        double exponent;
        double reported;
    };
    const Case cases[4] = {
        {"x^(+1/4)", &quarter.power, 0.25, quarter.max_relative_error},
        {"x^(-1/4)", &quarter.inverse, -0.25, quarter.max_relative_error},
        {"x^(+1/2)", &half.power, 0.5, half.max_relative_error},
        {"x^(-1/2)", &half.inverse, -0.5, half.max_relative_error},
    };
    bool mdwfPassed = true;
    for (const Case &c : cases) {
        const double sampled = maxSampledRelError(*c.pf, c.exponent, low, high, samples);
        const bool errorOk = c.reported < 1e-6 && sampled <= 1.01 * c.reported + 1e-15;
        const bool signsOk = checkSigns(*c.pf, c.exponent < 0.0, c.label);
        const double below = std::abs(mdwfRemezEvaluate(*c.pf, 0.5 * low) - std::pow(0.5 * low, c.exponent))
                             / std::pow(0.5 * low, c.exponent);
        const double above = std::abs(mdwfRemezEvaluate(*c.pf, 2.0 * high) - std::pow(2.0 * high, c.exponent))
                             / std::pow(2.0 * high, c.exponent);
        std::printf("MDWF Remez test %-8s on [%g, %g]: AlgRemez error = %.3e, sampled max error = %.3e, "
                    "passed = %d; outside: relErr(low/2) = %.3e, relErr(2*high) = %.3e\n",
                    c.label, low, high, c.reported, sampled, errorOk && signsOk, below, above);
        mdwfPassed &= errorOk && signsOk;
    }
    allPassed &= mdwfPassed;

    // --- Part 3: heatbath/action consistency r_{+1/4}^2 r_{-1/2} = 1. ---
    double maxDeviation = 0.0;
    for (int i = 0; i < samples; i++) {
        const double x = low * std::pow(high / low, static_cast<double>(i) / static_cast<double>(samples - 1));
        const double h = mdwfRemezEvaluate(quarter.power, x);
        maxDeviation = std::max(maxDeviation, std::abs(h * h * mdwfRemezEvaluate(half.inverse, x) - 1.0));
    }
    const double consistencyTolerance = 2.0 * quarter.max_relative_error + half.max_relative_error + 1e-14;
    const bool consistencyPassed = maxDeviation <= consistencyTolerance;
    std::printf("MDWF Remez test consistency: max |r_{1/4}(x)^2 r_{-1/2}(x) - 1| = %.3e (tolerance %.3e), "
                "passed = %d\n", maxDeviation, consistencyTolerance, consistencyPassed);
    allPassed &= consistencyPassed;

    std::printf("MDWF Remez test %s\n", allPassed ? "passed" : "FAILED");
    return allPassed ? 0 : 1;
}
