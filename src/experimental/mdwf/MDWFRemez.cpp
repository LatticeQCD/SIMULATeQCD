/*
 * Host-only implementation of MDWFRemez.h on top of AlgRemez
 * (src/tools/rational_approx, used unchanged). Compile as plain C++ with the
 * GMP and MPFR include paths and link -lmpfr -lgmp.
 */

#include "MDWFRemez.h"
#include "../../tools/rational_approx/alg_remez.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

MDWFRemezApproximation mdwfRemezPower(int pnum, int pden, double lambda_low, double lambda_high,
                                      int order, int digits) {
    if (pden <= 0 || pnum == 0) {
        throw std::runtime_error("mdwfRemezPower requires pden > 0 and pnum != 0");
    }
    if (!(lambda_low > 0.0) || !(lambda_high > lambda_low)) {
        throw std::runtime_error("mdwfRemezPower requires 0 < lambda_low < lambda_high");
    }
    if (order <= 0 || digits <= 0) {
        throw std::runtime_error("mdwfRemezPower requires positive order and digits");
    }

    MDWFRemezApproximation result{};
    result.pnum = pnum;
    result.pden = pden;
    result.lambda_low = lambda_low;
    result.lambda_high = lambda_high;
    result.order = order;
    result.digits = digits;

    // One power, no masses: the other three "flavors" have exponent 0 and are skipped by AlgRemez::func.
    AlgRemez remez(lambda_low, lambda_high, digits);
    result.max_relative_error = remez.generateApprox(order, order,
                                                     static_cast<double>(pnum), static_cast<double>(pden), 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 1.0, 0.0);

    std::vector<double> residues(order);
    std::vector<double> poles(order);
    double norm = 0.0;

    remez.getPFE(residues.data(), poles.data(), &norm);
    result.power = {norm, residues, poles};

    remez.getIPFE(residues.data(), poles.data(), &norm);
    result.inverse = {norm, residues, poles};

    return result;
}

double mdwfRemezEvaluate(const MDWFRemezPartialFractions &pf, double x) {
    double sum = pf.constant;
    for (size_t i = 0; i < pf.poles.size(); i++) {
        sum += pf.residues[i] / (x + pf.poles[i]);
    }
    return sum;
}

std::string mdwfRemezDescribe(const MDWFRemezApproximation &approx) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "# Remez x^(+-" << approx.pnum << "/" << approx.pden << ") on [" << approx.lambda_low << ", "
        << approx.lambda_high << "], order " << approx.order << ", " << approx.digits
        << " digits, max relative error " << approx.max_relative_error << "\n";
    const MDWFRemezPartialFractions *parts[2] = {&approx.power, &approx.inverse};
    const char *labels[2] = {"power", "inverse"};
    for (int p = 0; p < 2; p++) {
        out << "#   " << labels[p] << " constant = " << parts[p]->constant << "\n";
        for (size_t i = 0; i < parts[p]->poles.size(); i++) {
            out << "#   " << labels[p] << " residue[" << i << "] = " << parts[p]->residues[i]
                << "  pole[" << i << "] = " << parts[p]->poles[i] << "\n";
        }
    }
    return out.str();
}
