/*
 * Host-only implementation of MDWFRemez.h on top of AlgRemez
 * (src/tools/rational_approx, used unchanged). Compile as plain C++ with the
 * GMP and MPFR include paths and link -lmpfr -lgmp.
 */

#include "MDWFRemez.h"
#include "../../tools/rational_approx/alg_remez.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <unistd.h>

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

namespace {

// Optional on-disk cache: directory from the environment variable MDWF_REMEZ_CACHE (unset: memory only).
// One text file per (pnum, pden, interval, order, digits); doubles written with 17 significant digits.
std::string mdwfRemezCacheFile(int pnum, int pden, double low, double high, int order, int digits) {
    const char *dir = std::getenv("MDWF_REMEZ_CACHE");
    if (dir == nullptr || *dir == '\0') {
        return "";
    }
    std::ostringstream name;
    name << std::setprecision(17) << dir << "/remez_" << pnum << "_" << pden << "_" << low << "_" << high
         << "_o" << order << "_d" << digits << ".txt";
    return name.str();
}

bool mdwfRemezRead(const std::string &file, MDWFRemezApproximation &a) {
    std::ifstream in(file);
    if (file.empty() || !in) {
        return false;
    }
    in >> a.pnum >> a.pden >> a.lambda_low >> a.lambda_high >> a.order >> a.digits >> a.max_relative_error;
    MDWFRemezPartialFractions *parts[2] = {&a.power, &a.inverse};
    for (MDWFRemezPartialFractions *pf : parts) {
        pf->residues.assign(a.order, 0.0);
        pf->poles.assign(a.order, 0.0);
        in >> pf->constant;
        for (int i = 0; i < a.order; i++) {
            in >> pf->residues[i] >> pf->poles[i];
        }
    }
    return static_cast<bool>(in);
}

void mdwfRemezWrite(const std::string &file, const MDWFRemezApproximation &a) {
    if (file.empty()) {
        return;
    }
    const std::string tmp = file + ".tmp" + std::to_string(::getpid());
    {
        std::ofstream out(tmp);
        if (!out) {
            return;
        }
        out << std::setprecision(17);
        out << a.pnum << " " << a.pden << " " << a.lambda_low << " " << a.lambda_high << " " << a.order << " "
            << a.digits << " " << a.max_relative_error << "\n";
        const MDWFRemezPartialFractions *parts[2] = {&a.power, &a.inverse};
        for (const MDWFRemezPartialFractions *pf : parts) {
            out << pf->constant << "\n";
            for (int i = 0; i < a.order; i++) {
                out << pf->residues[i] << " " << pf->poles[i] << "\n";
            }
        }
    }
    std::rename(tmp.c_str(), file.c_str());
}

} // namespace

MDWFRemezApproximation mdwfRemezPowerForError(int pnum, int pden, double lambda_low, double lambda_high,
                                              double targetError, int maxOrder, int digits, int minOrder) {
    if (!(targetError > 0.0) || minOrder <= 0 || maxOrder < minOrder) {
        throw std::runtime_error("mdwfRemezPowerForError requires targetError > 0 and 0 < minOrder <= maxOrder");
    }
    using Key = std::tuple<int, int, double, double, int, int>;
    static std::map<Key, MDWFRemezApproximation> cache;

    auto approx = [&](int order) -> const MDWFRemezApproximation & {
        const Key key{pnum, pden, lambda_low, lambda_high, order, digits};
        auto found = cache.find(key);
        if (found == cache.end()) {
            const std::string file = mdwfRemezCacheFile(pnum, pden, lambda_low, lambda_high, order, digits);
            MDWFRemezApproximation a{};
            if (!mdwfRemezRead(file, a)) {
                a = mdwfRemezPower(pnum, pden, lambda_low, lambda_high, order, digits);
                mdwfRemezWrite(file, a);
            }
            found = cache.emplace(key, a).first;
        }
        return found->second;
    };
    auto passes = [&](int order) {
        const double e = approx(order).max_relative_error;
        return std::isfinite(e) && e <= targetError;
    };

    // The minimax error falls geometrically with the order. Measure the rate on two cheap low orders and start
    // one below the predicted order instead of computing every order (high orders dominate the cost).
    int order = minOrder;
    const int probe = std::min(minOrder + 4, maxOrder);
    if (probe > minOrder + 1 && !passes(probe)) {
        const double e1 = approx(probe - 1).max_relative_error;
        const double e2 = approx(probe).max_relative_error;
        if (std::isfinite(e1) && std::isfinite(e2) && e2 > 0.0 && e1 > e2) {
            const double steps = std::log(targetError / e2) / std::log(e2 / e1);
            order = std::max(probe + 1, probe + static_cast<int>(std::ceil(steps)) - 1);
            order = std::min(order, maxOrder);
        } else {
            order = probe + 1;
        }
    }
    // Lowest passing order: step down while the order below still passes, otherwise step up.
    if (order <= maxOrder && passes(order)) {
        while (order > minOrder && passes(order - 1)) {
            order--;
        }
        return approx(order);
    }
    for (order++; order <= maxOrder; order++) {
        if (passes(order)) {
            return approx(order);
        }
    }
    std::ostringstream message;
    message << "mdwfRemezPowerForError: x^(" << pnum << "/" << pden << ") on [" << lambda_low << ", " << lambda_high
            << "] reaches only max relative error " << approx(maxOrder).max_relative_error << " at order "
            << maxOrder << ", target " << targetError;
    throw std::runtime_error(message.str());
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
