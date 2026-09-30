/*
 * MDWF Wilson kernel normalization test.
 *
 * Pins the mass argument of the per-slice Wilson kernel to the standard
 * domain-wall kernel D_W(-M5). On a unit gauge field, with
 * mass = mdwfShamirKernelMass(M5), plane waves are eigenvectors of D_W:
 *
 *   D_W(p) = mass - sum_mu cos(p_mu) + i sum_mu gamma_mu sin(p_mu),
 *
 * so for M5 in the domain-wall window the kernel must give
 *
 *   p = 0                        (constant field):         D_W psi = -M5 psi
 *   p = (pi, 0, 0, 0)            (sign (-1)^x):             D_W psi = (2 - M5) psi
 *   p = (0, 0, 0, pi)            (sign (-1)^t):             D_W psi = (2 - M5) psi
 *
 * for the plain path (applyMDWFWilsonSlice, gamma5DiracWilson) and the clover
 * path (applyMDWFCloverWilsonSlice, preCalcFmunu + DiracWilsonEvenEven2/
 * EvenOdd2) at c_sw = 0 and c_sw = 0.5 (the clover term vanishes for U = 1).
 * The p = (0,0,0,pi) wave also shows that the kernel has no built-in
 * antiperiodic time boundary. The spinor varies over spin, color, and the
 * Ls stacks. Relative tolerance 1e-13, for M5 = 1.8 and M5 = 1.0.
 *
 * The first MDWF mapping used mass = 2 - M5/2, which gives -2.9 instead of
 * -1.8 at p = 0 for M5 = 1.8 (PHYSICAL_OPERATOR_MAPPING.md Section 1.3).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Plane wave psi(x, s) = v(s) * sign(x), sign = (-1)^(x_mu) for mu = direction, 1 for direction < 0.
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFKernelNormalizationWave {
    int _direction;

    explicit MDWFKernelNormalizationWave(int direction) : _direction(direction) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        for (int i = 0; i < 12; i++) {
            out.data[i] = COMPLEX(floatT)(static_cast<floatT>(0.1) * static_cast<floatT>(i + 1)
                                              - static_cast<floatT>(0.03) * static_cast<floatT>(site.stack),
                                          static_cast<floatT>(0.05) * static_cast<floatT>(site.stack + 1)
                                              - static_cast<floatT>(0.02) * static_cast<floatT>(i));
        }
        if (_direction >= 0 && (site.coord[_direction] % 2) == 1) {
            out = static_cast<floatT>(-1.0) * out;
        }
        return out;
    }
};

template<class Spinor>
double mdwfKernelNorm2(Spinor &spinor) {
    double sum = 0.0;
    for (const COMPLEX(double) &stackDot : spinor.dotProductStacked(spinor)) {
        sum += real<double>(stackDot);
    }
    return sum;
}

template<size_t Ls>
bool runMDWFWilsonKernelNormalizationCase(CommunicationBase &commBase, double M5) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using CloverField = Spinorfield<double, true, All, HaloDepth, 18, 1>;

    Gauge gauge(commBase, "MDWF_kernel_norm_gauge");
    gauge.one();
    gauge.updateAll();

    Spinor in(commBase, "MDWF_kernel_norm_in");
    Spinor out(commBase, "MDWF_kernel_norm_out");
    Spinor tmp(commBase, "MDWF_kernel_norm_tmp");
    CloverField fUpper(commBase, "MDWF_kernel_norm_f_upper");
    CloverField fLower(commBase, "MDWF_kernel_norm_f_lower");
    CloverField fInvUpper(commBase, "MDWF_kernel_norm_f_inv_upper");
    CloverField fInvLower(commBase, "MDWF_kernel_norm_f_inv_lower");

    const double mass = mdwfShamirKernelMass(M5);
    const int directions[3] = {-1, 0, 3};
    const char *labels[3] = {"p = 0", "p = (pi,0,0,0)", "p = (0,0,0,pi)"};
    const double expected[3] = {-M5, 2.0 - M5, 2.0 - M5};
    bool passed = true;

    for (int w = 0; w < 3; w++) {
        in.template iterateOverBulk<BLOCKSIZE>(MDWFKernelNormalizationWave<double, HaloDepth, Ls>(directions[w]));
        in.updateAll();
        const double inNorm2 = mdwfKernelNorm2(in);

        for (int path = 0; path < 3; path++) {
            const double csw = (path == 2) ? 0.5 : 0.0;
            if (path == 0) {
                applyMDWFWilsonSlice<double, HaloDepth, HaloDepth, Ls>(out, gauge, tmp, in, mass, 0.0, true);
            } else {
                applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, Ls>(
                    out, gauge, tmp, fUpper, fLower, fInvUpper, fInvLower, in, mass, csw, true);
            }
            // out - expected * in
            tmp = out;
            tmp.template axpyThisB<64>(-expected[w], in);
            const double relDiff = std::sqrt(mdwfKernelNorm2(tmp) / inNorm2);
            // The eigenvalue actually realized (Rayleigh quotient).
            COMPLEX(double) rq = 0.0;
            const std::vector<COMPLEX(double)> dots = in.dotProductStacked(out);
            for (const COMPLEX(double) &d : dots) {
                rq += d;
            }
            const double eigenvalue = real<double>(rq) / inNorm2;
            const bool ok = relDiff <= 1e-13;
            passed = passed && ok;
            rootLogger.info("MDWF Wilson kernel normalization: M5 = ", M5, ", mass = ", mass, ", ", labels[w],
                            ", ", path == 0 ? "plain" : "clover", " path (c_sw = ", csw, "): eigenvalue = ",
                            eigenvalue, ", expected ", expected[w], ", |D_W psi - expected psi| / |psi| = ", relDiff,
                            ", passed = ", ok);
        }
    }
    return passed;
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        const bool passed18 = runMDWFWilsonKernelNormalizationCase<8>(commBase, 1.8);
        const bool passed10 = runMDWFWilsonKernelNormalizationCase<8>(commBase, 1.0);
        if (!passed18 || !passed10) {
            throw std::runtime_error(stdLogger.fatal("MDWF Wilson kernel normalization test failed: M5 = 1.8 passed = ",
                                                     passed18, ", M5 = 1.0 passed = ", passed10));
        }
        rootLogger.info("MDWF Wilson kernel normalization test passed: D_W(-M5) has the standard normalization");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
