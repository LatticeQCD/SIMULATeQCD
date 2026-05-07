/*
 * MDWF shifted normal-equation CG solve smoke test.
 *
 * This test validates the isolated shifted operator
 *
 *     (M^\dagger M + sigma) x
 *
 * using the coupled-5D CG scaffold.  It checks that sigma = 0 reproduces the
 * unshifted normal operator and that a positive shift converges for both
 * c_sw = 0 and nonzero c_sw.  It does not touch RHMC/HMC, force code, HISQ, or
 * smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFShiftedNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFShiftedNormalPattern {
    floatT offset;

    explicit FillMDWFShiftedNormalPattern(floatT offset_in)
        : offset(offset_in) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                offset + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                0.0);
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFShiftedNormalSolveCase(CommunicationBase &commBase,
                                   double csw,
                                   double sigma,
                                   const std::string &caseName) {
    const size_t HaloDepth = 2;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperator, double>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ShiftedOperator>;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, caseName + "_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260508);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, caseName + "_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, caseName + "_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shiftedSolution(commBase, caseName + "_shifted_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> residual(commBase, caseName + "_residual");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalSource(commBase, caseName + "_normal_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shiftedZeroSource(commBase, caseName + "_shifted_zero_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> zeroShiftDifference(commBase, caseName + "_zero_shift_difference");
    MDWFSpinor<double, true, All, HaloDepth, Ls> rayleighProbe(commBase, caseName + "_rayleigh_probe");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shiftedRayleighProbe(commBase, caseName + "_shifted_rayleigh_probe");

    source.template iterateOverBulk<>(FillMDWFShiftedNormalPattern<double, All, HaloDepth, Ls>(0.25));
    rayleighProbe.template iterateOverBulk<>(FillMDWFShiftedNormalPattern<double, All, HaloDepth, Ls>(1.25));
    source.updateAll();
    rayleighProbe.updateAll();

    MDWFFifthDimCoefficients<double> coeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, coeff, 4.0, csw, caseName + "_forward");
    AdjointOperator adjoint(gauge, coeff, 4.0, csw, caseName + "_adjoint");
    NormalOperator normal(commBase, forward, adjoint, caseName + "_normal");
    ShiftedOperator shiftedZero(normal, 0.0, caseName + "_shifted_zero");
    ShiftedOperator shifted(normal, sigma, caseName + "_shifted");
    ShiftedAdapter shiftedZeroAdapter(shiftedZero);
    ShiftedAdapter shiftedAdapter(shifted);

    normal.apply(normalSource, source, true);
    shiftedZero.apply(shiftedZeroSource, source, true);
    zeroShiftDifference = shiftedZeroSource;
    zeroShiftDifference -= normalSource;
    const double zeroShiftDiffNorm2 = shiftedZeroAdapter.norm2(zeroShiftDifference);
    const double normalSourceNorm2 = shiftedZeroAdapter.norm2(normalSource);
    const double zeroShiftRelDiff = std::sqrt(zeroShiftDiffNorm2 / std::max(normalSourceNorm2, 1.0));

    shifted.apply(shiftedRayleighProbe, rayleighProbe, true);
    const COMPLEX(double) rayleigh = shiftedAdapter.dotProduct5D(rayleighProbe, shiftedRayleighProbe);
    const double rayleighReal = real<double>(rayleigh);
    const double rayleighImag = imag<double>(rayleigh);
    const double rayleighImagRel = std::abs(rayleighImag) / std::max(std::abs(rayleighReal), 1.0);

    MDWFCoupledCG<double, ShiftedAdapter> cg;
    MDWFCoupledCGResult<double> result = cg.invert(shiftedAdapter, solution, source, 512, 1e-8, true);

    shifted.apply(shiftedSolution, solution, true);
    residual = source;
    residual -= shiftedSolution;

    const double sourceNorm2 = shiftedAdapter.norm2(source);
    const double residualNorm2 = shiftedAdapter.norm2(residual);
    const double relativeResidual = std::sqrt(residualNorm2 / std::max(sourceNorm2, 1.0));

    if (forward.csw() != csw
        || adjoint.csw() != csw
        || shifted.sigma() != sigma
        || zeroShiftRelDiff > 1e-12
        || !std::isfinite(rayleighReal)
        || !std::isfinite(rayleighImag)
        || rayleighReal <= 0.0
        || rayleighImagRel > 1e-10
        || !result.converged
        || result.residue > 1e-8
        || relativeResidual > 1e-7) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF shifted normal-equation solve failed for ", caseName,
            ": c_sw = ", csw,
            ", sigma = ", sigma,
            ", zeroShiftRelDiff = ", zeroShiftRelDiff,
            ", rayleighReal = ", rayleighReal,
            ", rayleighImag = ", rayleighImag,
            ", rayleighImagRel = ", rayleighImagRel,
            ", converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", relativeResidual = ", relativeResidual));
    }

    rootLogger.info("MDWF shifted normal-equation solve passed for ", caseName,
                    " with Ls = ", Ls,
                    ", c_sw = ", csw,
                    ", sigma = ", sigma,
                    ", iterations = ", result.iterations,
                    ", residue = ", result.residue,
                    ", relativeResidual = ", relativeResidual,
                    ", zeroShiftRelDiff = ", zeroShiftRelDiff,
                    ", rayleighImagRel = ", rayleighImagRel);
}

template<size_t Ls>
void runMDWFShiftedNormalSolveSmokeTest(CommunicationBase &commBase) {
    runMDWFShiftedNormalSolveCase<Ls>(commBase, 0.0, 0.1, "MDWF_shifted_normal_csw0");
    runMDWFShiftedNormalSolveCase<Ls>(commBase, 0.5, 0.1, "MDWF_shifted_normal_csw05");
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

        runMDWFShiftedNormalSolveSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
