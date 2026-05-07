/*
 * MDWF rational normal-operator smoke test.
 *
 * This applies MDWFRationalOperator to the explicit MDWF normal operator and
 * compares the result against the already validated repeated single-shift path:
 *
 *     c0 b + sum_i numerator_i (M^\dagger M + shift_i)^(-1) b
 *
 * It is still pre-RHMC: no pseudofermions, no RHMC/HMC integration, no force
 * code, no HISQ, and no smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalOperator.h"
#include "../experimental/mdwf/MDWFShiftedNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFRationalNormalMdwfPattern {
    floatT offset;

    explicit FillMDWFRationalNormalMdwfPattern(floatT offset_in)
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
void runMDWFRationalNormalMdwfSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;

    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperator, double>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ShiftedOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFRationalCoefficients<double> coefficients{0.25, {0.5, -0.125, 0.75}, {0.0, 0.1, 0.3}};

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_rational_normal_mdwf_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260510);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    Spinor source(commBase, "MDWF_rational_normal_mdwf_source");
    Spinor rationalOutput(commBase, "MDWF_rational_normal_mdwf_output");
    Spinor reference(commBase, "MDWF_rational_normal_mdwf_reference");
    Spinor difference(commBase, "MDWF_rational_normal_mdwf_difference");
    Spinor shiftedSolution(commBase, "MDWF_rational_normal_mdwf_shifted_solution");
    Spinor residual(commBase, "MDWF_rational_normal_mdwf_residual");
    std::vector<std::unique_ptr<Spinor>> singleShiftSolutions;

    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        singleShiftSolutions.emplace_back(new Spinor(
            commBase, "MDWF_rational_normal_mdwf_single_shift_solution_" + std::to_string(term)));
    }

    source.template iterateOverBulk<>(
        FillMDWFRationalNormalMdwfPattern<double, All, HaloDepth, Ls>(0.25));
    source.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_rational_normal_mdwf_forward");
    AdjointOperator adjoint(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_rational_normal_mdwf_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_rational_normal_mdwf_normal");
    NormalAdapter normalAdapter(normal);

    MDWFRationalOperator<double, NormalAdapter> rationalOperator(
        coefficients, 512, 1e-8, "MDWF_rational_normal_mdwf");
    MDWFCoupledMultiShiftCGResults<double> rationalResult
        = rationalOperator.apply(normalAdapter, rationalOutput, source, true);

    reference = coefficients.constant * source;
    double maxSingleShiftResidue = 0.0;
    double maxRelativeResidual = 0.0;

    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        ShiftedOperator shifted(normal, coefficients.shift[term],
                                "MDWF_rational_normal_mdwf_shifted_" + std::to_string(term));
        ShiftedAdapter shiftedAdapter(shifted);
        MDWFCoupledCG<double, ShiftedAdapter> singleShiftCg;
        MDWFCoupledCGResult<double> singleShiftResult
            = singleShiftCg.invert(shiftedAdapter, *singleShiftSolutions[term], source, 512, 1e-8, true);

        shifted.apply(shiftedSolution, *singleShiftSolutions[term], true);
        residual = source;
        residual -= shiftedSolution;
        const double sourceNorm2 = normalAdapter.norm2(source);
        const double residualNorm2 = normalAdapter.norm2(residual);
        const double relativeResidual = std::sqrt(residualNorm2 / std::max(sourceNorm2, 1.0));

        reference.template axpyThisB<64>(coefficients.numerator[term], *singleShiftSolutions[term]);

        maxSingleShiftResidue = std::max(maxSingleShiftResidue, singleShiftResult.residue);
        maxRelativeResidual = std::max(maxRelativeResidual, relativeResidual);

        if (!singleShiftResult.converged
            || singleShiftResult.residue > 1e-8
            || relativeResidual > 1e-7) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational normal-equation reference solve failed for term ", term,
                ": shift = ", coefficients.shift[term],
                ", converged = ", singleShiftResult.converged,
                ", iterations = ", singleShiftResult.iterations,
                ", residue = ", singleShiftResult.residue,
                ", relativeResidual = ", relativeResidual));
        }
    }
    reference.updateAll();

    double maxRationalResidue = 0.0;
    for (size_t term = 0; term < rationalResult.shifts.size(); term++) {
        maxRationalResidue = std::max(maxRationalResidue, rationalResult.shifts[term].residue);
    }

    difference = rationalOutput;
    difference -= reference;

    const double referenceNorm2 = normalAdapter.norm2(reference);
    const double differenceNorm2 = normalAdapter.norm2(difference);
    const double relativeDifference = std::sqrt(differenceNorm2 / std::max(referenceNorm2, 1.0));

    if (forward.csw() != nonzeroCsw
        || adjoint.csw() != nonzeroCsw
        || !rationalResult.converged()
        || maxRationalResidue > 1e-8
        || maxSingleShiftResidue > 1e-8
        || maxRelativeResidual > 1e-7
        || relativeDifference > 1e-10) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational normal-equation comparison failed: c_sw = ", nonzeroCsw,
            ", rationalConverged = ", rationalResult.converged(),
            ", maxRationalResidue = ", maxRationalResidue,
            ", maxSingleShiftResidue = ", maxSingleShiftResidue,
            ", maxRelativeResidual = ", maxRelativeResidual,
            ", relativeDifference = ", relativeDifference));
    }

    rootLogger.info("MDWF rational normal-equation comparison passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", terms = ", coefficients.shift.size(),
                    ", maxRationalResidue = ", maxRationalResidue,
                    ", maxSingleShiftResidue = ", maxSingleShiftResidue,
                    ", maxRelativeResidual = ", maxRelativeResidual,
                    ", relativeDifference = ", relativeDifference);
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

        runMDWFRationalNormalMdwfSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
