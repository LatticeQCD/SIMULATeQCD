/*
 * MDWF coupled-5D multishift normal-operator smoke test.
 *
 * This applies MDWFCoupledMultiShiftCG to the explicit MDWF normal operator
 * with several shifts and compares every shifted solution against the already
 * validated single-shift path using MDWFShiftedNormalOperator + MDWFCoupledCG.
 * It is still pre-RHMC: no HMC/RHMC integration, no force code, no HISQ, and no
 * smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFCoupledMultiShiftCG.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFShiftedNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMultiShiftNormalMdwfPattern {
    floatT offset;

    explicit FillMDWFMultiShiftNormalMdwfPattern(floatT offset_in)
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

template<class Spinor>
std::vector<Spinor *> makeMDWFMultiShiftNormalSolutionPointers(std::vector<std::unique_ptr<Spinor>> &solutions) {
    std::vector<Spinor *> pointers;
    pointers.reserve(solutions.size());
    for (auto &solution : solutions) {
        pointers.push_back(solution.get());
    }
    return pointers;
}

template<size_t Ls>
void runMDWFMultiShiftNormalMdwfSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;
    const std::vector<double> sigma = {0.0, 0.1, 0.3};

    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperator, double>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ShiftedOperator>;
    using Spinor = typename NormalOperator::Spinor;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_multishift_normal_mdwf_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260509);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_multishift_normal_mdwf_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> difference(commBase, "MDWF_multishift_normal_mdwf_difference");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shiftedMultishiftSolution(
        commBase, "MDWF_multishift_normal_mdwf_shifted_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> residual(commBase, "MDWF_multishift_normal_mdwf_residual");
    std::vector<std::unique_ptr<Spinor>> multishiftSolutions;
    std::vector<std::unique_ptr<Spinor>> singleShiftSolutions;

    for (size_t shift = 0; shift < sigma.size(); shift++) {
        multishiftSolutions.emplace_back(new Spinor(
            commBase, "MDWF_multishift_normal_mdwf_multishift_solution_" + std::to_string(shift)));
        singleShiftSolutions.emplace_back(new Spinor(
            commBase, "MDWF_multishift_normal_mdwf_single_shift_solution_" + std::to_string(shift)));
    }

    source.template iterateOverBulk<>(
        FillMDWFMultiShiftNormalMdwfPattern<double, All, HaloDepth, Ls>(0.25));
    source.updateAll();

    MDWFFifthDimCoefficients<double> coeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, coeff, 4.0, nonzeroCsw, "MDWF_multishift_normal_mdwf_forward");
    AdjointOperator adjoint(gauge, coeff, 4.0, nonzeroCsw, "MDWF_multishift_normal_mdwf_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_multishift_normal_mdwf_normal");
    NormalAdapter normalAdapter(normal);

    MDWFCoupledMultiShiftCG<double, NormalAdapter> multishiftCg;
    std::vector<Spinor *> multishiftPointers = makeMDWFMultiShiftNormalSolutionPointers(multishiftSolutions);
    MDWFCoupledMultiShiftCGResults<double> multishiftResult
        = multishiftCg.invert(normalAdapter, multishiftPointers, source, sigma, 512, 1e-8, true);

    double maxRelativeDifference = 0.0;
    double maxRelativeResidual = 0.0;

    for (size_t shift = 0; shift < sigma.size(); shift++) {
        ShiftedOperator shifted(normal, sigma[shift],
                                "MDWF_multishift_normal_mdwf_shifted_" + std::to_string(shift));
        ShiftedAdapter shiftedAdapter(shifted);
        MDWFCoupledCG<double, ShiftedAdapter> singleShiftCg;
        MDWFCoupledCGResult<double> singleShiftResult
            = singleShiftCg.invert(shiftedAdapter, *singleShiftSolutions[shift], source, 512, 1e-8, true);

        difference = *multishiftSolutions[shift];
        difference -= *singleShiftSolutions[shift];

        const double singleShiftNorm2 = normalAdapter.norm2(*singleShiftSolutions[shift]);
        const double differenceNorm2 = normalAdapter.norm2(difference);
        const double relativeDifference = std::sqrt(differenceNorm2 / std::max(singleShiftNorm2, 1.0));

        shifted.apply(shiftedMultishiftSolution, *multishiftSolutions[shift], true);
        residual = source;
        residual -= shiftedMultishiftSolution;
        const double sourceNorm2 = normalAdapter.norm2(source);
        const double residualNorm2 = normalAdapter.norm2(residual);
        const double relativeResidual = std::sqrt(residualNorm2 / std::max(sourceNorm2, 1.0));

        if (relativeDifference > maxRelativeDifference) {
            maxRelativeDifference = relativeDifference;
        }
        if (relativeResidual > maxRelativeResidual) {
            maxRelativeResidual = relativeResidual;
        }

        if (!multishiftResult.shifts[shift].converged
            || !singleShiftResult.converged
            || multishiftResult.shifts[shift].residue > 1e-8
            || singleShiftResult.residue > 1e-8
            || relativeDifference > 1e-10
            || relativeResidual > 1e-7) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF multishift normal-equation comparison failed for shift ", shift,
                ": sigma = ", sigma[shift],
                ", multishift_converged = ", multishiftResult.shifts[shift].converged,
                ", single_shift_converged = ", singleShiftResult.converged,
                ", multishift_iterations = ", multishiftResult.shifts[shift].iterations,
                ", single_shift_iterations = ", singleShiftResult.iterations,
                ", multishift_residue = ", multishiftResult.shifts[shift].residue,
                ", single_shift_residue = ", singleShiftResult.residue,
                ", relativeDifference = ", relativeDifference,
                ", relativeResidual = ", relativeResidual));
        }
    }

    if (forward.csw() != nonzeroCsw || adjoint.csw() != nonzeroCsw || !multishiftResult.converged()) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF multishift normal-equation test failed final guard: forward_csw = ", forward.csw(),
            ", adjoint_csw = ", adjoint.csw(),
            ", converged = ", multishiftResult.converged()));
    }

    rootLogger.info("MDWF multishift normal-equation comparison passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", shifts = ", sigma.size(),
                    ", maxRelativeDifference = ", maxRelativeDifference,
                    ", maxRelativeResidual = ", maxRelativeResidual);
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

        runMDWFMultiShiftNormalMdwfSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
