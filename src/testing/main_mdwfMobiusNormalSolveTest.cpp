/*
 * MDWF general-Mobius (RBC/UKQCD convention) normal-equation CG solve test.
 *
 * Solves N x = b with N = M^\dagger M, where M / M^\dagger are the
 * cluster-validated MDWFMobiusCloverLinearOperator /
 * MDWFMobiusCloverAdjointLinearOperator pair, composed through the existing
 * MDWFNormalOperator and solved with the existing isolated MDWFCoupledCG.
 * No new solver or operator code is introduced.
 *
 * Cases, on a fixed nontrivial (random) gauge field, all at c_sw = 0.5:
 *
 *   1. Well-conditioned control (M5 = -2, i.e. positive Wilson kernel mass;
 *      not a domain-wall parameter choice), b5 = 1.5: adjoint identity,
 *      positive real Rayleigh quotient, CG convergence, and an independently
 *      recomputed normal-equation residual.
 *   2. b5 = 1 solve regression (same control M5): the Mobius normal solve
 *      reproduces the existing Shamir clover normal solve
 *      (MDWFLinearOperator / MDWFAdjointLinearOperator), in solution and
 *      iteration count.
 *   3. Physical-like parameters (M5 = 1.8, mf = 0.05, b5 = 1.5): the same
 *      checks as case 1 with a larger iteration budget; the iteration count
 *      is reported as a conditioning diagnostic.
 *
 * This does not touch RHMC/HMC, force code, HISQ, or smearing, and does not
 * validate gauge-dependent domain-wall physics.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusNormalSolvePattern {
    floatT offset;

    explicit FillMDWFMobiusNormalSolvePattern(floatT offset_in) : offset(offset_in) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                offset + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.011) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0017) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

struct MDWFMobiusNormalSolveDiagnostics {
    double adjointRelDiff;
    double rayleighReal;
    double rayleighImagRel;
    int iterations;
    double residue;
    bool converged;
    double relativeResidual;
};

template<size_t HaloDepth, size_t Ls, class ForwardOperator, class AdjointOperator>
MDWFMobiusNormalSolveDiagnostics solveMDWFMobiusNormalCase(
    CommunicationBase &commBase,
    ForwardOperator &forward,
    AdjointOperator &adjoint,
    MDWFSpinor<double, true, All, HaloDepth, Ls> &solution,
    const std::string &name,
    int maxIter) {
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, name + "_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalSolution(commBase, name + "_normal_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> residual(commBase, name + "_residual");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, name + "_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, name + "_probe_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, name + "_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, name + "_adjoint_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalProbeY(commBase, name + "_normal_probe_y");

    source.template iterateOverBulk<>(FillMDWFMobiusNormalSolvePattern<double, All, HaloDepth, Ls>(0.25));
    probeX.template iterateOverBulk<>(FillMDWFMobiusNormalSolvePattern<double, All, HaloDepth, Ls>(0.75));
    probeY.template iterateOverBulk<>(FillMDWFMobiusNormalSolvePattern<double, All, HaloDepth, Ls>(1.25));
    source.updateAll();
    probeX.updateAll();
    probeY.updateAll();

    NormalOperator normal(commBase, forward, adjoint, name + "_normal");
    ForwardAdapter forwardAdapter(forward);
    NormalAdapter normalAdapter(normal);

    MDWFMobiusNormalSolveDiagnostics diag{};

    forward.apply(forwardY, probeY, true);
    adjoint.apply(adjointX, probeX, true);
    const COMPLEX(double) left = forwardAdapter.dotProduct5D(probeX, forwardY);
    const COMPLEX(double) right = forwardAdapter.dotProduct5D(adjointX, probeY);
    const double adjointDiff = std::abs(real(left - right)) + std::abs(imag(left - right));
    const double adjointScale = std::max(1.0, std::max(std::abs(real(left)) + std::abs(imag(left)),
                                                       std::abs(real(right)) + std::abs(imag(right))));
    diag.adjointRelDiff = adjointDiff / adjointScale;

    normal.apply(normalProbeY, probeY, true);
    const COMPLEX(double) rayleigh = normalAdapter.dotProduct5D(probeY, normalProbeY);
    diag.rayleighReal = real<double>(rayleigh);
    diag.rayleighImagRel = std::abs(imag<double>(rayleigh)) / std::max(std::abs(diag.rayleighReal), 1.0);

    MDWFCoupledCG<double, NormalAdapter> cg;
    MDWFCoupledCGResult<double> result = cg.invert(normalAdapter, solution, source, maxIter, 1e-8, true);
    diag.iterations = result.iterations;
    diag.residue = result.residue;
    diag.converged = result.converged;

    normal.apply(normalSolution, solution, true);
    residual = source;
    residual -= normalSolution;
    const double sourceNorm2 = normalAdapter.norm2(source);
    const double residualNorm2 = normalAdapter.norm2(residual);
    diag.relativeResidual = std::sqrt(residualNorm2 / std::max(sourceNorm2, 1.0));

    return diag;
}

bool mdwfMobiusNormalSolvePassed(const MDWFMobiusNormalSolveDiagnostics &diag) {
    return diag.adjointRelDiff <= 1e-9
           && std::isfinite(diag.rayleighReal)
           && diag.rayleighReal > 0.0
           && diag.rayleighImagRel <= 1e-10
           && diag.converged
           && diag.residue <= 1e-8
           && diag.relativeResidual <= 1e-7;
}

template<size_t HaloDepth, size_t Ls>
void maxAbsAndDiff(CommunicationBase &commBase,
                   MDWFSpinor<double, true, All, HaloDepth, Ls> &a,
                   MDWFSpinor<double, true, All, HaloDepth, Ls> &b,
                   double &maxAbsA,
                   double &maxDiff) {
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> aHost(commBase, "MDWF_mobius_normal_solve_cmp_a_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> bHost(commBase, "MDWF_mobius_normal_solve_cmp_b_host");
    aHost = a;
    bHost = b;
    Vect12ArrayAcc<double> aAcc = aHost.getAccessor();
    Vect12ArrayAcc<double> bAcc = bHost.getAccessor();

    maxAbsA = 0.0;
    maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> valA = aAcc.getElement(site);
            Vect12<double> valB = bAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double absA = std::abs(real(valA.data[component])) + std::abs(imag(valA.data[component]));
                const double diff = std::abs(real(valA.data[component]) - real(valB.data[component]))
                                    + std::abs(imag(valA.data[component]) - imag(valB.data[component]));
                maxAbsA = std::max(maxAbsA, absA);
                maxDiff = std::max(maxDiff, diff);
            }
        }
}

template<size_t Ls>
void runMDWFMobiusNormalSolveTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using MobiusForward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusAdjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirForward = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirAdjoint = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;

    const double csw = 0.5;
    const double mf = 0.05;
    const double genericB5 = 1.5;
    const double controlM5 = -2.0;
    const double physicalM5 = 1.8;
    const int controlMaxIter = 2000;
    const int physicalMaxIter = 20000;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_mobius_normal_solve_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260922);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    // --- Case 1: well-conditioned control, generic b5. ---
    MobiusForward controlForward(gauge, controlM5, mf, genericB5, csw, "MDWF_mobius_normal_solve_control_forward");
    MobiusAdjoint controlAdjoint(gauge, controlM5, mf, genericB5, csw, "MDWF_mobius_normal_solve_control_adjoint");
    MDWFSpinor<double, true, All, HaloDepth, Ls> controlSolution(commBase, "MDWF_mobius_normal_solve_control_solution");
    const MDWFMobiusNormalSolveDiagnostics control = solveMDWFMobiusNormalCase<HaloDepth, Ls>(
        commBase, controlForward, controlAdjoint, controlSolution, "MDWF_mobius_normal_solve_control", controlMaxIter);

    // --- Case 2: b5 = 1 Mobius vs Shamir clover normal solve, control M5. ---
    MobiusForward b5OneForward(gauge, controlM5, mf, 1.0, csw, "MDWF_mobius_normal_solve_b5_one_forward");
    MobiusAdjoint b5OneAdjoint(gauge, controlM5, mf, 1.0, csw, "MDWF_mobius_normal_solve_b5_one_adjoint");
    MDWFSpinor<double, true, All, HaloDepth, Ls> b5OneSolution(commBase, "MDWF_mobius_normal_solve_b5_one_solution");
    const MDWFMobiusNormalSolveDiagnostics b5One = solveMDWFMobiusNormalCase<HaloDepth, Ls>(
        commBase, b5OneForward, b5OneAdjoint, b5OneSolution, "MDWF_mobius_normal_solve_b5_one", controlMaxIter);

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double controlMass = mdwfShamirKernelMass(controlM5);
    ShamirForward shamirForward(gauge, shamirCoeff, controlMass, csw, "MDWF_mobius_normal_solve_shamir_forward");
    ShamirAdjoint shamirAdjoint(gauge, shamirCoeff, controlMass, csw, "MDWF_mobius_normal_solve_shamir_adjoint");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shamirSolution(commBase, "MDWF_mobius_normal_solve_shamir_solution");
    const MDWFMobiusNormalSolveDiagnostics shamir = solveMDWFMobiusNormalCase<HaloDepth, Ls>(
        commBase, shamirForward, shamirAdjoint, shamirSolution, "MDWF_mobius_normal_solve_shamir", controlMaxIter);

    double shamirSolutionMaxAbs = 0.0;
    double b5OneSolutionDiff = 0.0;
    maxAbsAndDiff<HaloDepth, Ls>(commBase, shamirSolution, b5OneSolution, shamirSolutionMaxAbs, b5OneSolutionDiff);
    const double b5OneSolutionRelDiff = b5OneSolutionDiff / std::max(shamirSolutionMaxAbs, 1e-300);

    // --- Case 3: physical-like parameters, generic b5. ---
    MobiusForward physicalForward(gauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_normal_solve_physical_forward");
    MobiusAdjoint physicalAdjoint(gauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_normal_solve_physical_adjoint");
    MDWFSpinor<double, true, All, HaloDepth, Ls> physicalSolution(commBase, "MDWF_mobius_normal_solve_physical_solution");
    const MDWFMobiusNormalSolveDiagnostics physical = solveMDWFMobiusNormalCase<HaloDepth, Ls>(
        commBase, physicalForward, physicalAdjoint, physicalSolution, "MDWF_mobius_normal_solve_physical",
        physicalMaxIter);

    rootLogger.info("MDWF Mobius normal solve control (M5 = ", controlM5, ", b5 = ", genericB5,
                    "): iterations = ", control.iterations, ", residue = ", control.residue,
                    ", relativeResidual = ", control.relativeResidual,
                    ", adjointRelDiff = ", control.adjointRelDiff,
                    ", rayleighReal = ", control.rayleighReal,
                    ", rayleighImagRel = ", control.rayleighImagRel);
    rootLogger.info("MDWF Mobius normal solve b5 = 1 regression (M5 = ", controlM5,
                    "): mobius iterations = ", b5One.iterations, ", shamir iterations = ", shamir.iterations,
                    ", mobius relativeResidual = ", b5One.relativeResidual,
                    ", shamir relativeResidual = ", shamir.relativeResidual,
                    ", solutionMaxAbsDiff = ", b5OneSolutionDiff,
                    ", solutionRelDiff = ", b5OneSolutionRelDiff);
    rootLogger.info("MDWF Mobius normal solve physical-like (M5 = ", physicalM5, ", mf = ", mf,
                    ", b5 = ", genericB5, "): iterations = ", physical.iterations,
                    ", converged = ", physical.converged, ", residue = ", physical.residue,
                    ", relativeResidual = ", physical.relativeResidual,
                    ", adjointRelDiff = ", physical.adjointRelDiff,
                    ", rayleighReal = ", physical.rayleighReal,
                    ", rayleighImagRel = ", physical.rayleighImagRel);

    const bool b5OneRegressionPassed = b5One.iterations == shamir.iterations && b5OneSolutionRelDiff <= 1e-12;

    if (!mdwfMobiusNormalSolvePassed(control)
        || !mdwfMobiusNormalSolvePassed(b5One)
        || !mdwfMobiusNormalSolvePassed(shamir)
        || !b5OneRegressionPassed
        || !mdwfMobiusNormalSolvePassed(physical)) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius normal solve test failed: control passed = ", mdwfMobiusNormalSolvePassed(control),
            ", b5 = 1 mobius passed = ", mdwfMobiusNormalSolvePassed(b5One),
            ", b5 = 1 shamir passed = ", mdwfMobiusNormalSolvePassed(shamir),
            ", b5 = 1 regression passed = ", b5OneRegressionPassed,
            ", physical-like passed = ", mdwfMobiusNormalSolvePassed(physical),
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF Mobius normal solve test passed with Ls = ", Ls, ", c_sw = ", csw);
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

        runMDWFMobiusNormalSolveTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
