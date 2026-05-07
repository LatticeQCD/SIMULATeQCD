/*
 * MDWF nonzero-c_sw normal-equation CG solve smoke test.
 *
 * This is a true nonzero-c_sw CG solve through the explicitly supplied normal
 * form N = M^\dagger M, with M supplied by MDWFLinearOperator and M^\dagger
 * supplied by MDWFAdjointLinearOperator.  The test checks the nonzero-c_sw
 * adjoint identity, a positive Rayleigh quotient for N, and the final
 * normal-equation residual.  It does not touch RHMC/HMC, force code, HISQ, or
 * smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFNormalMdwfNonzeroCswSolvePattern {
    floatT offset;

    explicit FillMDWFNormalMdwfNonzeroCswSolvePattern(floatT offset_in)
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
void runMDWFNormalMdwfNonzeroCswSolveSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260507);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalSolution(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_normal_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> residual(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_residual");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_probe_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_adjoint_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalProbeY(commBase, "MDWF_normal_mdwf_nonzero_csw_solve_normal_probe_y");

    source.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswSolvePattern<double, All, HaloDepth, Ls>(0.25));
    probeX.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswSolvePattern<double, All, HaloDepth, Ls>(0.75));
    probeY.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswSolvePattern<double, All, HaloDepth, Ls>(1.25));
    source.updateAll();
    probeX.updateAll();
    probeY.updateAll();

    MDWFFifthDimCoefficients<double> coeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, coeff, 4.0, nonzeroCsw, "MDWF_normal_mdwf_nonzero_csw_solve_forward");
    AdjointOperator adjoint(gauge, coeff, 4.0, nonzeroCsw, "MDWF_normal_mdwf_nonzero_csw_solve_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_normal_mdwf_nonzero_csw_solve_normal");
    ForwardAdapter forwardAdapter(forward);
    NormalAdapter normalAdapter(normal);

    forward.apply(forwardY, probeY, true);
    adjoint.apply(adjointX, probeX, true);
    const COMPLEX(double) left = forwardAdapter.dotProduct5D(probeX, forwardY);
    const COMPLEX(double) right = forwardAdapter.dotProduct5D(adjointX, probeY);
    const double adjointDiff = std::abs(real(left - right)) + std::abs(imag(left - right));
    const double adjointScale = std::max(1.0, std::max(std::abs(real(left)) + std::abs(imag(left)),
                                                       std::abs(real(right)) + std::abs(imag(right))));
    const double adjointRelDiff = adjointDiff / adjointScale;

    normal.apply(normalProbeY, probeY, true);
    const COMPLEX(double) rayleigh = normalAdapter.dotProduct5D(probeY, normalProbeY);
    const double rayleighReal = real<double>(rayleigh);
    const double rayleighImag = imag<double>(rayleigh);
    const double rayleighImagRel = std::abs(rayleighImag) / std::max(std::abs(rayleighReal), 1.0);

    MDWFCoupledCG<double, NormalAdapter> cg;
    MDWFCoupledCGResult<double> result = cg.invert(normalAdapter, solution, source, 512, 1e-8, true);

    normal.apply(normalSolution, solution, true);
    residual = source;
    residual -= normalSolution;

    const double sourceNorm2 = normalAdapter.norm2(source);
    const double residualNorm2 = normalAdapter.norm2(residual);
    const double relativeResidual = std::sqrt(residualNorm2 / std::max(sourceNorm2, 1.0));

    if (forward.csw() != nonzeroCsw
        || adjoint.csw() != nonzeroCsw
        || adjointRelDiff > 1e-9
        || !std::isfinite(rayleighReal)
        || !std::isfinite(rayleighImag)
        || rayleighReal <= 0.0
        || rayleighImagRel > 1e-10
        || !result.converged
        || result.residue > 1e-8
        || relativeResidual > 1e-7) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw normal-equation solve failed: forward_csw = ", forward.csw(),
            ", adjoint_csw = ", adjoint.csw(),
            ", adjointRelDiff = ", adjointRelDiff,
            ", rayleighReal = ", rayleighReal,
            ", rayleighImag = ", rayleighImag,
            ", rayleighImagRel = ", rayleighImagRel,
            ", converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", relativeResidual = ", relativeResidual));
    }

    rootLogger.info("MDWF nonzero-c_sw normal-equation solve passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", iterations = ", result.iterations,
                    ", residue = ", result.residue,
                    ", relativeResidual = ", relativeResidual,
                    ", adjointRelDiff = ", adjointRelDiff,
                    ", rayleighReal = ", rayleighReal,
                    ", rayleighImagRel = ", rayleighImagRel);
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

        runMDWFNormalMdwfNonzeroCswSolveSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
