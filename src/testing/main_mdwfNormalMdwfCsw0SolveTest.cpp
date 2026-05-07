/*
 * MDWF normal-equation CG solve smoke test.
 *
 * This is the first nonzero-source CG solve using the raw MDWF forward
 * operator.  The solve is performed only through the explicitly supplied
 * normal form N = M^\dagger M, with M supplied by MDWFLinearOperator and
 * M^\dagger supplied by MDWFAdjointLinearOperator.  The test uses c_sw = 0,
 * checks the adjoint identity <x, M y> = <M^\dagger x, y>, and verifies the
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
struct FillMDWFNormalMdwfPattern {
    floatT offset;

    explicit FillMDWFNormalMdwfPattern(floatT offset_in)
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
void runMDWFNormalMdwfCsw0SolveSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_normal_mdwf_csw0_gauge");
    gauge.one();

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_normal_mdwf_csw0_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, "MDWF_normal_mdwf_csw0_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> normalSolution(commBase, "MDWF_normal_mdwf_csw0_normal_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> residual(commBase, "MDWF_normal_mdwf_csw0_residual");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, "MDWF_normal_mdwf_csw0_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, "MDWF_normal_mdwf_csw0_probe_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, "MDWF_normal_mdwf_csw0_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, "MDWF_normal_mdwf_csw0_adjoint_x");

    source.template iterateOverBulk<>(FillMDWFNormalMdwfPattern<double, All, HaloDepth, Ls>(0.25));
    probeX.template iterateOverBulk<>(FillMDWFNormalMdwfPattern<double, All, HaloDepth, Ls>(0.75));
    probeY.template iterateOverBulk<>(FillMDWFNormalMdwfPattern<double, All, HaloDepth, Ls>(1.25));
    source.updateAll();
    probeX.updateAll();
    probeY.updateAll();

    MDWFFifthDimCoefficients<double> coeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, coeff, 4.0, 0.0, "MDWF_normal_mdwf_csw0_forward");
    AdjointOperator adjoint(gauge, coeff, 4.0, 0.0, "MDWF_normal_mdwf_csw0_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_normal_mdwf_csw0_normal");
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

    MDWFCoupledCG<double, NormalAdapter> cg;
    MDWFCoupledCGResult<double> result = cg.invert(normalAdapter, solution, source, 256, 1e-8, true);

    normal.apply(normalSolution, solution, true);
    residual = source;
    residual -= normalSolution;

    const double sourceNorm = normalAdapter.norm2(source);
    const double residualNorm = normalAdapter.norm2(residual);
    const double relativeResidual = std::sqrt(residualNorm / std::max(sourceNorm, 1.0));

    if (forward.csw() != 0.0
        || adjoint.csw() != 0.0
        || adjointRelDiff > 1e-9
        || !result.converged
        || result.residue > 1e-8
        || relativeResidual > 1e-7) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF normal-equation c_sw = 0 solve failed: forward_csw = ", forward.csw(),
            ", adjoint_csw = ", adjoint.csw(),
            ", adjointRelDiff = ", adjointRelDiff,
            ", converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", relativeResidual = ", relativeResidual));
    }

    rootLogger.info("MDWF normal-equation c_sw = 0 solve passed with Ls = ", Ls,
                    ", iterations = ", result.iterations,
                    ", residue = ", result.residue,
                    ", relativeResidual = ", relativeResidual,
                    ", adjointRelDiff = ", adjointRelDiff);
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

        runMDWFNormalMdwfCsw0SolveSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
