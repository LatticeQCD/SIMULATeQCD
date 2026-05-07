/*
 * MDWF normal-operator diagonal mock smoke test.
 *
 * This test validates only the explicit N = M^\dagger M wrapper on a harmless
 * mock operator where M = M^\dagger = 2 I.  Therefore N = 4 I and CG should
 * converge in one iteration to x = b / 4.  It does not apply the MDWF operator,
 * does not define the MDWF adjoint, does not call the existing CG/MRHS inverter,
 * and does not touch RHMC/HMC, force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <cmath>

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFDiagonalForwardOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

private:
    floatT _diagonal;

public:
    explicit MDWFDiagonalForwardOperator(floatT diagonal)
        : _diagonal(diagonal) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out = _diagonal * spinor_in;
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFNormalOperatorDiagonalPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(1000 * site.stack + site.isite + component + 1), 0.0);
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFNormalOperatorDiagonalSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using ForwardOperator = MDWFDiagonalForwardOperator<double, HaloDepth, Ls>;
    using AdjointOperator = MDWFDiagonalForwardOperator<double, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_normal_diagonal_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, "MDWF_normal_diagonal_solution");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(commBase, "MDWF_normal_diagonal_source_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> solutionHost(commBase, "MDWF_normal_diagonal_solution_host");

    source.template iterateOverBulk<>(FillMDWFNormalOperatorDiagonalPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    ForwardOperator forward(2.0);
    AdjointOperator adjoint(2.0);
    NormalOperator normalOperator(commBase, forward, adjoint, "MDWF_normal_diagonal_operator");
    Adapter adapter(normalOperator);
    MDWFCoupledCG<double, Adapter> cg;

    MDWFCoupledCGResult<double> result = cg.invert(adapter, solution, source, 8, 1e-12, true);

    sourceHost = source;
    solutionHost = solution;

    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();
    Vect12ArrayAcc<double> solutionAcc = solutionHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> sourceValue = sourceAcc.getElement(site);
            Vect12<double> solutionValue = solutionAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const COMPLEX(double) expected = 0.25 * sourceValue.data[component];
                const double diff = std::abs(real(solutionValue.data[component] - expected))
                                    + std::abs(imag(solutionValue.data[component] - expected));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (!result.converged || result.iterations != 1 || result.residue > 1e-12 || maxDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF normal-operator diagonal smoke test failed: converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF normal-operator diagonal smoke test passed with Ls = ", Ls,
                    ", iterations = ", result.iterations,
                    ", residue = ", result.residue);
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

        runMDWFNormalOperatorDiagonalSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
