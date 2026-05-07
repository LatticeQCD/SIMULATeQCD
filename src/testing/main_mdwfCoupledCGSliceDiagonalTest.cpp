/*
 * MDWF coupled-5D CG slice-diagonal SPD smoke test.
 *
 * This test validates the coupled CG scaffold on a controlled positive
 * definite mock operator with fifth-slice eigenvalues A_s = 1 + 0.1 s.  The
 * exact solution is x_s = b_s / A_s, while the nontrivial spectrum should
 * require more than one CG iteration.  It does not apply the MDWF operator,
 * does not call the existing CG/MRHS inverter, and does not touch RHMC/HMC,
 * force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFSliceDiagonalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFSliceDiagonalApply(const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0) + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFSliceDiagonalLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFSliceDiagonalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCoupledCGSliceDiagonalPattern {
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
void runMDWFCoupledCGSliceDiagonalSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using SliceDiagonalOperator = MDWFSliceDiagonalLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, SliceDiagonalOperator>;

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_coupled_cg_slice_diagonal_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, "MDWF_coupled_cg_slice_diagonal_solution");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(commBase, "MDWF_coupled_cg_slice_diagonal_source_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> solutionHost(commBase, "MDWF_coupled_cg_slice_diagonal_solution_host");

    source.template iterateOverBulk<>(FillMDWFCoupledCGSliceDiagonalPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    SliceDiagonalOperator sliceDiagonalOperator;
    Adapter adapter(sliceDiagonalOperator);
    MDWFCoupledCG<double, Adapter> cg;

    MDWFCoupledCGResult<double> result = cg.invert(adapter, solution, source, 32, 1e-12, true);

    sourceHost = source;
    solutionHost = solution;

    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();
    Vect12ArrayAcc<double> solutionAcc = solutionHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            const double diagonal = 1.0 + 0.1 * static_cast<double>(stack);
            Vect12<double> sourceValue = sourceAcc.getElement(site);
            Vect12<double> solutionValue = solutionAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const COMPLEX(double) expected = sourceValue.data[component] / diagonal;
                const double diff = std::abs(real(solutionValue.data[component] - expected))
                                    + std::abs(imag(solutionValue.data[component] - expected));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (!result.converged || result.iterations <= 1 || result.residue > 1e-12 || maxDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF coupled CG slice-diagonal smoke test failed: converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF coupled CG slice-diagonal smoke test passed with Ls = ", Ls,
                    ", iterations = ", result.iterations,
                    ", residue = ", result.residue,
                    ", maxDiff = ", maxDiff);
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

        runMDWFCoupledCGSliceDiagonalSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
