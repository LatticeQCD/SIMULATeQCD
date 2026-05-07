/*
 * MDWF coupled-5D CG nearest-neighbor fifth-direction SPD smoke test.
 *
 * This test validates the coupled CG scaffold on an open-boundary tridiagonal
 * fifth-direction mock operator: A = 2 I - 0.25 T_s, where T_s couples nearest
 * fifth slices.  The operator is SPD by construction and the exact solution is
 * computed on the host with a small tridiagonal solve for each 4D site/component.
 * It does not apply the MDWF operator, does not call the existing CG/MRHS
 * inverter, and does not touch RHMC/HMC, force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"

#include <array>
#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFifthNeighborApply {
    Vect12ArrayAcc<floatT> spinor_in;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    explicit MDWFFifthNeighborApply(const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out = static_cast<floatT>(2.0) * spinor_in.getElement(site);

        if (site.stack > 0) {
            out += static_cast<floatT>(-0.25)
                   * spinor_in.getElement(GInd::getSiteStack(site, site.stack - 1));
        }
        if (site.stack + 1 < Ls) {
            out += static_cast<floatT>(-0.25)
                   * spinor_in.getElement(GInd::getSiteStack(site, site.stack + 1));
        }
        return out;
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFFifthNeighborLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFFifthNeighborApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCoupledCGFifthNeighborPattern {
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
std::array<double, Ls> solveFifthNeighborExact(const std::array<double, Ls> &rhs) {
    const double diagonal = 2.0;
    const double off_diagonal = -0.25;

    std::array<double, Ls> modified_upper;
    std::array<double, Ls> modified_rhs;
    std::array<double, Ls> solution;

    modified_upper[0] = off_diagonal / diagonal;
    modified_rhs[0] = rhs[0] / diagonal;

    for (size_t s = 1; s < Ls; s++) {
        const double denominator = diagonal - off_diagonal * modified_upper[s - 1];
        modified_upper[s] = (s + 1 < Ls) ? off_diagonal / denominator : 0.0;
        modified_rhs[s] = (rhs[s] - off_diagonal * modified_rhs[s - 1]) / denominator;
    }

    solution[Ls - 1] = modified_rhs[Ls - 1];
    for (size_t s_reverse = Ls - 1; s_reverse > 0; s_reverse--) {
        const size_t s = s_reverse - 1;
        solution[s] = modified_rhs[s] - modified_upper[s] * solution[s + 1];
    }

    return solution;
}

template<size_t Ls>
void runMDWFCoupledCGFifthNeighborSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using FifthNeighborOperator = MDWFFifthNeighborLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, FifthNeighborOperator>;

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_coupled_cg_fifth_neighbor_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> solution(commBase, "MDWF_coupled_cg_fifth_neighbor_solution");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(commBase, "MDWF_coupled_cg_fifth_neighbor_source_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> solutionHost(commBase, "MDWF_coupled_cg_fifth_neighbor_solution_host");

    source.template iterateOverBulk<>(FillMDWFCoupledCGFifthNeighborPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    FifthNeighborOperator fifthNeighborOperator;
    Adapter adapter(fifthNeighborOperator);
    MDWFCoupledCG<double, Adapter> cg;

    MDWFCoupledCGResult<double> result = cg.invert(adapter, solution, source, 64, 1e-12, true);

    sourceHost = source;
    solutionHost = solution;

    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();
    Vect12ArrayAcc<double> solutionAcc = solutionHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t component = 0; component < 12; component++) {
            std::array<double, Ls> rhs;
            for (size_t stack = 0; stack < Ls; stack++) {
                const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
                rhs[stack] = real(sourceAcc.getElement(site).data[component]);
            }

            const std::array<double, Ls> expected = solveFifthNeighborExact<Ls>(rhs);

            for (size_t stack = 0; stack < Ls; stack++) {
                const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
                const COMPLEX(double) solutionValue = solutionAcc.getElement(site).data[component];
                const double diff = std::abs(real(solutionValue) - expected[stack])
                                    + std::abs(imag(solutionValue));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (!result.converged || result.iterations <= 1 || result.residue > 1e-12 || maxDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF coupled CG fifth-neighbor smoke test failed: converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF coupled CG fifth-neighbor smoke test passed with Ls = ", Ls,
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

        runMDWFCoupledCGFifthNeighborSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
