/*
 * First safe MDWF-operator CG scaffold test.
 *
 * This test uses the real MDWFLinearOperator with c_sw = 0, but it does not
 * attempt a nonzero CG solve with the raw MDWF operator.  CG is only tested on
 * a zero RHS, which must converge immediately for any linear operator.  A
 * separate nonzero-source check preserves the c_sw = 0 regression by comparing
 * the clover-routed MDWFLinearOperator against the unclovered MDWF workspace
 * path.  It does not touch RHMC/HMC, force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCoupledCGMdwfCsw0Pattern {
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
void runMDWFCoupledCGMdwfCsw0SmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using LinearOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls>;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_coupled_cg_mdwf_csw0_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> probe(commBase, "MDWF_coupled_cg_mdwf_csw0_probe");
    MDWFSpinor<double, true, All, HaloDepth, Ls> linearOut(commBase, "MDWF_coupled_cg_mdwf_csw0_linear_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> referenceOut(commBase, "MDWF_coupled_cg_mdwf_csw0_reference_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> zeroSource(commBase, "MDWF_coupled_cg_mdwf_csw0_zero_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> zeroSolution(commBase, "MDWF_coupled_cg_mdwf_csw0_zero_solution");
    MDWFSpinor<double, true, All, HaloDepth, Ls> operatorZeroOut(commBase, "MDWF_coupled_cg_mdwf_csw0_operator_zero_out");
    MDWFSpinor<double, false, All, HaloDepth, Ls> linearHost(commBase, "MDWF_coupled_cg_mdwf_csw0_linear_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> referenceHost(commBase, "MDWF_coupled_cg_mdwf_csw0_reference_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> zeroSolutionHost(commBase, "MDWF_coupled_cg_mdwf_csw0_zero_solution_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> operatorZeroHost(commBase, "MDWF_coupled_cg_mdwf_csw0_operator_zero_host");

    probe.template iterateOverBulk<>(FillMDWFCoupledCGMdwfCsw0Pattern<double, All, HaloDepth, Ls>());
    probe.updateAll();
    zeroSource = static_cast<double>(0.0) * probe;
    zeroSource.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);
    LinearOperator linearOperator(gauge, coeff, 1.0, 0.0, "MDWF_coupled_cg_mdwf_csw0_linear_operator");
    Adapter adapter(linearOperator);
    MDWFOperatorWorkspace<double, HaloDepth, HaloDepth, Ls> referenceWorkspace(
        gauge, "MDWF_coupled_cg_mdwf_csw0_reference_workspace");

    linearOperator.apply(linearOut, probe, true);
    referenceWorkspace.apply(referenceOut, probe, coeff, 1.0, true);

    MDWFCoupledCG<double, Adapter> cg;
    MDWFCoupledCGResult<double> result = cg.invert(adapter, zeroSolution, zeroSource, 8, 1e-12, true);
    adapter.apply(operatorZeroOut, zeroSource, true);

    linearHost = linearOut;
    referenceHost = referenceOut;
    zeroSolutionHost = zeroSolution;
    operatorZeroHost = operatorZeroOut;

    Vect12ArrayAcc<double> linearAcc = linearHost.getAccessor();
    Vect12ArrayAcc<double> referenceAcc = referenceHost.getAccessor();
    Vect12ArrayAcc<double> zeroSolutionAcc = zeroSolutionHost.getAccessor();
    Vect12ArrayAcc<double> operatorZeroAcc = operatorZeroHost.getAccessor();

    double csw0MaxDiff = 0.0;
    double zeroSolutionMax = 0.0;
    double operatorZeroMax = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> linearValue = linearAcc.getElement(site);
            Vect12<double> referenceValue = referenceAcc.getElement(site);
            Vect12<double> zeroSolutionValue = zeroSolutionAcc.getElement(site);
            Vect12<double> operatorZeroValue = operatorZeroAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double csw0Diff = std::abs(real(linearValue.data[component] - referenceValue.data[component]))
                                        + std::abs(imag(linearValue.data[component] - referenceValue.data[component]));
                const double zeroSolutionAbs = std::abs(real(zeroSolutionValue.data[component]))
                                               + std::abs(imag(zeroSolutionValue.data[component]));
                const double operatorZeroAbs = std::abs(real(operatorZeroValue.data[component]))
                                               + std::abs(imag(operatorZeroValue.data[component]));
                if (csw0Diff > csw0MaxDiff) {
                    csw0MaxDiff = csw0Diff;
                }
                if (zeroSolutionAbs > zeroSolutionMax) {
                    zeroSolutionMax = zeroSolutionAbs;
                }
                if (operatorZeroAbs > operatorZeroMax) {
                    operatorZeroMax = operatorZeroAbs;
                }
            }
        }

    if (linearOperator.csw() != 0.0
        || !result.converged
        || result.iterations != 0
        || result.residue > 1e-12
        || csw0MaxDiff > 1e-12
        || zeroSolutionMax > 1e-12
        || operatorZeroMax > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 coupled CG scaffold test failed: c_sw = ", linearOperator.csw(),
            ", converged = ", result.converged,
            ", iterations = ", result.iterations,
            ", residue = ", result.residue,
            ", csw0MaxDiff = ", csw0MaxDiff,
            ", zeroSolutionMax = ", zeroSolutionMax,
            ", operatorZeroMax = ", operatorZeroMax));
    }

    rootLogger.info("MDWF c_sw = 0 coupled CG scaffold test passed with Ls = ", Ls,
                    ", csw0MaxDiff = ", csw0MaxDiff,
                    ", zeroSolutionMax = ", zeroSolutionMax,
                    ", operatorZeroMax = ", operatorZeroMax);
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

        runMDWFCoupledCGMdwfCsw0SmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
