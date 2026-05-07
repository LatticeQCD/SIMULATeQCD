/*
 * MDWF coupled-solver adapter smoke test.
 *
 * This test checks the non-solving adapter layer that future Krylov code should
 * use for true coupled-5D vector operations.  It verifies that stack-wise inner
 * products are summed into one 5D scalar and that apply() remains the same
 * MDWFLinearOperator matvec.  It does not call CG, RHMC/HMC, force code, HISQ,
 * or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"

#include <cmath>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCoupledSolverAdapterPattern {
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
void runMDWFCoupledSolverAdapterSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_coupled_solver_adapter_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_coupled_solver_adapter_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adapterOut(commBase, "MDWF_coupled_solver_adapter_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> referenceOut(commBase, "MDWF_coupled_solver_adapter_reference_out");
    MDWFSpinor<double, false, All, HaloDepth, Ls> adapterHost(commBase, "MDWF_coupled_solver_adapter_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> referenceHost(commBase, "MDWF_coupled_solver_adapter_reference_host");

    spinorIn.template iterateOverBulk<>(FillMDWFCoupledSolverAdapterPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);
    MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls> linearOperator(
        gauge, coeff, 1.0, 0.5, "MDWF_coupled_solver_adapter_linear_operator");
    MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls> adapter(linearOperator);

    adapter.apply(adapterOut, spinorIn, true);
    linearOperator.apply(referenceOut, spinorIn, true);

    adapterHost = adapterOut;
    referenceHost = referenceOut;

    Vect12ArrayAcc<double> adapterAcc = adapterHost.getAccessor();
    Vect12ArrayAcc<double> referenceAcc = referenceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> adapterValue = adapterAcc.getElement(site);
            Vect12<double> referenceValue = referenceAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(adapterValue.data[component] - referenceValue.data[component]))
                                    + std::abs(imag(adapterValue.data[component] - referenceValue.data[component]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    std::vector<double> stackNorms = spinorIn.realdotProductStacked(spinorIn);
    double referenceNorm = 0.0;
    for (size_t stack = 0; stack < Ls; stack++) {
        referenceNorm += stackNorms[stack];
    }
    const double adapterNorm = adapter.norm2(spinorIn);
    const double normDiff = std::abs(adapterNorm - referenceNorm);

    if (maxDiff > 1e-12 || normDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF coupled-solver adapter smoke test failed with maxDiff = ", maxDiff,
            ", normDiff = ", normDiff));
    }

    rootLogger.info("MDWF coupled-solver adapter smoke test passed with Ls = ", Ls,
                    ", norm2 = ", adapterNorm);
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

        runMDWFCoupledSolverAdapterSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
