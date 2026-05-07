/*
 * MDWF linear-operator scaffold smoke test.
 *
 * This test verifies that MDWFLinearOperator::apply() is only a thin wrapper
 * around MDWFOperatorWorkspace::applyClover().  It does not call CG, does not
 * use applyMdaggM(), and does not touch RHMC/HMC, force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFLinearOperator.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFLinearOperatorPattern {
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
void runMDWFLinearOperatorSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_linear_operator_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_linear_operator_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> linearOut(commBase, "MDWF_linear_operator_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> referenceOut(commBase, "MDWF_linear_operator_reference_out");
    MDWFSpinor<double, false, All, HaloDepth, Ls> linearHost(commBase, "MDWF_linear_operator_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> referenceHost(commBase, "MDWF_linear_operator_reference_host");

    spinorIn.template iterateOverBulk<>(FillMDWFLinearOperatorPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);
    MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls> linearOperator(
        gauge, coeff, 1.0, 0.5, "MDWF_linear_operator_internal");
    MDWFOperatorWorkspace<double, HaloDepth, HaloDepth, Ls> referenceWorkspace(
        gauge, "MDWF_linear_operator_reference_workspace");

    linearOperator.apply(linearOut, spinorIn, true);
    referenceWorkspace.applyClover(referenceOut, spinorIn, coeff, 1.0, 0.5, true);

    linearHost = linearOut;
    referenceHost = referenceOut;

    Vect12ArrayAcc<double> linearAcc = linearHost.getAccessor();
    Vect12ArrayAcc<double> referenceAcc = referenceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> linearValue = linearAcc.getElement(site);
            Vect12<double> referenceValue = referenceAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(linearValue.data[component] - referenceValue.data[component]))
                                    + std::abs(imag(linearValue.data[component] - referenceValue.data[component]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (maxDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF linear-operator scaffold smoke test failed with maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF linear-operator scaffold smoke test passed with Ls = ", Ls);
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

        runMDWFLinearOperatorSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
