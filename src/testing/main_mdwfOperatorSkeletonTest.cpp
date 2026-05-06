/*
 * MDWF operator skeleton smoke test.
 *
 * This test verifies that the Stage 4 wrapper combines the slice-wise 4D
 * Wilson application and the fifth-direction coupling as an explicit sum.  It
 * does not call clover, CG, RHMC/HMC, force code, or gauge-link smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFOperatorPattern {
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
void runMDWFOperatorSkeletonSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_operator_gauge");
    gauge.one();
    gauge.updateAll();

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_operator_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorOut(commBase, "MDWF_operator_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> referenceOut(commBase, "MDWF_operator_reference_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonPart(commBase, "MDWF_operator_wilson_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> fifthPart(commBase, "MDWF_operator_fifth_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonTmp(commBase, "MDWF_operator_wilson_tmp");
    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorOutHost(commBase, "MDWF_operator_out_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> referenceOutHost(commBase, "MDWF_operator_reference_host");

    spinorIn.template iterateOverBulk<>(FillMDWFOperatorPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);

    applyMDWFOperator<double, HaloDepth, HaloDepth, Ls>(
        spinorOut, gauge, wilsonPart, fifthPart, wilsonTmp, spinorIn, coeff, 1.0, 0.0, true);

    applyMDWFWilsonSlice<double, HaloDepth, HaloDepth, Ls>(
        referenceOut, gauge, wilsonTmp, spinorIn, 1.0, 0.0);
    applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(fifthPart, spinorIn, coeff);
    referenceOut += fifthPart;
    referenceOut.updateAll();

    spinorOutHost = spinorOut;
    referenceOutHost = referenceOut;

    Vect12ArrayAcc<double> outAcc = spinorOutHost.getAccessor();
    Vect12ArrayAcc<double> refAcc = referenceOutHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> out = outAcc.getElement(site);
            Vect12<double> ref = refAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(out.data[component] - ref.data[component]))
                                    + std::abs(imag(out.data[component] - ref.data[component]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (maxDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF operator skeleton smoke test failed with maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF operator skeleton smoke test passed with Ls = ", Ls);
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

        runMDWFOperatorSkeletonSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
