/*
 * MDWF clover c_sw = 0 regression smoke test.
 *
 * This test verifies that routing the MDWF Wilson slice through the existing
 * clover-capable Wilson path reproduces the unclovered MDWF operator when
 * c_sw = 0.  It does not enable nonzero clover, call CG, RHMC/HMC, force code,
 * or gauge-link smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCloverRegressionPattern {
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
void runMDWFCloverCsw0SmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_clover_csw0_gauge");
    gauge.one();
    gauge.updateAll();

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_clover_csw0_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> uncloveredOut(commBase, "MDWF_clover_csw0_unclovered_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cloverOut(commBase, "MDWF_clover_csw0_clover_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonPart(commBase, "MDWF_clover_csw0_wilson_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> fifthPart(commBase, "MDWF_clover_csw0_fifth_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonTmp(commBase, "MDWF_clover_csw0_wilson_tmp");
    MDWFSpinor<double, false, All, HaloDepth, Ls> uncloveredHost(commBase, "MDWF_clover_csw0_unclovered_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> cloverHost(commBase, "MDWF_clover_csw0_clover_host");

    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuUpper(commBase, "MDWF_clover_csw0_fmunu_upper");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuLower(commBase, "MDWF_clover_csw0_fmunu_lower");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuInvUpper(commBase, "MDWF_clover_csw0_fmunu_inv_upper");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuInvLower(commBase, "MDWF_clover_csw0_fmunu_inv_lower");

    spinorIn.template iterateOverBulk<>(FillMDWFCloverRegressionPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);

    applyMDWFOperator<double, HaloDepth, HaloDepth, Ls>(
        uncloveredOut, gauge, wilsonPart, fifthPart, wilsonTmp, spinorIn, coeff, 1.0, 0.0, true);
    applyMDWFCloverOperator<double, HaloDepth, HaloDepth, Ls>(
        cloverOut, gauge, wilsonPart, fifthPart, wilsonTmp, fmunuUpper, fmunuLower,
        fmunuInvUpper, fmunuInvLower, spinorIn, coeff, 1.0, 0.0, true);

    uncloveredHost = uncloveredOut;
    cloverHost = cloverOut;

    Vect12ArrayAcc<double> uncloveredAcc = uncloveredHost.getAccessor();
    Vect12ArrayAcc<double> cloverAcc = cloverHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> unclovered = uncloveredAcc.getElement(site);
            Vect12<double> clover = cloverAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(clover.data[component] - unclovered.data[component]))
                                    + std::abs(imag(clover.data[component] - unclovered.data[component]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (maxDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 clover regression failed with maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF c_sw = 0 clover regression passed with Ls = ", Ls);
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

        runMDWFCloverCsw0SmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
