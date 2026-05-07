/*
 * MDWF nonzero-c_sw clover sanity test.
 *
 * This test only checks that the existing clover-capable Wilson path produces
 * a nonzero difference from c_sw = 0 on a deterministic nontrivial gauge field.
 * It is not a full physics-correctness validation and does not call CG,
 * RHMC/HMC, force code, or gauge-link smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCloverNonzeroPattern {
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
void runMDWFCloverNonzeroSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_clover_nonzero_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(1337);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_clover_nonzero_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cswZeroOut(commBase, "MDWF_clover_nonzero_csw0_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cswNonzeroOut(commBase, "MDWF_clover_nonzero_csw_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonPart(commBase, "MDWF_clover_nonzero_wilson_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> fifthPart(commBase, "MDWF_clover_nonzero_fifth_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonTmp(commBase, "MDWF_clover_nonzero_wilson_tmp");
    MDWFSpinor<double, false, All, HaloDepth, Ls> cswZeroHost(commBase, "MDWF_clover_nonzero_csw0_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> cswNonzeroHost(commBase, "MDWF_clover_nonzero_csw_host");

    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuUpper(commBase, "MDWF_clover_nonzero_fmunu_upper");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuLower(commBase, "MDWF_clover_nonzero_fmunu_lower");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuInvUpper(commBase, "MDWF_clover_nonzero_fmunu_inv_upper");
    Spinorfield<double, true, All, HaloDepth, 18, 1> fmunuInvLower(commBase, "MDWF_clover_nonzero_fmunu_inv_lower");

    spinorIn.template iterateOverBulk<>(FillMDWFCloverNonzeroPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);

    applyMDWFCloverOperator<double, HaloDepth, HaloDepth, Ls>(
        cswZeroOut, gauge, wilsonPart, fifthPart, wilsonTmp, fmunuUpper, fmunuLower,
        fmunuInvUpper, fmunuInvLower, spinorIn, coeff, 1.0, 0.0, true);
    applyMDWFCloverOperator<double, HaloDepth, HaloDepth, Ls>(
        cswNonzeroOut, gauge, wilsonPart, fifthPart, wilsonTmp, fmunuUpper, fmunuLower,
        fmunuInvUpper, fmunuInvLower, spinorIn, coeff, 1.0, 0.5, true);

    cswZeroHost = cswZeroOut;
    cswNonzeroHost = cswNonzeroOut;

    Vect12ArrayAcc<double> cswZeroAcc = cswZeroHost.getAccessor();
    Vect12ArrayAcc<double> cswNonzeroAcc = cswNonzeroHost.getAccessor();

    double maxDiff = 0.0;
    double diffNorm = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> cswZero = cswZeroAcc.getElement(site);
            Vect12<double> cswNonzero = cswNonzeroAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(cswNonzero.data[component] - cswZero.data[component]))
                                    + std::abs(imag(cswNonzero.data[component] - cswZero.data[component]));
                if (!std::isfinite(diff)) {
                    throw std::runtime_error(stdLogger.fatal("MDWF nonzero-c_sw sanity test produced non-finite diff"));
                }
                diffNorm += diff;
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (maxDiff <= 1e-12 || diffNorm <= 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw sanity test found no clover response; maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF nonzero-c_sw sanity response detected with Ls = ", Ls,
                    ", c_sw = 0.5, maxDiff = ", maxDiff, ", diffNorm = ", diffNorm);
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

        runMDWFCloverNonzeroSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
