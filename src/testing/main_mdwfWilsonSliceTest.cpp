/*
 * MDWF slice-wise Wilson smoke test.
 *
 * This test only verifies that the existing 4D Wilson kernel can be applied
 * independently on MDWFSpinor stacks.  It does not call fifth-direction
 * coupling, clover, CG, RHMC/HMC, force code, or gauge-link smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFWilsonSlice.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls, size_t SourceStack>
struct FillSingleMDWFStackPattern {
    static_assert(SourceStack < Ls, "SourceStack must be inside the MDWF fifth dimension");

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        if (site.stack == SourceStack) {
            for (size_t component = 0; component < 12; component++) {
                out.data[component] = COMPLEX(floatT)(
                    static_cast<floatT>(site.isite + component + 1), 0.0);
            }
        }
        return out;
    }
};

template<size_t Ls, size_t SourceStack>
void runWilsonSliceSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_wilson_slice_gauge");
    gauge.one();
    gauge.updateAll();

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_wilson_slice_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorOut(commBase, "MDWF_wilson_slice_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorTmp(commBase, "MDWF_wilson_slice_tmp");
    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorOutHost(commBase, "MDWF_wilson_slice_out_host");

    spinorIn.template iterateOverBulk<>(
        FillSingleMDWFStackPattern<double, All, HaloDepth, Ls, SourceStack>());
    spinorIn.updateAll();

    applyMDWFWilsonSlice<double, HaloDepth, HaloDepth, Ls>(
        spinorOut, gauge, spinorTmp, spinorIn, 1.0, 0.0, true);

    spinorOutHost = spinorOut;
    Vect12ArrayAcc<double> outAcc = spinorOutHost.getAccessor();

    double inactiveStackMax = 0.0;
    double sourceStackNorm = 0.0;

    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            Vect12<double> out = outAcc.getElement(GInd::getSiteStack(GInd::getSite(isite), stack));
            for (size_t component = 0; component < 12; component++) {
                const double magnitude = std::abs(real(out.data[component]))
                                         + std::abs(imag(out.data[component]));
                if (stack == SourceStack) {
                    sourceStackNorm += magnitude;
                } else if (magnitude > inactiveStackMax) {
                    inactiveStackMax = magnitude;
                }
            }
        }

    if (inactiveStackMax > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Wilson slice smoke test leaked into inactive stacks with max = ", inactiveStackMax));
    }
    if (sourceStackNorm <= 0.0) {
        throw std::runtime_error(stdLogger.fatal("MDWF Wilson slice smoke test produced zero source-stack norm"));
    }

    rootLogger.info("MDWF Wilson slice smoke test passed with Ls = ", Ls,
                    " and active stack = ", SourceStack);
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

        runWilsonSliceSmokeTest<8, 3>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
