/*
 * MDWF fifth-direction smoke test.
 *
 * This test only exercises the local fifth-direction coupling over the
 * Spinorfield stack dimension.  It does not call Wilson Dslash, clover, CG,
 * RHMC/HMC, force code, or any gauge-link smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFFifthDim.h"

#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFStackPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t i = 0; i < 12; i++) {
            out.data[i] = COMPLEX(floatT)(static_cast<floatT>(100 * site.stack + i + 1), 0.0);
        }
        return out;
    }
};

template<size_t Ls>
double expectedFifthDimValue(size_t stack,
                             size_t component,
                             const MDWFFifthDimCoefficients<double> &coeff) {
    const size_t forward_stack = (stack + 1 == Ls) ? 0 : stack + 1;
    const size_t backward_stack = (stack == 0) ? Ls - 1 : stack - 1;
    const double source_value = static_cast<double>(100 * stack + component + 1);
    const double forward_value = static_cast<double>(100 * forward_stack + component + 1);
    const double backward_value = static_cast<double>(100 * backward_stack + component + 1);
    const double forward_coeff = (stack + 1 == Ls) ? coeff.forward_boundary : coeff.forward_hop;
    const double backward_coeff = (stack == 0) ? coeff.backward_boundary : coeff.backward_hop;

    double expected = coeff.diagonal * source_value;
    if (component < 6) {
        expected += backward_coeff * backward_value;
    } else {
        expected += forward_coeff * forward_value;
    }
    return expected;
}

template<size_t Ls>
void runFifthDimSmokeTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_spinor_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorOut(commBase, "MDWF_spinor_out");
    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorOutHost(commBase, "MDWF_spinor_out_host");

    spinorIn.template iterateOverBulk<>(FillMDWFStackPattern<double, All, HaloDepth, Ls>());

    MDWFFifthDimCoefficients<double> coeff(2.0, 3.0, 5.0, 7.0, 11.0);
    applyMDWFFifthDimCoupling(spinorOut, spinorIn, coeff);

    spinorOutHost = spinorOut;
    Vect12ArrayAcc<double> outAcc = spinorOutHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t x = 0; x < GInd::getLatData().lx; x++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
                for (size_t t = 0; t < GInd::getLatData().lt; t++)
                    for (size_t stack = 0; stack < Ls; stack++) {
                        Vect12<double> out = outAcc.getElement(GInd::getSiteStack(x, y, z, t, stack));
                        for (size_t component = 0; component < 12; component++) {
                            const double expected = expectedFifthDimValue<Ls>(stack, component, coeff);
                            const double diff = std::abs(real(out.data[component]) - expected)
                                                + std::abs(imag(out.data[component]));
                            if (diff > maxDiff) {
                                maxDiff = diff;
                            }
                        }
                    }

    if (maxDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal("MDWF fifth-direction smoke test failed with maxDiff = ", maxDiff));
    }
    rootLogger.info("MDWF fifth-direction smoke test passed with Ls = ", Ls);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/stackedSpinorTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runFifthDimSmokeTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
