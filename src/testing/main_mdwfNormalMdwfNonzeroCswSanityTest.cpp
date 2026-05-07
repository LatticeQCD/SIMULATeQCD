/*
 * MDWF nonzero-c_sw normal-operator sanity test.
 *
 * This test applies only the explicitly supplied normal form N = M^\dagger M.
 * It compares c_sw = 0 against nonzero c_sw on the same deterministic source
 * and nontrivial gauge field, and checks that the nonzero-c_sw path produces a
 * finite, nonzero response.  It is not a full physics-correctness validation,
 * does not solve CG, and does not touch RHMC/HMC, force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <cmath>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFNormalMdwfNonzeroCswPattern {
    floatT offset;

    explicit FillMDWFNormalMdwfNonzeroCswPattern(floatT offset_in)
        : offset(offset_in) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                offset + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                0.0);
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFNormalMdwfNonzeroCswSanityTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;
    typedef GIndexer<All, HaloDepth> GInd;

    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_normal_mdwf_nonzero_csw_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260507);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_normal_mdwf_nonzero_csw_source");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, "MDWF_normal_mdwf_nonzero_csw_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, "MDWF_normal_mdwf_nonzero_csw_probe_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, "MDWF_normal_mdwf_nonzero_csw_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, "MDWF_normal_mdwf_nonzero_csw_adjoint_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cswZeroOut(commBase, "MDWF_normal_mdwf_nonzero_csw_zero_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cswNonzeroOut(commBase, "MDWF_normal_mdwf_nonzero_csw_nonzero_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> difference(commBase, "MDWF_normal_mdwf_nonzero_csw_difference");
    MDWFSpinor<double, false, All, HaloDepth, Ls> cswZeroHost(commBase, "MDWF_normal_mdwf_nonzero_csw_zero_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> cswNonzeroHost(commBase, "MDWF_normal_mdwf_nonzero_csw_nonzero_host");

    source.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswPattern<double, All, HaloDepth, Ls>(0.25));
    probeX.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswPattern<double, All, HaloDepth, Ls>(0.75));
    probeY.template iterateOverBulk<>(FillMDWFNormalMdwfNonzeroCswPattern<double, All, HaloDepth, Ls>(1.25));
    source.updateAll();
    probeX.updateAll();
    probeY.updateAll();

    MDWFFifthDimCoefficients<double> coeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forwardCsw0(gauge, coeff, 4.0, 0.0, "MDWF_normal_mdwf_nonzero_csw_forward_csw0");
    AdjointOperator adjointCsw0(gauge, coeff, 4.0, 0.0, "MDWF_normal_mdwf_nonzero_csw_adjoint_csw0");
    ForwardOperator forwardCswNonzero(gauge, coeff, 4.0, nonzeroCsw, "MDWF_normal_mdwf_nonzero_csw_forward");
    AdjointOperator adjointCswNonzero(gauge, coeff, 4.0, nonzeroCsw, "MDWF_normal_mdwf_nonzero_csw_adjoint");
    NormalOperator normalCsw0(commBase, forwardCsw0, adjointCsw0, "MDWF_normal_mdwf_nonzero_csw_normal_csw0");
    NormalOperator normalCswNonzero(commBase, forwardCswNonzero, adjointCswNonzero, "MDWF_normal_mdwf_nonzero_csw_normal");

    ForwardAdapter forwardAdapter(forwardCswNonzero);
    NormalAdapter normalAdapter(normalCswNonzero);

    forwardCswNonzero.apply(forwardY, probeY, true);
    adjointCswNonzero.apply(adjointX, probeX, true);
    const COMPLEX(double) left = forwardAdapter.dotProduct5D(probeX, forwardY);
    const COMPLEX(double) right = forwardAdapter.dotProduct5D(adjointX, probeY);
    const double adjointDiff = std::abs(real(left - right)) + std::abs(imag(left - right));
    const double adjointScale = std::max(1.0, std::max(std::abs(real(left)) + std::abs(imag(left)),
                                                       std::abs(real(right)) + std::abs(imag(right))));
    const double adjointRelDiff = adjointDiff / adjointScale;

    normalCsw0.apply(cswZeroOut, source, true);
    normalCswNonzero.apply(cswNonzeroOut, source, true);

    difference = cswNonzeroOut;
    difference -= cswZeroOut;

    const double cswZeroNorm2 = normalAdapter.norm2(cswZeroOut);
    const double cswNonzeroNorm2 = normalAdapter.norm2(cswNonzeroOut);
    const double diffNorm2 = normalAdapter.norm2(difference);
    const double cswZeroL2 = std::sqrt(cswZeroNorm2);
    const double cswNonzeroL2 = std::sqrt(cswNonzeroNorm2);
    const double diffL2 = std::sqrt(diffNorm2);
    const double relativeDiff = diffL2 / std::max(cswZeroL2, 1.0);

    cswZeroHost = cswZeroOut;
    cswNonzeroHost = cswNonzeroOut;

    Vect12ArrayAcc<double> cswZeroAcc = cswZeroHost.getAccessor();
    Vect12ArrayAcc<double> cswNonzeroAcc = cswNonzeroHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> cswZero = cswZeroAcc.getElement(site);
            Vect12<double> cswNonzero = cswNonzeroAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(cswNonzero.data[component] - cswZero.data[component]))
                                    + std::abs(imag(cswNonzero.data[component] - cswZero.data[component]));
                if (!std::isfinite(diff)) {
                    throw std::runtime_error(stdLogger.fatal(
                        "MDWF nonzero-c_sw normal-equation sanity test produced non-finite component diff"));
                }
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

    if (forwardCsw0.csw() != 0.0
        || adjointCsw0.csw() != 0.0
        || forwardCswNonzero.csw() != nonzeroCsw
        || adjointCswNonzero.csw() != nonzeroCsw
        || adjointRelDiff > 1e-9
        || !std::isfinite(cswZeroNorm2)
        || !std::isfinite(cswNonzeroNorm2)
        || !std::isfinite(diffNorm2)
        || !std::isfinite(cswZeroL2)
        || !std::isfinite(cswNonzeroL2)
        || !std::isfinite(diffL2)
        || !std::isfinite(relativeDiff)
        || diffNorm2 <= 1e-24
        || maxDiff <= 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw normal-equation sanity test failed: csw0_forward = ", forwardCsw0.csw(),
            ", csw0_adjoint = ", adjointCsw0.csw(),
            ", csw_forward = ", forwardCswNonzero.csw(),
            ", csw_adjoint = ", adjointCswNonzero.csw(),
            ", adjointRelDiff = ", adjointRelDiff,
            ", cswZeroNorm2 = ", cswZeroNorm2,
            ", cswNonzeroNorm2 = ", cswNonzeroNorm2,
            ", diffNorm2 = ", diffNorm2,
            ", cswZeroL2 = ", cswZeroL2,
            ", cswNonzeroL2 = ", cswNonzeroL2,
            ", diffL2 = ", diffL2,
            ", relativeDiff = ", relativeDiff,
            ", maxComponentDiff = ", maxDiff));
    }

    rootLogger.info("MDWF nonzero-c_sw normal-equation sanity response detected with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", adjointRelDiff = ", adjointRelDiff,
                    ", diffNorm2 = ", diffNorm2,
                    ", diffL2 = ", diffL2,
                    ", relativeDiff = ", relativeDiff,
                    ", maxComponentDiff = ", maxDiff);
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

        runMDWFNormalMdwfNonzeroCswSanityTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
