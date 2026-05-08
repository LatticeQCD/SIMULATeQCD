/*
 * MDWF pseudofermion heatbath mock smoke test.
 *
 * This validates only the pre-RHMC heatbath wrapper on a controlled diagonal
 * normal operator with fifth-slice eigenvalues A_s = 1 + 0.1 s.  The input is
 * deterministic Gaussian-like noise, not a real RNG heatbath.  The expected
 * output is phi_s = R_heatbath(A_s) eta_s.  This test does not apply MDWF,
 * does not call RHMC/HMC, and does not touch force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"
#include "../experimental/mdwf/MDWFPseudofermionAction.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFHeatbathMockSliceDiagonalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFHeatbathMockSliceDiagonalApply(
        const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0)
                                + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFHeatbathMockSliceDiagonalNormalOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFHeatbathMockSliceDiagonalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFHeatbathMockNoise {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.01) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.002) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<class floatT>
floatT mdwfHeatbathMockRationalFactor(
    floatT diagonal,
    const MDWFRationalCoefficients<floatT> &coefficients) {

    floatT factor = coefficients.constant;
    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        factor += coefficients.numerator[term] / (diagonal + coefficients.shift[term]);
    }
    return factor;
}

template<class Spinor, size_t Ls>
double compareMDWFHeatbathMockPseudofermion(
    Spinor &pseudofermion,
    Spinor &noise,
    const MDWFRationalCoefficients<double> &coefficients) {

    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> pseudofermionHost(
        pseudofermion.getComm(), "MDWF_heatbath_mock_pseudofermion_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> noiseHost(
        noise.getComm(), "MDWF_heatbath_mock_noise_host");

    pseudofermionHost = pseudofermion;
    noiseHost = noise;

    Vect12ArrayAcc<double> pseudofermionAcc = pseudofermionHost.getAccessor();
    Vect12ArrayAcc<double> noiseAcc = noiseHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            const double diagonal = 1.0 + 0.1 * static_cast<double>(stack);
            const double factor = mdwfHeatbathMockRationalFactor(diagonal, coefficients);
            const Vect12<double> pseudofermionValue = pseudofermionAcc.getElement(site);
            const Vect12<double> noiseValue = noiseAcc.getElement(site);

            for (size_t component = 0; component < 12; component++) {
                const COMPLEX(double) expected = factor * noiseValue.data[component];
                const double diff = std::abs(real(pseudofermionValue.data[component] - expected))
                                    + std::abs(imag(pseudofermionValue.data[component] - expected));
                maxDiff = std::max(maxDiff, diff);
            }
        }
    return maxDiff;
}

template<size_t Ls>
void runMDWFPseudofermionHeatbathMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using NormalOperator = MDWFHeatbathMockSliceDiagonalNormalOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> heatbathInput{
        "mock_heatbath_coefficients",
        MDWFRationalCoefficientRole::Heatbath,
        0.125,
        {0.5, 0.25},
        {0.0, 0.2}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(heatbathInput);

    Spinor noise(commBase, "MDWF_heatbath_mock_noise");
    Spinor pseudofermion(commBase, "MDWF_heatbath_mock_pseudofermion");

    noise.template iterateOverBulk<>(FillMDWFHeatbathMockNoise<double, All, HaloDepth, Ls>());
    noise.updateAll();

    NormalOperator normalOperator;
    Adapter adapter(normalOperator);
    MDWFPseudofermionHeatbathResult<double> result
        = applyMDWFPseudofermionHeatbath<double, Adapter>(
            adapter, pseudofermion, noise, coefficients, 32, 1e-12, "MDWF_heatbath_mock");

    double maxResidue = 0.0;
    bool sawMultipleIterations = false;
    for (size_t term = 0; term < result.rational_result.shifts.size(); term++) {
        maxResidue = std::max(maxResidue, result.rational_result.shifts[term].residue);
        sawMultipleIterations = sawMultipleIterations
                                || result.rational_result.shifts[term].iterations > 1;
    }

    const double maxDiff = compareMDWFHeatbathMockPseudofermion<Spinor, Ls>(
        pseudofermion, noise, coefficients);

    if (mdwfRationalCoefficientRoleName(heatbathInput.role) != "heatbath"
        || !result.rational_result.converged()
        || !sawMultipleIterations
        || maxResidue > 1e-12
        || maxDiff > 1e-8
        || result.noise_norm2 <= 0.0
        || result.pseudofermion_norm2 <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF pseudofermion heatbath mock test failed: converged = ",
            result.rational_result.converged(),
            ", sawMultipleIterations = ", sawMultipleIterations,
            ", maxResidue = ", maxResidue,
            ", maxDiff = ", maxDiff,
            ", noiseNorm2 = ", result.noise_norm2,
            ", pseudofermionNorm2 = ", result.pseudofermion_norm2));
    }

    rootLogger.info("MDWF pseudofermion heatbath mock test passed with Ls = ", Ls,
                    ", terms = ", coefficients.shift.size(),
                    ", maxResidue = ", maxResidue,
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

        runMDWFPseudofermionHeatbathMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
