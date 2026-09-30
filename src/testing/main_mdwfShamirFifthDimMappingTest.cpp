/*
 * MDWF Shamir-kernel physical-parameter mapping smoke test.
 *
 * This checks the (M5, mf) -> (mass, fifth-direction coefficients) mapping
 * proposed in PHYSICAL_OPERATOR_MAPPING.md for the Shamir special case
 * (b5 = 1, c5 = 0). It is a design-stage arithmetic/wiring check, not a
 * gauge-dependent physics validation:
 *
 *   - independently recomputes the mapping formulas from
 *     PHYSICAL_OPERATOR_MAPPING.md and compares them against
 *     MDWFPhysicalMapping.h, to catch a transcription mistake between the
 *     design note and the header;
 *   - checks the documented mf = 1 Pauli-Villars magnitude-symmetry property
 *     directly on the produced coefficients, and that a generic physical-like
 *     mf does not share that property;
 *   - feeds the mapped coefficients through the already-validated
 *     applyMDWFFifthDimCoupling wiring, confirms the result still matches the
 *     fifth-direction stencil oracle used by mdwfFifthDimTest, and checks
 *     that a deliberately wrong boundary sign is detectably different.
 *
 * This does not validate the mapping against gauge-dependent domain-wall
 * physics (chiral zero mode, residual mass, spectral flow); that needs
 * propagator/eigenvalue computations and is deferred. It does not touch
 * clover, CG, RHMC/HMC, or force code, and does not change
 * MDWFFifthDimCoupling, MDWFOperator, or MDWFPhysicalMapping.h.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFPhysicalMapping.h"

#include <cmath>
#include <stdexcept>

namespace {

template<class floatT>
floatT independentShamirKernelMass(floatT M5) {
    // Recomputed directly from PHYSICAL_OPERATOR_MAPPING.md Sections 1.3/2
    // (corrected): mass = m_std + 4 with m_std = -M5.
    const floatT m_std = -M5;
    return m_std + static_cast<floatT>(4.0);
}

template<class floatT>
void checkShamirMappingFormulas(floatT M5, floatT mf, double tolerance) {
    const floatT expectedMass = independentShamirKernelMass<floatT>(M5);
    const floatT expectedDiagonal = static_cast<floatT>(1.0);
    const floatT expectedForwardHop = static_cast<floatT>(-1.0);
    const floatT expectedBackwardHop = static_cast<floatT>(-1.0);
    const floatT expectedForwardBoundary = mf;
    const floatT expectedBackwardBoundary = mf;

    MDWFShamirOperatorParameters<floatT> params(M5, mf);

    const double massDiff = std::abs(params.mass - expectedMass);
    const double diagonalDiff = std::abs(params.fifth_coeff.diagonal - expectedDiagonal);
    const double forwardHopDiff = std::abs(params.fifth_coeff.forward_hop - expectedForwardHop);
    const double backwardHopDiff = std::abs(params.fifth_coeff.backward_hop - expectedBackwardHop);
    const double forwardBoundaryDiff = std::abs(params.fifth_coeff.forward_boundary - expectedForwardBoundary);
    const double backwardBoundaryDiff = std::abs(params.fifth_coeff.backward_boundary - expectedBackwardBoundary);

    if (massDiff > tolerance || diagonalDiff > tolerance || forwardHopDiff > tolerance
        || backwardHopDiff > tolerance || forwardBoundaryDiff > tolerance
        || backwardBoundaryDiff > tolerance) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Shamir mapping formula check failed for M5 = ", M5, ", mf = ", mf,
            ": massDiff = ", massDiff, ", diagonalDiff = ", diagonalDiff,
            ", forwardHopDiff = ", forwardHopDiff, ", backwardHopDiff = ", backwardHopDiff,
            ", forwardBoundaryDiff = ", forwardBoundaryDiff,
            ", backwardBoundaryDiff = ", backwardBoundaryDiff));
    }
}

template<class floatT>
void checkShamirPauliVillarsEndpoint(double tolerance) {
    // mf = 1 is the reference PV endpoint (PHYSICAL_OPERATOR_MAPPING.md,
    // Section 2.1): the wraparound coefficients then match the interior hop
    // magnitude (both 1), unlike a generic physical-like mf, removing the
    // asymmetry that otherwise distinguishes the wall from the bulk.
    MDWFShamirOperatorParameters<floatT> pv(static_cast<floatT>(1.8), static_cast<floatT>(1.0));

    const double interiorMagnitude = std::abs(pv.fifth_coeff.forward_hop);
    const double pvBoundaryMagnitude = std::abs(pv.fifth_coeff.forward_boundary);
    const double pvMagnitudeDiff = std::abs(interiorMagnitude - pvBoundaryMagnitude);

    if (pvMagnitudeDiff > tolerance) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Shamir PV endpoint magnitude check failed: interiorMagnitude = ",
            interiorMagnitude, ", pvBoundaryMagnitude = ", pvBoundaryMagnitude));
    }

    MDWFShamirOperatorParameters<floatT> physicalLike(static_cast<floatT>(1.8), static_cast<floatT>(0.05));
    const double physicalBoundaryMagnitude = std::abs(physicalLike.fifth_coeff.forward_boundary);
    if (std::abs(physicalBoundaryMagnitude - interiorMagnitude) <= tolerance) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Shamir PV endpoint check found no magnitude asymmetry away from mf = 1: ",
            "physicalBoundaryMagnitude = ", physicalBoundaryMagnitude));
    }
}

} // namespace

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFShamirMappingPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t i = 0; i < 12; i++) {
            out.data[i] = COMPLEX(floatT)(static_cast<floatT>(100 * site.stack + i + 1), 0.0);
        }
        return out;
    }
};

template<size_t Ls>
double expectedMDWFShamirFifthDimValue(size_t stack,
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
void runMDWFShamirFifthDimMappingTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    const double M5 = 1.8;
    const double mf = 0.05;

    checkShamirMappingFormulas<double>(M5, mf, 1e-12);
    checkShamirMappingFormulas<double>(1.0, 0.2, 1e-12);
    checkShamirMappingFormulas<double>(0.5, 1.0, 1e-12);
    checkShamirPauliVillarsEndpoint<double>(1e-12);

    MDWFShamirOperatorParameters<double> params(M5, mf);
    const MDWFFifthDimCoefficients<double> coeff = params.fifth_coeff;

    // Deliberately wrong boundary sign, to check that the wiring comparison
    // below is sensitive to a sign mistake in the physical mapping.
    const MDWFFifthDimCoefficients<double> wrongCoeff(
        coeff.diagonal, coeff.forward_hop, coeff.backward_hop, -mf, -mf);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_shamir_mapping_in");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorOut(commBase, "MDWF_shamir_mapping_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorOutWrong(commBase, "MDWF_shamir_mapping_out_wrong");
    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorOutHost(commBase, "MDWF_shamir_mapping_out_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorOutWrongHost(commBase, "MDWF_shamir_mapping_out_wrong_host");

    spinorIn.template iterateOverBulk<>(FillMDWFShamirMappingPattern<double, All, HaloDepth, Ls>());

    applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(spinorOut, spinorIn, coeff);
    applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(spinorOutWrong, spinorIn, wrongCoeff);

    spinorOutHost = spinorOut;
    spinorOutWrongHost = spinorOutWrong;

    Vect12ArrayAcc<double> outAcc = spinorOutHost.getAccessor();
    Vect12ArrayAcc<double> outWrongAcc = spinorOutWrongHost.getAccessor();

    double maxOracleDiff = 0.0;
    double maxWrongOracleDiff = 0.0;
    double maxDetectionDiff = 0.0;

    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> out = outAcc.getElement(site);
            Vect12<double> outWrong = outWrongAcc.getElement(site);

            for (size_t component = 0; component < 12; component++) {
                const double expected = expectedMDWFShamirFifthDimValue<Ls>(stack, component, coeff);
                const double expectedWrong = expectedMDWFShamirFifthDimValue<Ls>(stack, component, wrongCoeff);

                const double oracleDiff = std::abs(real(out.data[component]) - expected)
                                          + std::abs(imag(out.data[component]));
                const double wrongOracleDiff = std::abs(real(outWrong.data[component]) - expectedWrong)
                                               + std::abs(imag(outWrong.data[component]));
                const double detectionDiff = std::abs(real(out.data[component]) - real(outWrong.data[component]))
                                             + std::abs(imag(out.data[component]) - imag(outWrong.data[component]));

                if (oracleDiff > maxOracleDiff) {
                    maxOracleDiff = oracleDiff;
                }
                if (wrongOracleDiff > maxWrongOracleDiff) {
                    maxWrongOracleDiff = wrongOracleDiff;
                }
                if (detectionDiff > maxDetectionDiff) {
                    maxDetectionDiff = detectionDiff;
                }
            }
        }

    if (maxOracleDiff > 1e-12 || maxWrongOracleDiff > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Shamir mapping wiring check failed: maxOracleDiff = ", maxOracleDiff,
            ", maxWrongOracleDiff = ", maxWrongOracleDiff));
    }

    if (maxDetectionDiff <= 1e-6) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Shamir mapping sign-sensitivity check found no detectable difference "
            "from a wrong boundary sign: maxDetectionDiff = ", maxDetectionDiff));
    }

    rootLogger.info("MDWF Shamir physical-parameter mapping test passed with Ls = ", Ls,
                    ", M5 = ", M5, ", mf = ", mf, ", mass = ", params.mass,
                    ", maxOracleDiff = ", maxOracleDiff,
                    ", maxWrongOracleDiff = ", maxWrongOracleDiff,
                    ", maxDetectionDiff (wrong boundary sign) = ", maxDetectionDiff);
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

        runMDWFShamirFifthDimMappingTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
