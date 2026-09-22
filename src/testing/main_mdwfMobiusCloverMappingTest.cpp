/*
 * MDWF general-Mobius (RBC/UKQCD convention) clover extension smoke test.
 *
 * This validates MDWFMobiusCloverLinearOperator / MDWFMobiusCloverAdjointLinearOperator
 * from MDWFMobiusMapping.h. Per the project's clover guardrail, clover is routed
 * only through the existing Wilson-kernel path (applyMDWFCloverWilsonSlice); Din,
 * the fifth-direction shift term, and the gamma5-Hermiticity adjoint composition
 * are unchanged from the already-validated c_sw = 0 Mobius classes.
 *
 * Checks, on a fixed nontrivial (random) gauge field:
 *
 *   1. c_sw = 0 regression at a generic b5 (1.5): the clover-capable forward and
 *      adjoint Mobius operators reproduce the existing, already-validated
 *      c_sw = 0 Mobius classes (MDWFMobiusLinearOperator /
 *      MDWFMobiusAdjointLinearOperator) exactly.
 *   2. b5 = 1 exact regression against the existing Shamir clover path, at
 *      c_sw = 0.5: since Din/Din^dagger reduce to the identity at b5 = 1, the
 *      Mobius clover forward/adjoint operators must reproduce the plain
 *      Shamir clover operators (MDWFLinearOperator / MDWFAdjointLinearOperator,
 *      which already route every apply() through the clover-capable path)
 *      exactly.
 *   3. Nonzero-c_sw sanity at a generic b5 (1.5): the clover forward operator
 *      at c_sw = 0.5 gives a detectably nonzero difference from c_sw = 0.
 *   4. Generic-b5 coupled-5D adjoint identity <x, M y> = <M^dagger y> at
 *      c_sw = 0.5, using the same aggregated dot product already validated
 *      for the Shamir/normal-operator and c_sw = 0 Mobius adjoint scaffolds.
 *
 * This does not validate the mapping against gauge-dependent domain-wall
 * physics, does not implement zMobius, and does not touch CG, RHMC/HMC, or
 * force code.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusCloverPattern {
    floatT offset;

    explicit FillMDWFMobiusCloverPattern(floatT offset_in) : offset(offset_in) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t i = 0; i < 12; i++) {
            out.data[i] = COMPLEX(floatT)(
                offset + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + i + 1),
                static_cast<floatT>(0.013) * static_cast<floatT>(i + 1)
                - static_cast<floatT>(0.0021) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
double maxSpinorDiff(CommunicationBase &commBase,
                     MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &a,
                     MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &b) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    MDWFSpinor<floatT, false, LatLayout, HaloDepth, Ls> aHost(commBase, "MDWF_mobius_clover_diff_a_host");
    MDWFSpinor<floatT, false, LatLayout, HaloDepth, Ls> bHost(commBase, "MDWF_mobius_clover_diff_b_host");
    aHost = a;
    bHost = b;

    Vect12ArrayAcc<floatT> aAcc = aHost.getAccessor();
    Vect12ArrayAcc<floatT> bAcc = bHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<floatT> valA = aAcc.getElement(site);
            Vect12<floatT> valB = bAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(valA.data[component]) - real(valB.data[component]))
                                    + std::abs(imag(valA.data[component]) - imag(valB.data[component]));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
    return maxDiff;
}

template<size_t Ls>
void runMDWFMobiusCloverMappingTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;

    const double M5 = 1.8;
    const double mf = 0.05;
    const double csw = 0.5;
    const double genericB5 = 1.5;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_mobius_clover_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(24681357);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_mobius_clover_in");
    spinorIn.template iterateOverBulk<>(FillMDWFMobiusCloverPattern<double, All, HaloDepth, Ls>(0.2));
    spinorIn.updateAll();

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double mass = mdwfShamirKernelMass(M5);

    // --- Part 1: c_sw = 0 regression against the existing plain Mobius classes, generic b5. ---

    MDWFMobiusLinearOperator<double, HaloDepth, HaloDepth, Ls> plainForwardGeneric(
        gauge, M5, mf, genericB5, "MDWF_mobius_clover_plain_forward_generic");
    MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls> cloverForwardGenericCswZero(
        gauge, M5, mf, genericB5, 0.0, "MDWF_mobius_clover_clover_forward_generic_csw0");

    MDWFSpinor<double, true, All, HaloDepth, Ls> plainForwardOut(commBase, "MDWF_mobius_clover_plain_forward_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cloverForwardCswZeroOut(
        commBase, "MDWF_mobius_clover_clover_forward_csw0_out");
    plainForwardGeneric.apply(plainForwardOut, spinorIn, true);
    cloverForwardGenericCswZero.apply(cloverForwardCswZeroOut, spinorIn, true);

    const double forwardCswZeroRegressionDiff =
        maxSpinorDiff<double, All, HaloDepth, Ls>(commBase, plainForwardOut, cloverForwardCswZeroOut);

    MDWFMobiusAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls> plainAdjointGeneric(
        gauge, M5, mf, genericB5, "MDWF_mobius_clover_plain_adjoint_generic");
    MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls> cloverAdjointGenericCswZero(
        gauge, M5, mf, genericB5, 0.0, "MDWF_mobius_clover_clover_adjoint_generic_csw0");

    MDWFSpinor<double, true, All, HaloDepth, Ls> plainAdjointOut(commBase, "MDWF_mobius_clover_plain_adjoint_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cloverAdjointCswZeroOut(
        commBase, "MDWF_mobius_clover_clover_adjoint_csw0_out");
    plainAdjointGeneric.apply(plainAdjointOut, spinorIn, true);
    cloverAdjointGenericCswZero.apply(cloverAdjointCswZeroOut, spinorIn, true);

    const double adjointCswZeroRegressionDiff =
        maxSpinorDiff<double, All, HaloDepth, Ls>(commBase, plainAdjointOut, cloverAdjointCswZeroOut);

    // --- Part 2: b5 = 1 exact regression against the existing Shamir clover path, c_sw = 0.5. ---

    MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls> shamirCloverForward(
        gauge, shamirCoeff, mass, csw, "MDWF_mobius_clover_shamir_clover_forward");
    MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls> mobiusCloverForwardB5One(
        gauge, M5, mf, 1.0, csw, "MDWF_mobius_clover_mobius_clover_forward_b5_one");

    MDWFSpinor<double, true, All, HaloDepth, Ls> shamirCloverForwardOut(
        commBase, "MDWF_mobius_clover_shamir_clover_forward_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> mobiusCloverForwardB5OneOut(
        commBase, "MDWF_mobius_clover_mobius_clover_forward_b5_one_out");
    shamirCloverForward.apply(shamirCloverForwardOut, spinorIn, true);
    mobiusCloverForwardB5One.apply(mobiusCloverForwardB5OneOut, spinorIn, true);

    const double forwardShamirRegressionDiff = maxSpinorDiff<double, All, HaloDepth, Ls>(
        commBase, shamirCloverForwardOut, mobiusCloverForwardB5OneOut);

    MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls> shamirCloverAdjoint(
        gauge, shamirCoeff, mass, csw, "MDWF_mobius_clover_shamir_clover_adjoint");
    MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls> mobiusCloverAdjointB5One(
        gauge, M5, mf, 1.0, csw, "MDWF_mobius_clover_mobius_clover_adjoint_b5_one");

    MDWFSpinor<double, true, All, HaloDepth, Ls> shamirCloverAdjointOut(
        commBase, "MDWF_mobius_clover_shamir_clover_adjoint_out");
    MDWFSpinor<double, true, All, HaloDepth, Ls> mobiusCloverAdjointB5OneOut(
        commBase, "MDWF_mobius_clover_mobius_clover_adjoint_b5_one_out");
    shamirCloverAdjoint.apply(shamirCloverAdjointOut, spinorIn, true);
    mobiusCloverAdjointB5One.apply(mobiusCloverAdjointB5OneOut, spinorIn, true);

    const double adjointShamirRegressionDiff = maxSpinorDiff<double, All, HaloDepth, Ls>(
        commBase, shamirCloverAdjointOut, mobiusCloverAdjointB5OneOut);

    // --- Part 3: nonzero-c_sw sanity at generic b5: detectable response vs c_sw = 0. ---

    MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls> cloverForwardGenericCswNonzero(
        gauge, M5, mf, genericB5, csw, "MDWF_mobius_clover_clover_forward_generic_csw_nonzero");
    MDWFSpinor<double, true, All, HaloDepth, Ls> cloverForwardCswNonzeroOut(
        commBase, "MDWF_mobius_clover_clover_forward_csw_nonzero_out");
    cloverForwardGenericCswNonzero.apply(cloverForwardCswNonzeroOut, spinorIn, true);

    const double cswResponseDiff = maxSpinorDiff<double, All, HaloDepth, Ls>(
        commBase, cloverForwardCswZeroOut, cloverForwardCswNonzeroOut);

    // --- Part 4: generic-b5 coupled-5D adjoint identity <x, M y> = <M^dagger x, y>, c_sw = 0.5. ---

    using ForwardOperator = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;

    ForwardOperator forwardGenericCswNonzero(
        gauge, M5, mf, genericB5, csw, "MDWF_mobius_clover_adjoint_id_forward");
    AdjointOperator adjointGenericCswNonzero(
        gauge, M5, mf, genericB5, csw, "MDWF_mobius_clover_adjoint_id_adjoint");
    ForwardAdapter forwardAdapter(forwardGenericCswNonzero);

    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, "MDWF_mobius_clover_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, "MDWF_mobius_clover_probe_y");
    probeX.template iterateOverBulk<>(FillMDWFMobiusCloverPattern<double, All, HaloDepth, Ls>(0.7));
    probeY.template iterateOverBulk<>(FillMDWFMobiusCloverPattern<double, All, HaloDepth, Ls>(1.3));
    probeX.updateAll();
    probeY.updateAll();

    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, "MDWF_mobius_clover_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, "MDWF_mobius_clover_adjoint_x");
    forwardGenericCswNonzero.apply(forwardY, probeY, true);
    adjointGenericCswNonzero.apply(adjointX, probeX, true);

    const COMPLEX(double) left = forwardAdapter.dotProduct5D(probeX, forwardY);
    const COMPLEX(double) right = forwardAdapter.dotProduct5D(adjointX, probeY);
    const double adjointDiff = std::abs(real(left - right)) + std::abs(imag(left - right));
    const double adjointScale = std::max(1.0, std::max(std::abs(real(left)) + std::abs(imag(left)),
                                                       std::abs(real(right)) + std::abs(imag(right))));
    const double adjointRelDiff = adjointDiff / adjointScale;

    if (forwardCswZeroRegressionDiff > 1e-10 || adjointCswZeroRegressionDiff > 1e-10
        || forwardShamirRegressionDiff > 1e-10 || adjointShamirRegressionDiff > 1e-10
        || cswResponseDiff <= 1e-12 || adjointRelDiff > 1e-9) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius clover mapping test failed: forwardCswZeroRegressionDiff = ",
            forwardCswZeroRegressionDiff, ", adjointCswZeroRegressionDiff = ", adjointCswZeroRegressionDiff,
            ", forwardShamirRegressionDiff = ", forwardShamirRegressionDiff,
            ", adjointShamirRegressionDiff = ", adjointShamirRegressionDiff,
            ", cswResponseDiff = ", cswResponseDiff, ", adjointRelDiff = ", adjointRelDiff));
    }

    rootLogger.info("MDWF Mobius clover mapping test passed with Ls = ", Ls,
                    ", M5 = ", M5, ", mf = ", mf, ", mass = ", mass, ", c_sw = ", csw,
                    ", forwardCswZeroRegressionDiff (generic b5 vs plain Mobius) = ", forwardCswZeroRegressionDiff,
                    ", adjointCswZeroRegressionDiff (generic b5 vs plain Mobius adjoint) = ",
                    adjointCswZeroRegressionDiff,
                    ", forwardShamirRegressionDiff (b5 = 1 vs Shamir clover) = ", forwardShamirRegressionDiff,
                    ", adjointShamirRegressionDiff (b5 = 1 vs Shamir clover adjoint) = ",
                    adjointShamirRegressionDiff,
                    ", cswResponseDiff (generic b5, csw 0 vs 0.5) = ", cswResponseDiff,
                    ", adjointRelDiff (b5 = 1.5 coupled-5D identity) = ", adjointRelDiff);
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

        runMDWFMobiusCloverMappingTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
