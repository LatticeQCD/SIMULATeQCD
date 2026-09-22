/*
 * MDWF general-Mobius (RBC/UKQCD convention) adjoint-operator smoke test.
 *
 * This validates MDWFMobiusAdjointLinearOperator / applyMDWFMobiusAdjointOperator
 * from MDWFMobiusMapping.h, built entirely from existing, already-validated
 * pieces: MDWFFifthDimAdjointCoupling (the formal transpose-adjoint of
 * MDWFFifthDimCoupling for any five coefficients) and the standard Wilson
 * gamma5-Hermiticity D_W^dagger = gamma5 D_W gamma5 already used by
 * MDWFAdjointOperator.h. No new adjoint machinery is introduced.
 *
 * Checks, on a fixed nontrivial (random) gauge field at c_sw = 0:
 *
 *   1. At b5 = 1 (c5 = 0), the general-Mobius adjoint reproduces the
 *      already-validated Shamir adjoint (MDWFAdjointLinearOperator) exactly,
 *      the adjoint counterpart of the forward-operator regression check in
 *      mdwfMobiusFifthDimMappingTest.
 *   2. At a generic b5 (b5 = 1.5, c5 = 0.5), the coupled-5D adjoint identity
 *      <x, M y> = <M^dagger x, y> holds, using the same aggregated dot
 *      product already validated for the Shamir/normal-operator scaffold.
 *
 * This does not validate the mapping against gauge-dependent domain-wall
 * physics, does not implement zMobius or clover for the Mobius path, and
 * does not touch CG, RHMC/HMC, or force code.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusAdjointPattern {
    floatT offset;

    explicit FillMDWFMobiusAdjointPattern(floatT offset_in) : offset(offset_in) {}

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

template<size_t Ls>
void runMDWFMobiusAdjointMappingTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    const double M5 = 1.8;
    const double mf = 0.05;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_mobius_adjoint_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(837465921);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_mobius_adjoint_in");
    spinorIn.template iterateOverBulk<>(FillMDWFMobiusAdjointPattern<double, All, HaloDepth, Ls>(0.2));
    spinorIn.updateAll();

    // --- Part 1: b5 = 1 exact regression against the existing Shamir adjoint. ---

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double mass = mdwfShamirKernelMass(M5);

    using ShamirAdjoint = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    ShamirAdjoint shamirAdjoint(gauge, shamirCoeff, mass, 0.0, "MDWF_mobius_adjoint_shamir");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shamirAdjointOut(commBase, "MDWF_mobius_adjoint_shamir_out");
    shamirAdjoint.apply(shamirAdjointOut, spinorIn, true);

    MDWFMobiusAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls> mobiusAdjointB5One(
        gauge, M5, mf, 1.0, "MDWF_mobius_adjoint_b5_one");
    MDWFSpinor<double, true, All, HaloDepth, Ls> mobiusAdjointB5OneOut(
        commBase, "MDWF_mobius_adjoint_b5_one_out");
    mobiusAdjointB5One.apply(mobiusAdjointB5OneOut, spinorIn, true);

    MDWFSpinor<double, false, All, HaloDepth, Ls> shamirAdjointOutHost(
        commBase, "MDWF_mobius_adjoint_shamir_out_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> mobiusAdjointB5OneOutHost(
        commBase, "MDWF_mobius_adjoint_b5_one_out_host");
    shamirAdjointOutHost = shamirAdjointOut;
    mobiusAdjointB5OneOutHost = mobiusAdjointB5OneOut;

    Vect12ArrayAcc<double> shamirAdjointAcc = shamirAdjointOutHost.getAccessor();
    Vect12ArrayAcc<double> mobiusAdjointB5OneAcc = mobiusAdjointB5OneOutHost.getAccessor();

    double maxAdjointRegressionDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> shamirVal = shamirAdjointAcc.getElement(site);
            Vect12<double> mobiusVal = mobiusAdjointB5OneAcc.getElement(site);
            for (size_t component = 0; component < 12; component++) {
                const double diff = std::abs(real(mobiusVal.data[component]) - real(shamirVal.data[component]))
                                    + std::abs(imag(mobiusVal.data[component]) - imag(shamirVal.data[component]));
                if (diff > maxAdjointRegressionDiff) {
                    maxAdjointRegressionDiff = diff;
                }
            }
        }

    // --- Part 2: generic-b5 coupled-5D adjoint identity <x, M y> = <M^dagger x, y>. ---

    using ForwardOperator = MDWFMobiusLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFMobiusAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ForwardAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ForwardOperator>;

    const double genericB5 = 1.5;
    ForwardOperator forwardGeneric(gauge, M5, mf, genericB5, "MDWF_mobius_adjoint_forward_generic");
    AdjointOperator adjointGeneric(gauge, M5, mf, genericB5, "MDWF_mobius_adjoint_adjoint_generic");
    ForwardAdapter forwardAdapter(forwardGeneric);

    MDWFSpinor<double, true, All, HaloDepth, Ls> probeX(commBase, "MDWF_mobius_adjoint_probe_x");
    MDWFSpinor<double, true, All, HaloDepth, Ls> probeY(commBase, "MDWF_mobius_adjoint_probe_y");
    probeX.template iterateOverBulk<>(FillMDWFMobiusAdjointPattern<double, All, HaloDepth, Ls>(0.7));
    probeY.template iterateOverBulk<>(FillMDWFMobiusAdjointPattern<double, All, HaloDepth, Ls>(1.3));
    probeX.updateAll();
    probeY.updateAll();

    MDWFSpinor<double, true, All, HaloDepth, Ls> forwardY(commBase, "MDWF_mobius_adjoint_forward_y");
    MDWFSpinor<double, true, All, HaloDepth, Ls> adjointX(commBase, "MDWF_mobius_adjoint_adjoint_x");
    forwardGeneric.apply(forwardY, probeY, true);
    adjointGeneric.apply(adjointX, probeX, true);

    const COMPLEX(double) left = forwardAdapter.dotProduct5D(probeX, forwardY);
    const COMPLEX(double) right = forwardAdapter.dotProduct5D(adjointX, probeY);
    const double adjointDiff = std::abs(real(left - right)) + std::abs(imag(left - right));
    const double adjointScale = std::max(1.0, std::max(std::abs(real(left)) + std::abs(imag(left)),
                                                       std::abs(real(right)) + std::abs(imag(right))));
    const double adjointRelDiff = adjointDiff / adjointScale;

    if (maxAdjointRegressionDiff > 1e-10 || adjointRelDiff > 1e-9) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius adjoint mapping test failed: maxAdjointRegressionDiff = ",
            maxAdjointRegressionDiff, ", adjointRelDiff (generic b5) = ", adjointRelDiff));
    }

    rootLogger.info("MDWF Mobius adjoint mapping test passed with Ls = ", Ls,
                    ", M5 = ", M5, ", mf = ", mf, ", mass = ", mass,
                    ", maxAdjointRegressionDiff (b5 = 1 vs Shamir adjoint) = ", maxAdjointRegressionDiff,
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

        runMDWFMobiusAdjointMappingTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
