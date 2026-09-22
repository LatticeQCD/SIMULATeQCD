/*
 * MDWF general-Mobius (RBC/UKQCD convention) forward-operator composition
 * smoke test.
 *
 * This validates the composition proposed in PHYSICAL_OPERATOR_MAPPING.md
 * Section 2.3/5, built entirely from existing, unmodified pieces
 * (applyMDWFFifthDimCoupling and applyMDWFWilsonSlice): no new
 * gauge-dependent fifth-direction coupling class is introduced.
 *
 * Checks, on a fixed nontrivial (random) gauge field at c_sw = 0:
 *
 *   1. At b5 = 1 (so c5 = 0 under the project's b5 - c5 = 1 convention),
 *      the Din construction reduces exactly to the input spinor, and the
 *      full Mobius composition reproduces the already-validated
 *      applyMDWFOperator fed the Shamir mapping bit-for-bit. This is the
 *      exact-regression check promised when the Shamir path was first
 *      implemented.
 *   2. At a generic b5 (b5 = 1.5, c5 = 0.5), the output is detectably
 *      different from the b5 = 1 baseline, confirming b5/c5 are actually
 *      wired into the computation.
 *
 * This does not implement zMobius, clover for the Mobius path, the adjoint
 * of the general Mobius operator, or a production M5/mf/b5 choice. It does
 * not validate the mapping against gauge-dependent domain-wall physics
 * (chiral zero mode, residual mass); that needs propagator/eigenvalue
 * computations and is deferred. It does not touch CG, RHMC/HMC, or force
 * code, and does not change MDWFFifthDimCoupling, MDWFWilsonSlice,
 * MDWFOperator, or MDWFPhysicalMapping.h.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFOperator.h"

#include <cmath>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusMappingPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t i = 0; i < 12; i++) {
            out.data[i] = COMPLEX(floatT)(
                static_cast<floatT>(0.3) + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + i + 1),
                static_cast<floatT>(0.01) * static_cast<floatT>(i + 1)
                - static_cast<floatT>(0.002) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFMobiusFifthDimMappingTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    const double M5 = 1.8;
    const double mf = 0.05;

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_mobius_mapping_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(918273645);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    MDWFSpinor<double, true, All, HaloDepth, Ls> spinorIn(commBase, "MDWF_mobius_mapping_in");
    spinorIn.template iterateOverBulk<>(FillMDWFMobiusMappingPattern<double, All, HaloDepth, Ls>());
    spinorIn.updateAll();

    // Shamir baseline (b5 = 1, c5 = 0), via the already-validated
    // applyMDWFOperator path fed the Shamir mapping directly.
    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double mass = mdwfShamirKernelMass(M5);

    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonPart(commBase, "MDWF_mobius_mapping_wilson_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> fifthPart(commBase, "MDWF_mobius_mapping_fifth_part");
    MDWFSpinor<double, true, All, HaloDepth, Ls> wilsonTmp(commBase, "MDWF_mobius_mapping_wilson_tmp");
    MDWFSpinor<double, true, All, HaloDepth, Ls> shamirOut(commBase, "MDWF_mobius_mapping_shamir_out");

    applyMDWFOperator<double, HaloDepth, HaloDepth, Ls>(
        shamirOut, gauge, wilsonPart, fifthPart, wilsonTmp, spinorIn, shamirCoeff, mass, 0.0, true);

    // General Mobius composition at b5 = 1 (c5 = 0): must reduce exactly.
    MDWFMobiusOperatorParameters<double> paramsB5One(M5, mf, 1.0);
    MDWFMobiusOperatorWorkspace<double, HaloDepth, HaloDepth, Ls> mobiusWorkspaceB5One(
        gauge, "MDWF_mobius_mapping_b5_one");
    MDWFSpinor<double, true, All, HaloDepth, Ls> mobiusOutB5One(commBase, "MDWF_mobius_mapping_out_b5_one");
    mobiusWorkspaceB5One.apply(mobiusOutB5One, spinorIn, paramsB5One, true);

    // Din at b5 = 1, c5 = 0 should reduce exactly to spinorIn.
    MDWFSpinor<double, true, All, HaloDepth, Ls> dinB5One(commBase, "MDWF_mobius_mapping_din_b5_one");
    applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(
        dinB5One, spinorIn, paramsB5One.dinCoeff, true);

    // General Mobius at a generic b5 (b5 = 1.5, c5 = 0.5 under b5 - c5 = 1).
    MDWFMobiusOperatorParameters<double> paramsGeneric(M5, mf, 1.5);
    MDWFMobiusOperatorWorkspace<double, HaloDepth, HaloDepth, Ls> mobiusWorkspaceGeneric(
        gauge, "MDWF_mobius_mapping_generic");
    MDWFSpinor<double, true, All, HaloDepth, Ls> mobiusOutGeneric(commBase, "MDWF_mobius_mapping_out_generic");
    mobiusWorkspaceGeneric.apply(mobiusOutGeneric, spinorIn, paramsGeneric, true);

    MDWFSpinor<double, false, All, HaloDepth, Ls> spinorInHost(commBase, "MDWF_mobius_mapping_in_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> dinB5OneHost(commBase, "MDWF_mobius_mapping_din_b5_one_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> shamirOutHost(commBase, "MDWF_mobius_mapping_shamir_out_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> mobiusOutB5OneHost(
        commBase, "MDWF_mobius_mapping_out_b5_one_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> mobiusOutGenericHost(
        commBase, "MDWF_mobius_mapping_out_generic_host");

    spinorInHost = spinorIn;
    dinB5OneHost = dinB5One;
    shamirOutHost = shamirOut;
    mobiusOutB5OneHost = mobiusOutB5One;
    mobiusOutGenericHost = mobiusOutGeneric;

    Vect12ArrayAcc<double> inAcc = spinorInHost.getAccessor();
    Vect12ArrayAcc<double> dinB5OneAcc = dinB5OneHost.getAccessor();
    Vect12ArrayAcc<double> shamirAcc = shamirOutHost.getAccessor();
    Vect12ArrayAcc<double> mobiusB5OneAcc = mobiusOutB5OneHost.getAccessor();
    Vect12ArrayAcc<double> mobiusGenericAcc = mobiusOutGenericHost.getAccessor();

    double maxDinDiff = 0.0;
    double maxShamirRegressionDiff = 0.0;
    double maxGenericDetectionDiff = 0.0;

    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            Vect12<double> inVal = inAcc.getElement(site);
            Vect12<double> dinVal = dinB5OneAcc.getElement(site);
            Vect12<double> shamirVal = shamirAcc.getElement(site);
            Vect12<double> mobiusB5OneVal = mobiusB5OneAcc.getElement(site);
            Vect12<double> mobiusGenericVal = mobiusGenericAcc.getElement(site);

            for (size_t component = 0; component < 12; component++) {
                const double dinDiff = std::abs(real(dinVal.data[component]) - real(inVal.data[component]))
                                       + std::abs(imag(dinVal.data[component]) - imag(inVal.data[component]));
                const double shamirRegressionDiff =
                    std::abs(real(mobiusB5OneVal.data[component]) - real(shamirVal.data[component]))
                    + std::abs(imag(mobiusB5OneVal.data[component]) - imag(shamirVal.data[component]));
                const double genericDetectionDiff =
                    std::abs(real(mobiusGenericVal.data[component]) - real(mobiusB5OneVal.data[component]))
                    + std::abs(imag(mobiusGenericVal.data[component]) - imag(mobiusB5OneVal.data[component]));

                if (dinDiff > maxDinDiff) {
                    maxDinDiff = dinDiff;
                }
                if (shamirRegressionDiff > maxShamirRegressionDiff) {
                    maxShamirRegressionDiff = shamirRegressionDiff;
                }
                if (genericDetectionDiff > maxGenericDetectionDiff) {
                    maxGenericDetectionDiff = genericDetectionDiff;
                }
            }
        }

    if (maxDinDiff > 1e-12 || maxShamirRegressionDiff > 1e-10) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius mapping b5 = 1 regression check failed: maxDinDiff = ", maxDinDiff,
            ", maxShamirRegressionDiff = ", maxShamirRegressionDiff));
    }

    if (maxGenericDetectionDiff <= 1e-6) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius mapping found no detectable difference at generic b5: "
            "maxGenericDetectionDiff = ", maxGenericDetectionDiff));
    }

    rootLogger.info("MDWF Mobius fifth-direction mapping test passed with Ls = ", Ls,
                    ", M5 = ", M5, ", mf = ", mf, ", mass = ", mass,
                    ", maxDinDiff (b5 = 1 Din == psi) = ", maxDinDiff,
                    ", maxShamirRegressionDiff (b5 = 1 vs Shamir) = ", maxShamirRegressionDiff,
                    ", maxGenericDetectionDiff (b5 = 1.5 vs b5 = 1) = ", maxGenericDetectionDiff);
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

        runMDWFMobiusFifthDimMappingTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
