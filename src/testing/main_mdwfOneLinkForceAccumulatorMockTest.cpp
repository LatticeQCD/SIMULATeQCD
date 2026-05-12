/*
 * MDWF one-link force-accumulator mock test.
 *
 * This validates the first accumulator boundary after the scalar contraction
 * tests: write exactly one selected gauge-force-like link, then contract that
 * stored link with the existing analytic single-link helper.  It does not
 * implement production force accumulation, projection, momentum updates,
 * RHMC/HMC wiring, HISQ reuse, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFOneLinkForceAccumulatorMock.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<size_t HaloDepth>
MDWFOneLinkForceAccumulatorMockResult<double> runMDWFOneLinkForceAccumulatorMockCase(
    Gaugefield<double, true, HaloDepth, R18> &force,
    const MDWFFiniteDifferenceProbe<double> &probe,
    const SU3<double> &baseLink,
    const SU3<double> &linearActionMatrix,
    CommunicationBase &commBase,
    const std::string &name) {

    const SU3<double> selectedLinkActionDerivative
        = mdwfSingleLinkLinearActionDerivativeMatrix(
            baseLink, linearActionMatrix, probe.multiplication_side);
    const double expectedDerivative = mdwfContractSingleLinkActionDerivative(
        selectedLinkActionDerivative, probe);

    writeMDWFOneLinkForceAccumulatorMock(force, probe, selectedLinkActionDerivative);
    MDWFOneLinkForceAccumulatorMockResult<double> result
        = inspectMDWFOneLinkForceAccumulatorMock(
            force, probe, selectedLinkActionDerivative, commBase, name);

    const double absoluteDifference = std::abs(result.contracted_derivative
                                               - expectedDerivative);
    const double scale = std::max(1.0, std::abs(expectedDerivative));
    const double relativeDifference = absoluteDifference / scale;

    if (result.selected_link_count != 1
        || result.selected_link_difference > 1e-12
        || result.max_off_probe_norm > 1e-12
        || absoluteDifference > 1e-12
        || relativeDifference > 1e-12
        || !std::isfinite(result.contracted_derivative)) {
        const char *side = probe.multiplication_side
                           == MDWFFiniteDifferenceMultiplicationSide::Left ? "left" : "right";
        throw std::runtime_error(stdLogger.fatal(
            "MDWF one-link force-accumulator mock failed for ", name,
            ": x = ", probe.x,
            ", y = ", probe.y,
            ", z = ", probe.z,
            ", t = ", probe.t,
            ", mu = ", static_cast<int>(probe.mu),
            ", generator = ", probe.generator_id,
            ", side = ", side,
            ", expectedDerivative = ", expectedDerivative,
            ", accumulatedDerivative = ", result.contracted_derivative,
            ", absDiff = ", absoluteDifference,
            ", relDiff = ", relativeDifference,
            ", selectedLinkDifference = ", result.selected_link_difference,
            ", maxOffProbeNorm = ", result.max_off_probe_norm,
            ", selectedLinkCount = ", result.selected_link_count));
    }

    return result;
}

template<size_t Ls>
void runMDWFOneLinkForceAccumulatorMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using ForceField = Gaugefield<double, true, HaloDepth, R18>;

    ForceField force(commBase, "MDWF_one_link_force_accumulator_mock_force");

    const COMPLEX(double) zero(0.0, 0.0);
    const SU3<double> baseLinkLeft(
        COMPLEX(double)(0.92, 0.04), COMPLEX(double)(-0.11, 0.07), zero,
        COMPLEX(double)(0.05, -0.03), COMPLEX(double)(0.87, 0.02), COMPLEX(double)(0.09, -0.08),
        zero, COMPLEX(double)(-0.06, 0.05), COMPLEX(double)(1.03, -0.01));
    const SU3<double> baseLinkRight(
        COMPLEX(double)(1.01, -0.02), zero, COMPLEX(double)(0.08, 0.05),
        COMPLEX(double)(-0.04, 0.06), COMPLEX(double)(0.94, 0.03), COMPLEX(double)(0.07, -0.01),
        COMPLEX(double)(0.03, -0.09), COMPLEX(double)(-0.05, 0.02), COMPLEX(double)(0.89, 0.04));
    const SU3<double> actionMatrixLeft(
        COMPLEX(double)(0.11, 0.03), COMPLEX(double)(-0.07, 0.02), zero,
        COMPLEX(double)(0.05, -0.04), COMPLEX(double)(-0.13, 0.01), COMPLEX(double)(0.09, 0.06),
        zero, COMPLEX(double)(-0.02, 0.08), COMPLEX(double)(0.04, -0.05));
    const SU3<double> actionMatrixRight(
        COMPLEX(double)(-0.06, 0.05), zero, COMPLEX(double)(0.12, -0.01),
        COMPLEX(double)(0.03, 0.07), COMPLEX(double)(0.08, -0.02), COMPLEX(double)(-0.04, 0.09),
        COMPLEX(double)(-0.11, -0.03), COMPLEX(double)(0.06, 0.04), COMPLEX(double)(0.02, 0.01));

    const MDWFFiniteDifferenceProbe<double> leftProbe{
        1, 2, 3, 0,
        1,
        0,
        1e-5,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };
    const MDWFFiniteDifferenceProbe<double> rightProbe{
        2, 1, 3, 0,
        2,
        1,
        1e-5,
        MDWFFiniteDifferenceMultiplicationSide::Right
    };

    const MDWFOneLinkForceAccumulatorMockResult<double> leftResult
        = runMDWFOneLinkForceAccumulatorMockCase(
            force, leftProbe, baseLinkLeft, actionMatrixLeft, commBase,
            "MDWF_one_link_force_accumulator_left_mock");
    const MDWFOneLinkForceAccumulatorMockResult<double> rightResult
        = runMDWFOneLinkForceAccumulatorMockCase(
            force, rightProbe, baseLinkRight, actionMatrixRight, commBase,
            "MDWF_one_link_force_accumulator_right_mock");

    rootLogger.info("MDWF one-link force-accumulator mock test passed with Ls = ", Ls,
                    ", leftDerivative = ", leftResult.contracted_derivative,
                    ", rightDerivative = ", rightResult.contracted_derivative,
                    ", maxOffProbeNorm = ",
                    std::max(leftResult.max_off_probe_norm, rightResult.max_off_probe_norm));
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

        runMDWFOneLinkForceAccumulatorMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
