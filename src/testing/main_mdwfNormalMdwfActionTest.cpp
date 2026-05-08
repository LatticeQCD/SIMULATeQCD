/*
 * MDWF normal-operator rational action smoke test.
 *
 * This evaluates phi^\dagger R(M^\dagger M) phi using the isolated MDWF
 * normal operator on a fixed random gauge field.  It checks only the
 * solver/action wrapper interface: finite positive real action, small
 * imaginary part, and converged shifted solves.  It does not call RHMC/HMC,
 * compute forces, or touch HISQ/smearing code.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFPseudofermionAction.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFNormalActionField {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.25)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.02) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.003) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFNormalMdwfActionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;

    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> actionInput{
        "fixed_gauge_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_normal_action_fixed_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260511);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    Spinor field(commBase, "MDWF_normal_action_field");
    Spinor actionWorkspace(commBase, "MDWF_normal_action_workspace");

    field.template iterateOverBulk<>(FillMDWFNormalActionField<double, All, HaloDepth, Ls>());
    field.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_normal_action_forward");
    AdjointOperator adjoint(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_normal_action_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_normal_action_normal");
    NormalAdapter normalAdapter(normal);

    MDWFRationalActionResult<double> actionResult
        = computeMDWFRationalAction<double, NormalAdapter>(
            normalAdapter, actionWorkspace, field, coefficients, 512, 1e-8, "MDWF_normal_action");

    double maxResidue = 0.0;
    for (size_t term = 0; term < actionResult.rational_result.shifts.size(); term++) {
        maxResidue = std::max(maxResidue, actionResult.rational_result.shifts[term].residue);
    }

    const double fieldNorm2 = normalAdapter.norm2(field);
    const double workspaceNorm2 = normalAdapter.norm2(actionWorkspace);
    const double actionScale = std::max(1.0, std::abs(actionResult.action_real));
    const double actionImagRel = std::abs(actionResult.action_imag) / actionScale;

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || forward.csw() != nonzeroCsw
        || adjoint.csw() != nonzeroCsw
        || !actionResult.rational_result.converged()
        || maxResidue > 1e-8
        || fieldNorm2 <= 0.0
        || workspaceNorm2 <= 0.0
        || !std::isfinite(actionResult.action_real)
        || !std::isfinite(actionResult.action_imag)
        || actionResult.action_real <= 0.0
        || actionImagRel > 1e-9) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF normal-operator rational action test failed: c_sw = ", nonzeroCsw,
            ", converged = ", actionResult.rational_result.converged(),
            ", maxResidue = ", maxResidue,
            ", actionReal = ", actionResult.action_real,
            ", actionImag = ", actionResult.action_imag,
            ", actionImagRel = ", actionImagRel,
            ", fieldNorm2 = ", fieldNorm2,
            ", workspaceNorm2 = ", workspaceNorm2));
    }

    rootLogger.info("MDWF normal-operator rational action test passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", terms = ", coefficients.shift.size(),
                    ", actionReal = ", actionResult.action_real,
                    ", actionImagRel = ", actionImagRel,
                    ", maxResidue = ", maxResidue);
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

        runMDWFNormalMdwfActionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
