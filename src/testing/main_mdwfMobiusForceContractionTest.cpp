/*
 * MDWF general-Mobius (RBC/UKQCD convention) selected-link force-contraction test.
 *
 * For the Mobius operator M = D_W(U) Din + Shift, Din and Shift are
 * gauge-independent, so dM = dD_W Din (D_W including clover). For
 *
 *     S = phi^\dagger R(M^\dagger M) phi,  R(x) = c0 + sum_i a_i / (x + sigma_i),
 *
 * with chi_i = (M^\dagger M + sigma_i)^{-1} phi and eta_i = M chi_i,
 *
 *     dS = -2 sum_i a_i Re[ eta_i^\dagger dD_W (Din chi_i) ].
 *
 * This is the Shamir selected-link contraction with the right vector chi_i
 * replaced by Din chi_i. The test evaluates it with the existing shared
 * selected-link helpers (mdwfAllLinkWilsonContractionTerm,
 * mdwfSelectedLinkCloverContractionTerm) and the existing
 * MDWFFermionForceWorkspace, and compares it against the finite-difference
 * action derivative. No new library code is introduced.
 *
 * Checks, on a fixed nontrivial (random) gauge field, all at c_sw = 0.5, for
 * two left-multiplication probes (the halo-touching link used by
 * mdwfMobiusFiniteDifferenceTest, and an interior link with a different mu
 * and generator):
 *
 *   1. Control (M5 = -2, i.e. positive Wilson kernel mass; not a domain-wall
 *      choice), b5 = 1.5: analytic Wilson + clover versus the epsilon = 1e-4
 *      finite difference, relative tolerance 1e-6.
 *   2. Physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5): analytic versus the
 *      Richardson extrapolation of the epsilon = {3e-4, 1e-4} finite
 *      differences, relative tolerance 5e-6.
 *   3. Din sensitivity control: the same contraction with chi_i instead of
 *      Din chi_i must disagree with the finite difference (relative > 1e-3)
 *      at b5 = 1.5, for both parameter sets.
 *   4. b5 = 1 regression: the Mobius analytic Wilson/clover derivatives equal
 *      the Shamir clover analytic ones (MDWFLinearOperator workspace, right
 *      vector chi_i) at the control M5.
 *
 * This does not accumulate a force field, define a production projection,
 * ipdot convention, or HMC sign, update momenta, call RHMC/HMC, or use
 * smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFAllLinkWilsonContraction.h"
#include "../experimental/mdwf/MDWFCloverAllLinkContraction.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

static constexpr size_t MDWFMobiusForceProbeCount = 2;

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusForceContractionSource {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.017) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFMobiusForceContractionActionEvaluator {
public:
    using ForwardOperator = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_commBase;
    Spinor &_field;
    MDWFRationalCoefficients<double> _coefficients;
    double _M5;
    double _mf;
    double _b5;
    double _csw;
    int _max_iter;
    double _precision;

public:
    MDWFMobiusForceContractionActionEvaluator(CommunicationBase &commBase,
                                              Spinor &field,
                                              const MDWFRationalCoefficients<double> &coefficients,
                                              double M5, double mf, double b5, double csw,
                                              int max_iter, double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _M5(M5),
          _mf(mf),
          _b5(b5),
          _csw(csw),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_force_fd_forward");
        AdjointOperator adjoint(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_force_fd_adjoint");
        NormalOperator normal(_commBase, forward, adjoint, "MDWF_mobius_force_fd_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_mobius_force_fd_action_workspace");

        MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, Adapter>(
            adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
            "MDWF_mobius_force_fd_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

bool mdwfMobiusForceFiniteDifferenceValid(const MDWFFiniteDifferenceResult<double> &result) {
    return result.converged
           && std::isfinite(result.derivative)
           && result.action_imag_relative <= 1e-8
           && result.plus.action_real > 0.0
           && result.minus.action_real > 0.0;
}

/*
 * Accumulates -2 a_i Re[eta_i^\dagger dD_W right_i] per probe, split into
 * Wilson and clover parts, with right_i = Din chi_i when din_coeff is given
 * and right_i = chi_i otherwise.
 */
template<size_t HaloDepth, size_t Ls, class Workspace>
void mdwfMobiusForceAnalyticDerivatives(
    CommunicationBase &commBase,
    const Workspace &workspace,
    const MDWFRationalCoefficients<double> &forceCoefficients,
    SU3Accessor<double, R18> gaugeAcc,
    const std::array<MDWFFiniteDifferenceProbe<double>, MDWFMobiusForceProbeCount> &probes,
    double csw,
    const MDWFFifthDimCoefficients<double> *din_coeff,
    std::array<double, MDWFMobiusForceProbeCount> &wilson,
    std::array<double, MDWFMobiusForceProbeCount> &clover,
    const std::string &name) {

    typedef GIndexer<All, HaloDepth> GInd;
    using DeviceSpinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    wilson.fill(0.0);
    clover.fill(0.0);

    for (const auto &probe : probes) {
        if (probe.multiplication_side != MDWFFiniteDifferenceMultiplicationSide::Left) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF Mobius force contraction test supports left-multiplication probes only"));
        }
    }

    for (size_t term = 0; term < workspace.size(); term++) {
        DeviceSpinor right(commBase, name + "_right_" + std::to_string(term));
        if (din_coeff != nullptr) {
            applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(
                right, workspace.chi(term), *din_coeff, true);
        } else {
            right = workspace.chi(term);
            right.updateAll();
        }

        HostSpinor rightHost(commBase, name + "_right_host_" + std::to_string(term));
        HostSpinor etaHost(commBase, name + "_eta_host_" + std::to_string(term));
        rightHost = right;
        etaHost = workspace.eta(term);
        const Vect12ArrayAcc<double> rightAcc = rightHost.getAccessor();
        const Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();
        const double weight = -2.0 * forceCoefficients.numerator[term];

        for (size_t p = 0; p < probes.size(); p++) {
            const gSite site = GInd::getSite(probes[p].x, probes[p].y, probes[p].z, probes[p].t);
            const gSiteMu link = GInd::getSiteMu(site, probes[p].mu);
            const SU3<double> direction = mdwfFiniteDifferenceGenerator<double>(probes[p].generator_id);

            wilson[p] += weight * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                rightAcc, etaAcc, gaugeAcc, site, probes[p].mu, direction);
            clover[p] += weight * mdwfSelectedLinkCloverContractionTerm<HaloDepth, Ls>(
                gaugeAcc, rightAcc, etaAcc, link, direction, probes[p].multiplication_side, csw);
        }
    }
}

template<class Workspace>
bool mdwfMobiusForceWorkspaceValid(const Workspace &workspace, double precision, double &maxResidue) {
    maxResidue = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        maxResidue = std::max(maxResidue, info.residue);
    }
    return workspace.converged() && maxResidue <= precision;
}

double mdwfMobiusForceRelDiff(double reference, double value) {
    return std::abs(reference - value) / std::max(1.0, std::abs(reference));
}

template<size_t Ls>
void runMDWFMobiusForceContractionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MobiusForward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusAdjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusNormal = MDWFNormalOperator<MobiusForward, MobiusAdjoint>;
    using MobiusWorkspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, MobiusNormal, MobiusForward>;
    using ShamirForward = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirAdjoint = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirNormal = MDWFNormalOperator<ShamirForward, ShamirAdjoint>;
    using ShamirWorkspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, ShamirNormal, ShamirForward>;
    using Evaluator = MDWFMobiusForceContractionActionEvaluator<HaloDepth, Ls>;
    using ProbeArray = std::array<MDWFFiniteDifferenceProbe<double>, MDWFMobiusForceProbeCount>;
    using ValueArray = std::array<double, MDWFMobiusForceProbeCount>;

    const double csw = 0.5;
    const double mf = 0.05;
    const double genericB5 = 1.5;
    const double controlM5 = -2.0;
    const double physicalM5 = 1.8;
    const int controlMaxIter = 2000;
    const int physicalMaxIter = 20000;
    const double precision = 1e-10;
    const double controlTolerance = 1e-6;
    const double physicalTolerance = 5e-6;
    const double dinSensitivityThreshold = 1e-3;
    const double b5OneTolerance = 1e-13;

    MDWFExplicitRationalInput<double> actionInput{
        "mobius_force_contraction_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFExplicitRationalInput<double> forceInput{
        "mobius_force_contraction_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> actionCoefficients = makeMDWFRationalCoefficients(actionInput);
    const MDWFRationalCoefficients<double> forceCoefficients = makeMDWFRationalCoefficients(forceInput);

    Gauge baseGauge(commBase, "MDWF_mobius_force_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_mobius_force_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_mobius_force_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    HostGauge gaugeHost(commBase, "MDWF_mobius_force_gauge_host");
    gaugeHost = baseGauge;
    const SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();

    Spinor field(commBase, "MDWF_mobius_force_field");
    field.template iterateOverBulk<>(FillMDWFMobiusForceContractionSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    ProbeArray probes = {{
        {1, 2, 3, 0, 1, 0, 1e-4, MDWFFiniteDifferenceMultiplicationSide::Left},
        {2, 2, 2, 2, 2, 1, 1e-4, MDWFFiniteDifferenceMultiplicationSide::Left}
    }};
    auto withEpsilon = [](MDWFFiniteDifferenceProbe<double> probe, double epsilon) {
        probe.epsilon = epsilon;
        return probe;
    };

    bool allPassed = true;

    // --- Part 1/3: control, b5 = 1.5. ---
    Evaluator controlEvaluator(commBase, field, actionCoefficients, controlM5, mf, genericB5, csw,
                               controlMaxIter, precision);
    ValueArray controlFd{};
    bool controlFdValid = true;
    for (size_t p = 0; p < probes.size(); p++) {
        const MDWFFiniteDifferenceResult<double> fd = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, withEpsilon(probes[p], 1e-4), controlEvaluator);
        controlFd[p] = fd.derivative;
        controlFdValid = controlFdValid && mdwfMobiusForceFiniteDifferenceValid(fd);
    }

    MobiusForward controlForward(baseGauge, controlM5, mf, genericB5, csw, "MDWF_mobius_force_control_forward");
    MobiusAdjoint controlAdjoint(baseGauge, controlM5, mf, genericB5, csw, "MDWF_mobius_force_control_adjoint");
    MobiusNormal controlNormal(commBase, controlForward, controlAdjoint, "MDWF_mobius_force_control_normal");
    MobiusWorkspace controlWorkspace;
    controlWorkspace.prepare(controlNormal, controlForward, field, forceCoefficients, controlMaxIter, precision,
                             "MDWF_mobius_force_control_workspace");
    double controlWorkspaceResidue = 0.0;
    const bool controlWorkspaceValid = mdwfMobiusForceWorkspaceValid(controlWorkspace, precision,
                                                                      controlWorkspaceResidue);

    ValueArray controlWilson{}, controlClover{}, controlNoDinWilson{}, controlNoDinClover{};
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, controlWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        &controlForward.params().dinCoeff, controlWilson, controlClover, "MDWF_mobius_force_control");
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, controlWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        nullptr, controlNoDinWilson, controlNoDinClover, "MDWF_mobius_force_control_no_din");

    bool controlPassed = controlFdValid && controlWorkspaceValid;
    for (size_t p = 0; p < probes.size(); p++) {
        const double total = controlWilson[p] + controlClover[p];
        const double noDinTotal = controlNoDinWilson[p] + controlNoDinClover[p];
        const double relDiff = mdwfMobiusForceRelDiff(controlFd[p], total);
        const double noDinRelDiff = mdwfMobiusForceRelDiff(controlFd[p], noDinTotal);
        controlPassed = controlPassed
                        && relDiff <= controlTolerance
                        && std::abs(controlClover[p]) > 1e-8
                        && noDinRelDiff > dinSensitivityThreshold;
        rootLogger.info("MDWF Mobius force contraction control (M5 = -2, b5 = 1.5) probe ", p,
                        " (x,y,z,t = ", probes[p].x, ",", probes[p].y, ",", probes[p].z, ",", probes[p].t,
                        ", mu = ", static_cast<int>(probes[p].mu), ", generator = ", probes[p].generator_id,
                        "): finiteDifference = ", controlFd[p],
                        ", wilsonAnalytic = ", controlWilson[p],
                        ", cloverAnalytic = ", controlClover[p],
                        ", totalAnalytic = ", total,
                        ", relDiff = ", relDiff,
                        ", noDinTotal = ", noDinTotal,
                        ", noDinRelDiff = ", noDinRelDiff);
    }
    rootLogger.info("MDWF Mobius force contraction control workspace: converged = ", controlWorkspace.converged(),
                    ", maxResidue = ", controlWorkspaceResidue, ", passed = ", controlPassed);
    allPassed = allPassed && controlPassed;

    // --- Part 4: b5 = 1 Mobius vs Shamir analytic, control M5. ---
    MobiusForward b5OneForward(baseGauge, controlM5, mf, 1.0, csw, "MDWF_mobius_force_b5_one_forward");
    MobiusAdjoint b5OneAdjoint(baseGauge, controlM5, mf, 1.0, csw, "MDWF_mobius_force_b5_one_adjoint");
    MobiusNormal b5OneNormal(commBase, b5OneForward, b5OneAdjoint, "MDWF_mobius_force_b5_one_normal");
    MobiusWorkspace b5OneWorkspace;
    b5OneWorkspace.prepare(b5OneNormal, b5OneForward, field, forceCoefficients, controlMaxIter, precision,
                           "MDWF_mobius_force_b5_one_workspace");

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double controlMass = mdwfShamirKernelMass(controlM5);
    ShamirForward shamirForward(baseGauge, shamirCoeff, controlMass, csw, "MDWF_mobius_force_shamir_forward");
    ShamirAdjoint shamirAdjoint(baseGauge, shamirCoeff, controlMass, csw, "MDWF_mobius_force_shamir_adjoint");
    ShamirNormal shamirNormal(commBase, shamirForward, shamirAdjoint, "MDWF_mobius_force_shamir_normal");
    ShamirWorkspace shamirWorkspace;
    shamirWorkspace.prepare(shamirNormal, shamirForward, field, forceCoefficients, controlMaxIter, precision,
                            "MDWF_mobius_force_shamir_workspace");

    double b5OneResidue = 0.0;
    double shamirResidue = 0.0;
    const bool b5OneWorkspaceValid = mdwfMobiusForceWorkspaceValid(b5OneWorkspace, precision, b5OneResidue);
    const bool shamirWorkspaceValid = mdwfMobiusForceWorkspaceValid(shamirWorkspace, precision, shamirResidue);

    ValueArray b5OneWilson{}, b5OneClover{}, shamirWilson{}, shamirClover{};
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, b5OneWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        &b5OneForward.params().dinCoeff, b5OneWilson, b5OneClover, "MDWF_mobius_force_b5_one");
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, shamirWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        nullptr, shamirWilson, shamirClover, "MDWF_mobius_force_shamir");

    bool b5OnePassed = b5OneWorkspaceValid && shamirWorkspaceValid;
    for (size_t p = 0; p < probes.size(); p++) {
        const double wilsonDiff = mdwfMobiusForceRelDiff(shamirWilson[p], b5OneWilson[p]);
        const double cloverDiff = mdwfMobiusForceRelDiff(shamirClover[p], b5OneClover[p]);
        b5OnePassed = b5OnePassed && wilsonDiff <= b5OneTolerance && cloverDiff <= b5OneTolerance;
        rootLogger.info("MDWF Mobius force contraction b5 = 1 regression (M5 = -2) probe ", p,
                        ": mobiusWilson = ", b5OneWilson[p], ", shamirWilson = ", shamirWilson[p],
                        ", wilsonRelDiff = ", wilsonDiff,
                        ", mobiusClover = ", b5OneClover[p], ", shamirClover = ", shamirClover[p],
                        ", cloverRelDiff = ", cloverDiff);
    }
    rootLogger.info("MDWF Mobius force contraction b5 = 1 regression: mobiusMaxResidue = ", b5OneResidue,
                    ", shamirMaxResidue = ", shamirResidue, ", passed = ", b5OnePassed);
    allPassed = allPassed && b5OnePassed;

    // --- Part 2/3: physical-like, b5 = 1.5, Richardson-extrapolated finite difference. ---
    Evaluator physicalEvaluator(commBase, field, actionCoefficients, physicalM5, mf, genericB5, csw,
                                physicalMaxIter, precision);
    ValueArray physicalFdCoarse{}, physicalFdFine{}, physicalFdRichardson{};
    bool physicalFdValid = true;
    for (size_t p = 0; p < probes.size(); p++) {
        const MDWFFiniteDifferenceResult<double> coarse = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, withEpsilon(probes[p], 3e-4), physicalEvaluator);
        const MDWFFiniteDifferenceResult<double> fine = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, withEpsilon(probes[p], 1e-4), physicalEvaluator);
        physicalFdCoarse[p] = coarse.derivative;
        physicalFdFine[p] = fine.derivative;
        // Centered differences have error c * epsilon^2; with epsilon ratio 3 the
        // leading term cancels in d_fine + (d_fine - d_coarse) / 8.
        physicalFdRichardson[p] = fine.derivative + (fine.derivative - coarse.derivative) / 8.0;
        physicalFdValid = physicalFdValid
                          && mdwfMobiusForceFiniteDifferenceValid(coarse)
                          && mdwfMobiusForceFiniteDifferenceValid(fine);
    }

    MobiusForward physicalForward(baseGauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_force_physical_forward");
    MobiusAdjoint physicalAdjoint(baseGauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_force_physical_adjoint");
    MobiusNormal physicalNormal(commBase, physicalForward, physicalAdjoint, "MDWF_mobius_force_physical_normal");
    MobiusWorkspace physicalWorkspace;
    physicalWorkspace.prepare(physicalNormal, physicalForward, field, forceCoefficients, physicalMaxIter, precision,
                              "MDWF_mobius_force_physical_workspace");
    double physicalWorkspaceResidue = 0.0;
    const bool physicalWorkspaceValid = mdwfMobiusForceWorkspaceValid(physicalWorkspace, precision,
                                                                       physicalWorkspaceResidue);

    ValueArray physicalWilson{}, physicalClover{}, physicalNoDinWilson{}, physicalNoDinClover{};
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, physicalWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        &physicalForward.params().dinCoeff, physicalWilson, physicalClover, "MDWF_mobius_force_physical");
    mdwfMobiusForceAnalyticDerivatives<HaloDepth, Ls>(
        commBase, physicalWorkspace, forceCoefficients, gaugeAcc, probes, csw,
        nullptr, physicalNoDinWilson, physicalNoDinClover, "MDWF_mobius_force_physical_no_din");

    bool physicalPassed = physicalFdValid && physicalWorkspaceValid;
    for (size_t p = 0; p < probes.size(); p++) {
        const double total = physicalWilson[p] + physicalClover[p];
        const double noDinTotal = physicalNoDinWilson[p] + physicalNoDinClover[p];
        const double relDiff = mdwfMobiusForceRelDiff(physicalFdRichardson[p], total);
        const double fineRelDiff = mdwfMobiusForceRelDiff(physicalFdFine[p], total);
        const double noDinRelDiff = mdwfMobiusForceRelDiff(physicalFdRichardson[p], noDinTotal);
        physicalPassed = physicalPassed
                         && relDiff <= physicalTolerance
                         && std::abs(physicalClover[p]) > 1e-8
                         && noDinRelDiff > dinSensitivityThreshold;
        rootLogger.info("MDWF Mobius force contraction physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5) probe ", p,
                        " (x,y,z,t = ", probes[p].x, ",", probes[p].y, ",", probes[p].z, ",", probes[p].t,
                        ", mu = ", static_cast<int>(probes[p].mu), ", generator = ", probes[p].generator_id,
                        "): fd_3e-4 = ", physicalFdCoarse[p],
                        ", fd_1e-4 = ", physicalFdFine[p],
                        ", fdRichardson = ", physicalFdRichardson[p],
                        ", wilsonAnalytic = ", physicalWilson[p],
                        ", cloverAnalytic = ", physicalClover[p],
                        ", totalAnalytic = ", total,
                        ", relDiff (vs Richardson) = ", relDiff,
                        ", relDiff (vs 1e-4) = ", fineRelDiff,
                        ", noDinTotal = ", noDinTotal,
                        ", noDinRelDiff = ", noDinRelDiff);
    }
    rootLogger.info("MDWF Mobius force contraction physical-like workspace: converged = ",
                    physicalWorkspace.converged(), ", maxResidue = ", physicalWorkspaceResidue,
                    ", passed = ", physicalPassed);
    allPassed = allPassed && physicalPassed;

    if (!allPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius force contraction test failed: control passed = ", controlPassed,
            ", b5 = 1 regression passed = ", b5OnePassed,
            ", physical-like passed = ", physicalPassed,
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF Mobius force contraction test passed with Ls = ", Ls, ", c_sw = ", csw,
                    ", probes = ", probes.size(), ", terms = ", forceCoefficients.shift.size());
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

        runMDWFMobiusForceContractionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
