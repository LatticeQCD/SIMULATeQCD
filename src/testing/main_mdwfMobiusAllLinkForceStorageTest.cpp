/*
 * MDWF general-Mobius (RBC/UKQCD convention) all-link force-storage test.
 *
 * Runs the existing, workspace-generic single-rank all-link storage
 * (overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero) on an
 * MDWFMobiusForceWorkspaceView, which presents Din chi_i as the right
 * contraction vector. The storage writes one projected matrix K_l = TA(B_W + B_C)
 * per bulk link, in the left-variation convention dS(H) = Re tr(H K_l) for
 * U_l -> exp(epsilon H) U_l. No new storage or contraction code is introduced;
 * the overwrite/sentinel semantics of the storage function are already
 * validated and do not depend on the workspace type.
 *
 * Checks, on a fixed nontrivial (random) gauge field, all at c_sw = 0.5:
 *
 *   1. Control (M5 = -2, i.e. positive Wilson kernel mass; not a domain-wall
 *      choice), b5 = 1.5: sum_l Re tr(H_l K_l) with the existing deterministic
 *      all-link direction field equals the all-link centered finite
 *      difference of the Mobius action (epsilon = 1e-4, relative 1e-6;
 *      epsilon = 3e-4 reported for stability).
 *   2. Physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5): the same comparison at
 *      epsilon = 1e-4, relative 1e-5.
 *   3. Selected-link cross-check: Re tr(G K_l) at the two probe links of
 *      mdwfMobiusForceContractionTest equals the direct selected-link
 *      Wilson + clover contraction (a separate code path), relative 1e-10.
 *   4. Storage sanity: every bulk link finalized once, with one Wilson and one
 *      clover term addition per rational term and 24 clover path additions
 *      per term; stored matrices finite, anti-Hermitian, and traceless.
 *   5. Din sensitivity control: storage from the plain workspace (chi_i
 *      instead of Din chi_i) disagrees with the finite difference (relative
 *      > 1e-3) for both parameter sets.
 *   6. b5 = 1 regression: Mobius storage equals the Shamir clover storage
 *      (projected output and raw Wilson/clover buffers) at the control M5.
 *
 * Single rank only. This does not define ipdot, an HMC sign, a caller-facing
 * additive API, output halo refresh, MPI ownership, or RHMC/HMC wiring.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFAllLinkDirectionIndependentStorage.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFMobiusForceWorkspace.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t MDWFMobiusStorageProbeCount = 2;

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusAllLinkStorageSource {
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
class MDWFMobiusAllLinkStorageActionEvaluator {
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
    MDWFMobiusAllLinkStorageActionEvaluator(CommunicationBase &commBase,
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
        ForwardOperator forward(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_storage_fd_forward");
        AdjointOperator adjoint(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_storage_fd_adjoint");
        NormalOperator normal(_commBase, forward, adjoint, "MDWF_mobius_storage_fd_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_mobius_storage_fd_action_workspace");

        MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, Adapter>(
            adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
            "MDWF_mobius_storage_fd_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<size_t HaloDepth, class ActionEvaluator>
MDWFFiniteDifferenceResult<double> evaluateMDWFMobiusAllLinkFiniteDifference(
    Gaugefield<double, true, HaloDepth, R18> &gaugePlus,
    Gaugefield<double, true, HaloDepth, R18> &gaugeMinus,
    const Gaugefield<double, true, HaloDepth, R18> &baseGauge,
    double epsilon,
    ActionEvaluator &evaluator) {

    applyMDWFAllLinkPerturbation(gaugePlus, baseGauge, epsilon, 1);
    applyMDWFAllLinkPerturbation(gaugeMinus, baseGauge, epsilon, -1);

    MDWFFiniteDifferenceActionValue<double> plus = evaluator(gaugePlus);
    MDWFFiniteDifferenceActionValue<double> minus = evaluator(gaugeMinus);
    const double plusScale = std::max(1.0, std::abs(plus.action_real));
    const double minusScale = std::max(1.0, std::abs(minus.action_real));

    return {
        plus,
        minus,
        (plus.action_real - minus.action_real) / (2.0 * epsilon),
        std::max(std::abs(plus.action_imag) / plusScale, std::abs(minus.action_imag) / minusScale),
        std::max(plus.max_shifted_residual, minus.max_shifted_residual),
        plus.converged && minus.converged
    };
}

bool mdwfMobiusStorageFiniteDifferenceValid(const MDWFFiniteDifferenceResult<double> &result) {
    return result.converged
           && std::isfinite(result.derivative)
           && result.action_imag_relative <= 1e-8
           && result.plus.action_real > 0.0
           && result.minus.action_real > 0.0;
}

// sum over bulk links of Re tr(H_l K_l) with the deterministic all-link direction field.
template<size_t HaloDepth>
double mdwfMobiusStoredDirectionalDerivative(const Gaugefield<double, false, HaloDepth, R18> &destination) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = destination.getAccessor();
    double derivative = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            const SU3<double> direction = mdwfAllLinkDeterministicDirection<double, HaloDepth>(siteMu);
            derivative += real(tr_c(direction, acc.getLink(siteMu)));
        }
    }
    return derivative;
}

struct MDWFMobiusStorageSanity {
    size_t invalidLinks;
    size_t badCoverageLinks;
    double maxAntiHermitianViolation;
    double maxTraceViolation;
    double maxStoredNorm;
};

template<size_t HaloDepth>
MDWFMobiusStorageSanity mdwfMobiusStorageSanity(const Gaugefield<double, false, HaloDepth, R18> &destination,
                                                const MDWFAllLinkDirectionIndependentStorageResult<double> &result,
                                                size_t terms) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = destination.getAccessor();
    MDWFMobiusStorageSanity sanity{0, 0, 0.0, 0.0, 0.0};

    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const size_t bulkLink = site.isite * 4 + mu;
            const SU3<double> stored = acc.getLink(GInd::getSiteMu(site, mu));
            const double norm = static_cast<double>(infnorm(stored));
            const double antiHermitian = static_cast<double>(infnorm(stored + dagger(stored)));
            const double trace = static_cast<double>(abs(tr_c(stored)));

            if (!std::isfinite(norm) || !std::isfinite(antiHermitian) || !std::isfinite(trace)) {
                sanity.invalidLinks++;
            }
            sanity.maxStoredNorm = std::max(sanity.maxStoredNorm, norm);
            sanity.maxAntiHermitianViolation = std::max(sanity.maxAntiHermitianViolation, antiHermitian);
            sanity.maxTraceViolation = std::max(sanity.maxTraceViolation, trace);

            if (result.finalize_counts[bulkLink] != 1
                || result.wilson_term_additions[bulkLink] != terms
                || result.clover_term_additions[bulkLink] != terms
                || result.clover_path_additions[bulkLink] != 24 * terms) {
                sanity.badCoverageLinks++;
            }
        }
    }
    return sanity;
}

bool mdwfMobiusStorageSanityPassed(const MDWFMobiusStorageSanity &sanity,
                                   const MDWFAllLinkDirectionIndependentStorageResult<double> &result,
                                   size_t expectedLinks) {
    const double scale = std::max(1.0, sanity.maxStoredNorm);
    return result.bulk_links == expectedLinks
           && sanity.invalidLinks == 0
           && sanity.badCoverageLinks == 0
           && sanity.maxAntiHermitianViolation <= 1e-12 * scale
           && sanity.maxTraceViolation <= 1e-12 * scale;
}

template<size_t HaloDepth>
double mdwfMobiusMaxDestinationDiff(const Gaugefield<double, false, HaloDepth, R18> &a,
                                    const Gaugefield<double, false, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> accA = a.getAccessor();
    const SU3Accessor<double, R18> accB = b.getAccessor();
    double maxDiff = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            maxDiff = std::max(maxDiff, static_cast<double>(infnorm(accA.getLink(siteMu) - accB.getLink(siteMu))));
        }
    }
    return maxDiff;
}

double mdwfMobiusMaxRawDiff(const std::vector<SU3<double>> &a, const std::vector<SU3<double>> &b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double maxDiff = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        maxDiff = std::max(maxDiff, static_cast<double>(infnorm(a[i] - b[i])));
    }
    return maxDiff;
}

// Direct selected-link Wilson + clover contraction, independent of the storage code path.
template<size_t HaloDepth, size_t Ls, class Workspace>
std::array<double, MDWFMobiusStorageProbeCount> mdwfMobiusSelectedLinkDirect(
    CommunicationBase &commBase,
    const Workspace &workspace,
    const MDWFRationalCoefficients<double> &forceCoefficients,
    SU3Accessor<double, R18> gaugeAcc,
    const std::array<MDWFFiniteDifferenceProbe<double>, MDWFMobiusStorageProbeCount> &probes,
    double csw,
    const std::string &name) {

    typedef GIndexer<All, HaloDepth> GInd;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;
    std::array<double, MDWFMobiusStorageProbeCount> derivative{};

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor rightHost(commBase, name + "_right_host_" + std::to_string(term));
        HostSpinor etaHost(commBase, name + "_eta_host_" + std::to_string(term));
        rightHost = workspace.chi(term);
        etaHost = workspace.eta(term);
        const Vect12ArrayAcc<double> rightAcc = rightHost.getAccessor();
        const Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();
        const double weight = -2.0 * forceCoefficients.numerator[term];

        for (size_t p = 0; p < probes.size(); p++) {
            const gSite site = GInd::getSite(probes[p].x, probes[p].y, probes[p].z, probes[p].t);
            const gSiteMu link = GInd::getSiteMu(site, probes[p].mu);
            const SU3<double> direction = mdwfFiniteDifferenceGenerator<double>(probes[p].generator_id);
            derivative[p] += weight * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                rightAcc, etaAcc, gaugeAcc, site, probes[p].mu, direction);
            derivative[p] += weight * mdwfSelectedLinkCloverContractionTerm<HaloDepth, Ls>(
                gaugeAcc, rightAcc, etaAcc, link, direction, probes[p].multiplication_side, csw);
        }
    }
    return derivative;
}

template<size_t HaloDepth>
std::array<double, MDWFMobiusStorageProbeCount> mdwfMobiusSelectedLinkStored(
    const Gaugefield<double, false, HaloDepth, R18> &destination,
    const std::array<MDWFFiniteDifferenceProbe<double>, MDWFMobiusStorageProbeCount> &probes) {

    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = destination.getAccessor();
    std::array<double, MDWFMobiusStorageProbeCount> derivative{};
    for (size_t p = 0; p < probes.size(); p++) {
        const gSiteMu link = GInd::getSiteMu(probes[p].x, probes[p].y, probes[p].z, probes[p].t, probes[p].mu);
        const SU3<double> generator = mdwfFiniteDifferenceGenerator<double>(probes[p].generator_id);
        derivative[p] = real(tr_c(generator, acc.getLink(link)));
    }
    return derivative;
}

double mdwfMobiusStorageRelDiff(double reference, double value) {
    return std::abs(reference - value) / std::max(1.0, std::abs(reference));
}

template<size_t Ls>
void runMDWFMobiusAllLinkForceStorageTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MobiusForward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusAdjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusNormal = MDWFNormalOperator<MobiusForward, MobiusAdjoint>;
    using MobiusWorkspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, MobiusNormal, MobiusForward>;
    using MobiusView = MDWFMobiusForceWorkspaceView<double, HaloDepth, Ls, MobiusWorkspace>;
    using ShamirForward = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirAdjoint = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirNormal = MDWFNormalOperator<ShamirForward, ShamirAdjoint>;
    using ShamirWorkspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, ShamirNormal, ShamirForward>;
    using Evaluator = MDWFMobiusAllLinkStorageActionEvaluator<HaloDepth, Ls>;
    using StorageResult = MDWFAllLinkDirectionIndependentStorageResult<double>;
    using ProbeArray = std::array<MDWFFiniteDifferenceProbe<double>, MDWFMobiusStorageProbeCount>;
    using ValueArray = std::array<double, MDWFMobiusStorageProbeCount>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius all-link force storage test is single-rank only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }
    const size_t expectedLinks = 4 * lat.vol4;

    const double csw = 0.5;
    const double mf = 0.05;
    const double genericB5 = 1.5;
    const double controlM5 = -2.0;
    const double physicalM5 = 1.8;
    const int controlMaxIter = 2000;
    const int physicalMaxIter = 20000;
    const double precision = 1e-10;
    const double controlTolerance = 1e-6;
    const double physicalTolerance = 1e-5;
    const double selectedLinkTolerance = 1e-10;
    const double dinSensitivityThreshold = 1e-3;
    const double b5OneTolerance = 1e-12;

    MDWFExplicitRationalInput<double> actionInput{
        "mobius_all_link_storage_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFExplicitRationalInput<double> forceInput{
        "mobius_all_link_storage_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> actionCoefficients = makeMDWFRationalCoefficients(actionInput);
    const MDWFRationalCoefficients<double> forceCoefficients = makeMDWFRationalCoefficients(forceInput);
    const size_t terms = forceCoefficients.shift.size();

    Gauge baseGauge(commBase, "MDWF_mobius_storage_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_mobius_storage_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_mobius_storage_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    HostGauge gaugeHost(commBase, "MDWF_mobius_storage_gauge_host");
    gaugeHost = baseGauge;
    const SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();

    Spinor field(commBase, "MDWF_mobius_storage_field");
    field.template iterateOverBulk<>(FillMDWFMobiusAllLinkStorageSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    const ProbeArray probes = {{
        {1, 2, 3, 0, 1, 0, 1e-4, MDWFFiniteDifferenceMultiplicationSide::Left},
        {2, 2, 2, 2, 2, 1, 1e-4, MDWFFiniteDifferenceMultiplicationSide::Left}
    }};

    auto zero = [](HostGauge &destination) {
        destination.template iterateOverFullAllMu<>(MDWFAllLinkZeroMatrix<double>());
    };

    bool allPassed = true;

    // Runs storage with and without Din for one parameter set and logs all diagnostics.
    auto runCase = [&](const std::string &label, MobiusForward &forward, MobiusNormal &normal, int maxIter,
                       const std::vector<double> &fdEpsilons, Evaluator &evaluator, double tolerance) {
        std::vector<MDWFFiniteDifferenceResult<double>> fds;
        bool fdValid = true;
        for (double epsilon : fdEpsilons) {
            fds.push_back(evaluateMDWFMobiusAllLinkFiniteDifference<HaloDepth>(
                gaugePlus, gaugeMinus, baseGauge, epsilon, evaluator));
            fdValid = fdValid && mdwfMobiusStorageFiniteDifferenceValid(fds.back());
            rootLogger.info("MDWF Mobius all-link storage ", label, " finite difference epsilon = ", epsilon,
                            ": derivative = ", fds.back().derivative,
                            ", actionPlus = ", fds.back().plus.action_real,
                            ", actionImagRelative = ", fds.back().action_imag_relative,
                            ", maxShiftedResidual = ", fds.back().max_shifted_residual);
        }
        const double fd = fds.back().derivative;

        MobiusWorkspace workspace;
        workspace.prepare(normal, forward, field, forceCoefficients, maxIter, precision,
                          "MDWF_mobius_storage_" + label + "_workspace");
        double workspaceResidue = 0.0;
        for (const auto &info : workspace.shiftInfo()) {
            workspaceResidue = std::max(workspaceResidue, info.residue);
        }
        const bool workspaceValid = workspace.converged() && workspaceResidue <= precision;
        MobiusView view(workspace, forward.params().dinCoeff, "MDWF_mobius_storage_" + label + "_view");

        HostGauge destination(commBase, "MDWF_mobius_storage_" + label + "_destination");
        HostGauge noDinDestination(commBase, "MDWF_mobius_storage_" + label + "_no_din_destination");
        zero(destination);
        zero(noDinDestination);
        const StorageResult result = overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
            destination, gaugeHost, view, forceCoefficients, csw, commBase,
            "MDWF_mobius_storage_" + label);
        const StorageResult noDinResult = overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
            noDinDestination, gaugeHost, workspace, forceCoefficients, csw, commBase,
            "MDWF_mobius_storage_" + label + "_no_din");

        const double stored = mdwfMobiusStoredDirectionalDerivative<HaloDepth>(destination);
        const double noDinStored = mdwfMobiusStoredDirectionalDerivative<HaloDepth>(noDinDestination);
        const double relDiff = mdwfMobiusStorageRelDiff(fd, stored);
        const double noDinRelDiff = mdwfMobiusStorageRelDiff(fd, noDinStored);
        const double stabilityRelDiff = fds.size() > 1
                                        ? mdwfMobiusStorageRelDiff(fd, fds.front().derivative) : 0.0;

        const MDWFMobiusStorageSanity sanity = mdwfMobiusStorageSanity<HaloDepth>(destination, result, terms);
        const bool sanityPassed = mdwfMobiusStorageSanityPassed(sanity, result, expectedLinks);

        const ValueArray direct = mdwfMobiusSelectedLinkDirect<HaloDepth, Ls>(
            commBase, view, forceCoefficients, gaugeAcc, probes, csw, "MDWF_mobius_storage_" + label + "_direct");
        const ValueArray storedSelected = mdwfMobiusSelectedLinkStored<HaloDepth>(destination, probes);
        double maxSelectedRelDiff = 0.0;
        for (size_t p = 0; p < probes.size(); p++) {
            const double selectedRelDiff = mdwfMobiusStorageRelDiff(direct[p], storedSelected[p]);
            maxSelectedRelDiff = std::max(maxSelectedRelDiff, selectedRelDiff);
            rootLogger.info("MDWF Mobius all-link storage ", label, " selected link ", p,
                            " (x,y,z,t = ", probes[p].x, ",", probes[p].y, ",", probes[p].z, ",", probes[p].t,
                            ", mu = ", static_cast<int>(probes[p].mu), ", generator = ", probes[p].generator_id,
                            "): direct = ", direct[p], ", stored = ", storedSelected[p],
                            ", relDiff = ", selectedRelDiff);
        }

        const bool passed = fdValid && workspaceValid && sanityPassed
                            && relDiff <= tolerance
                            && noDinRelDiff > dinSensitivityThreshold
                            && maxSelectedRelDiff <= selectedLinkTolerance;
        rootLogger.info("MDWF Mobius all-link storage ", label,
                        ": finiteDifference = ", fd,
                        ", storedDirectional = ", stored,
                        ", relDiff = ", relDiff,
                        ", epsilonStabilityRelDiff = ", stabilityRelDiff,
                        ", noDinStoredDirectional = ", noDinStored,
                        ", noDinRelDiff = ", noDinRelDiff,
                        ", maxSelectedRelDiff = ", maxSelectedRelDiff,
                        ", bulkLinks = ", result.bulk_links,
                        ", badCoverageLinks = ", sanity.badCoverageLinks,
                        ", invalidLinks = ", sanity.invalidLinks,
                        ", maxAntiHermitianViolation = ", sanity.maxAntiHermitianViolation,
                        ", maxTraceViolation = ", sanity.maxTraceViolation,
                        ", maxStoredNorm = ", sanity.maxStoredNorm,
                        ", workspaceMaxResidue = ", workspaceResidue,
                        ", noDinBulkLinks = ", noDinResult.bulk_links,
                        ", passed = ", passed);
        return passed;
    };

    // --- Parts 1, 3, 4, 5: control, b5 = 1.5. ---
    MobiusForward controlForward(baseGauge, controlM5, mf, genericB5, csw, "MDWF_mobius_storage_control_forward");
    MobiusAdjoint controlAdjoint(baseGauge, controlM5, mf, genericB5, csw, "MDWF_mobius_storage_control_adjoint");
    MobiusNormal controlNormal(commBase, controlForward, controlAdjoint, "MDWF_mobius_storage_control_normal");
    Evaluator controlEvaluator(commBase, field, actionCoefficients, controlM5, mf, genericB5, csw,
                               controlMaxIter, precision);
    const bool controlPassed = runCase("control", controlForward, controlNormal, controlMaxIter,
                                       {3e-4, 1e-4}, controlEvaluator, controlTolerance);
    allPassed = allPassed && controlPassed;

    // --- Part 6: b5 = 1 Mobius storage vs Shamir clover storage. ---
    MobiusForward b5OneForward(baseGauge, controlM5, mf, 1.0, csw, "MDWF_mobius_storage_b5_one_forward");
    MobiusAdjoint b5OneAdjoint(baseGauge, controlM5, mf, 1.0, csw, "MDWF_mobius_storage_b5_one_adjoint");
    MobiusNormal b5OneNormal(commBase, b5OneForward, b5OneAdjoint, "MDWF_mobius_storage_b5_one_normal");
    MobiusWorkspace b5OneWorkspace;
    b5OneWorkspace.prepare(b5OneNormal, b5OneForward, field, forceCoefficients, controlMaxIter, precision,
                           "MDWF_mobius_storage_b5_one_workspace");
    MobiusView b5OneView(b5OneWorkspace, b5OneForward.params().dinCoeff, "MDWF_mobius_storage_b5_one_view");

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double controlMass = mdwfShamirKernelMass(controlM5);
    ShamirForward shamirForward(baseGauge, shamirCoeff, controlMass, csw, "MDWF_mobius_storage_shamir_forward");
    ShamirAdjoint shamirAdjoint(baseGauge, shamirCoeff, controlMass, csw, "MDWF_mobius_storage_shamir_adjoint");
    ShamirNormal shamirNormal(commBase, shamirForward, shamirAdjoint, "MDWF_mobius_storage_shamir_normal");
    ShamirWorkspace shamirWorkspace;
    shamirWorkspace.prepare(shamirNormal, shamirForward, field, forceCoefficients, controlMaxIter, precision,
                            "MDWF_mobius_storage_shamir_workspace");

    HostGauge b5OneDestination(commBase, "MDWF_mobius_storage_b5_one_destination");
    HostGauge shamirDestination(commBase, "MDWF_mobius_storage_shamir_destination");
    zero(b5OneDestination);
    zero(shamirDestination);
    const StorageResult b5OneResult = overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
        b5OneDestination, gaugeHost, b5OneView, forceCoefficients, csw, commBase, "MDWF_mobius_storage_b5_one");
    const StorageResult shamirResult = overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
        shamirDestination, gaugeHost, shamirWorkspace, forceCoefficients, csw, commBase,
        "MDWF_mobius_storage_shamir");

    const double b5OneOutputDiff = mdwfMobiusMaxDestinationDiff<HaloDepth>(b5OneDestination, shamirDestination);
    const double b5OneRawWilsonDiff = mdwfMobiusMaxRawDiff(b5OneResult.raw_wilson, shamirResult.raw_wilson);
    const double b5OneRawCloverDiff = mdwfMobiusMaxRawDiff(b5OneResult.raw_clover, shamirResult.raw_clover);
    const bool b5OnePassed = b5OneWorkspace.converged() && shamirWorkspace.converged()
                             && b5OneOutputDiff <= b5OneTolerance
                             && b5OneRawWilsonDiff <= b5OneTolerance
                             && b5OneRawCloverDiff <= b5OneTolerance;
    rootLogger.info("MDWF Mobius all-link storage b5 = 1 regression (M5 = -2): outputMaxDiff = ", b5OneOutputDiff,
                    ", rawWilsonMaxDiff = ", b5OneRawWilsonDiff, ", rawCloverMaxDiff = ", b5OneRawCloverDiff,
                    ", mobiusStoredDirectional = ", mdwfMobiusStoredDirectionalDerivative<HaloDepth>(b5OneDestination),
                    ", shamirStoredDirectional = ", mdwfMobiusStoredDirectionalDerivative<HaloDepth>(shamirDestination),
                    ", passed = ", b5OnePassed);
    allPassed = allPassed && b5OnePassed;

    // --- Parts 2, 3, 4, 5: physical-like, b5 = 1.5. ---
    MobiusForward physicalForward(baseGauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_storage_physical_forward");
    MobiusAdjoint physicalAdjoint(baseGauge, physicalM5, mf, genericB5, csw, "MDWF_mobius_storage_physical_adjoint");
    MobiusNormal physicalNormal(commBase, physicalForward, physicalAdjoint, "MDWF_mobius_storage_physical_normal");
    Evaluator physicalEvaluator(commBase, field, actionCoefficients, physicalM5, mf, genericB5, csw,
                                physicalMaxIter, precision);
    const bool physicalPassed = runCase("physical-like", physicalForward, physicalNormal, physicalMaxIter,
                                        {1e-4}, physicalEvaluator, physicalTolerance);
    allPassed = allPassed && physicalPassed;

    if (!allPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius all-link force storage test failed: control passed = ", controlPassed,
            ", b5 = 1 regression passed = ", b5OnePassed,
            ", physical-like passed = ", physicalPassed,
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF Mobius all-link force storage test passed with Ls = ", Ls, ", c_sw = ", csw,
                    ", links = ", expectedLinks, ", terms = ", terms);
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

        runMDWFMobiusAllLinkForceStorageTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
