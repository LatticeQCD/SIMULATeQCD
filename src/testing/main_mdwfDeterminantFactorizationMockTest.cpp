/*
 * Determinant metadata and commuting scalar/diagonal algebra only.
 * No gauge field, physical MDWF operator, CG, rational approximation, force,
 * RNG heatbath, MPI ownership, or RHMC/HMC integration is tested here.
 */
#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDeterminantFactorMetadata.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool valid, const std::string &label) {
    if (!valid) {
        throw std::runtime_error("MDWF determinant factorization mock: " + label);
    }
}

double requireClose(double value, double expected, const std::string &label) {
    require(std::isfinite(value) && std::isfinite(expected), label + " is nonfinite");
    const double difference = std::abs(value - expected);
    require(difference <= 2e-13 * std::max(1.0, std::abs(expected)), label + " differs");
    return difference;
}

template<class Function>
void requireRejected(Function function, const std::string &label, size_t &count) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        ++count;
        return;
    }
    throw std::runtime_error("MDWF determinant metadata accepted " + label);
}

MDWFMassLadderMetadata makeLadder(const std::vector<double> &masses, double exponent) {
    require(masses.size() >= 2, "test ladder construction");
    MDWFMassLadderMetadata ladder{
        "synthetic-sector", "positive-diagonal-mock-v1", masses.front(), masses.back(),
        exponent, {}
    };
    for (size_t i = 0; i + 1 < masses.size(); ++i) {
        ladder.factors.push_back({ladder.sector_id, "factor-" + std::to_string(i),
            ladder.common_operator_identity, masses[i], masses[i + 1], exponent,
            MDWFDeterminantFactorRepresentation::DiagonalMockOnly});
    }
    return ladder;
}

// Arbitrary positive mock eigenvalues, not a domain-wall mass formula.
std::array<double, 4> normalEigenvalues(double mass) {
    return {{2.0 + 0.7 * mass, 3.0 + 1.1 * mass,
             5.0 + 0.2 * mass, 7.0 + 2.0 * mass}};
}

struct Diagnostics {
    size_t ladders = 0;
    size_t identityFactors = 0;
    double maxTelescopeDifference = 0.0;
    double maxHeatbathActionDifference = 0.0;
};

void checkLadder(const MDWFMassLadderMetadata &ladder, MDWFMassLadderPolicy policy,
                 Diagnostics &diagnostics) {
    validateMDWFMassLadderMetadata(ladder, policy);
    const auto physical = normalEigenvalues(ladder.physical_boundary_mass);
    const auto pv = normalEigenvalues(ladder.pv_boundary_mass);
    const std::array<std::complex<double>, 4> noise{{
        {0.3, -0.7}, {1.2, 0.4}, {-0.6, 0.2}, {0.8, -1.1}
    }};
    double noiseNorm = 0.0;
    for (const auto &component : noise) noiseNorm += std::norm(component);
    require(noiseNorm > 0.0, "nonzero deterministic noise");

    std::array<double, 4> products{{1.0, 1.0, 1.0, 1.0}};
    double scalarProduct = 1.0;
    double determinantProduct = 1.0;
    double logDeterminantSum = 0.0;
    size_t pvDenominators = 0;
    for (const auto &factor : ladder.factors) {
        const auto numerator = normalEigenvalues(factor.numerator_boundary_mass);
        const auto denominator = normalEigenvalues(factor.denominator_boundary_mass);
        const double p = factor.determinant_exponent;
        double numeratorDeterminant = 1.0;
        double denominatorDeterminant = 1.0;
        double action = 0.0;
        const bool identity = factor.numerator_boundary_mass == factor.denominator_boundary_mass;
        if (identity) ++diagnostics.identityFactors;
        if (factor.denominator_boundary_mass == ladder.pv_boundary_mass) ++pvDenominators;
        if (policy == MDWFMassLadderPolicy::StrictIncreasing) {
            require(factor.numerator_boundary_mass != ladder.pv_boundary_mass,
                    "PV must not reappear in a numerator");
        }
        for (size_t j = 0; j < products.size(); ++j) {
            const double ratio = numerator[j] / denominator[j];
            require(ratio > 0.0 && std::isfinite(ratio), "positive diagonal ratio");
            products[j] *= std::pow(ratio, p);
            numeratorDeterminant *= numerator[j];
            denominatorDeterminant *= denominator[j];
            logDeterminantSum += p * (std::log(numerator[j]) - std::log(denominator[j]));
            // Exact commuting algebra for C^(p/2) and C^(-p), not rational/CG code.
            const auto phi = std::pow(ratio, p / 2.0) * noise[j];
            action += std::norm(phi) * std::pow(ratio, -p);
            if (identity) {
                requireClose(ratio, 1.0, "identity ratio");
                requireClose(std::abs(phi - noise[j]), 0.0, "identity heatbath");
            }
        }
        scalarProduct *= std::pow(numerator[0] / denominator[0], p);
        determinantProduct *= std::pow(numeratorDeterminant / denominatorDeterminant, p);
        diagnostics.maxHeatbathActionDifference = std::max(
            diagnostics.maxHeatbathActionDifference,
            requireClose(action, noiseNorm, "generated pseudofermion action"));
        // Equal masses imply constant Gaussian action, NOT zero action.
        if (identity) require(action > 0.0, "identity action must remain nonzero");
    }
    require(pvDenominators == 1, "PV endpoint counted exactly once in tested ladders");
    const double p = ladder.determinant_exponent;
    double endpointLogDeterminant = 0.0;
    double physicalDeterminant = 1.0;
    double pvDeterminant = 1.0;
    auto compare = [&](double value, double expected, const std::string &label) {
        diagnostics.maxTelescopeDifference = std::max(diagnostics.maxTelescopeDifference,
                                                       requireClose(value, expected, label));
    };
    compare(scalarProduct, std::pow(physical[0] / pv[0], p), "scalar endpoints");
    for (size_t j = 0; j < products.size(); ++j) {
        compare(products[j], std::pow(physical[j] / pv[j], p), "diagonal endpoints");
        endpointLogDeterminant += p * (std::log(physical[j]) - std::log(pv[j]));
        physicalDeterminant *= physical[j];
        pvDeterminant *= pv[j];
    }
    compare(logDeterminantSum, endpointLogDeterminant, "log determinant endpoints");
    compare(determinantProduct, std::pow(physicalDeterminant / pvDeterminant, p),
            "determinant weight endpoints");
    compare(determinantProduct, std::exp(endpointLogDeterminant), "determinant log/product");
    if (ladder.physical_boundary_mass < ladder.pv_boundary_mass) {
        require(determinantProduct < 1.0, "numerator/denominator orientation");
    } else {
        compare(determinantProduct, 1.0, "identity determinant weight");
    }
    ++diagnostics.ladders;
}

void runTest() {
    Diagnostics diagnostics;
    for (double p : {0.5, 1.0, 1.5}) {
        checkLadder(makeLadder({0.05, 1.0}, p), MDWFMassLadderPolicy::StrictIncreasing, diagnostics);
        checkLadder(makeLadder({0.05, 0.2, 0.6, 1.0}, p),
                    MDWFMassLadderPolicy::StrictIncreasing, diagnostics);
    }
    size_t rejected = 0;
    const auto identity = makeLadder({0.4, 0.4}, 0.5);
    const auto embeddedIdentity = makeLadder({0.05, 0.2, 0.2, 1.0}, 1.0);
    requireRejected([&] { validateMDWFMassLadderMetadata(identity); }, "strict identity", rejected);
    requireRejected([&] { validateMDWFMassLadderMetadata(embeddedIdentity); },
                    "strict embedded identity", rejected);
    checkLadder(identity, MDWFMassLadderPolicy::AllowIdentityFactorsForTests, diagnostics);
    checkLadder(embeddedIdentity, MDWFMassLadderPolicy::AllowIdentityFactorsForTests, diagnostics);

    const auto baseline = makeLadder({0.05, 0.2, 0.6, 1.0}, 1.0);
    auto rejectMutation = [&](auto mutation, const std::string &label) {
        auto invalid = baseline;
        mutation(invalid);
        requireRejected([&] { validateMDWFMassLadderMetadata(invalid); }, label, rejected);
    };
    rejectMutation([](auto &x) { x.factors.clear(); }, "empty ladder");
    rejectMutation([](auto &x) { x.sector_id.clear(); }, "empty sector");
    rejectMutation([](auto &x) { x.common_operator_identity = " \t"; }, "blank operator label");
    rejectMutation([](auto &x) { x.factors[0].factor_id.clear(); }, "empty factor identifier");
    rejectMutation([](auto &x) { x.factors[0].sector_id = "other"; }, "sector mismatch");
    rejectMutation([](auto &x) { x.factors[0].common_operator_identity = "other"; }, "operator mismatch");
    rejectMutation([](auto &x) { x.factors[1].factor_id = x.factors[0].factor_id; }, "duplicate factor");
    rejectMutation([](auto &x) { x.factors[1].numerator_boundary_mass += 0.01; }, "disconnected ladder");
    rejectMutation([](auto &x) { x.factors[1].numerator_boundary_mass =
        std::nextafter(x.factors[1].numerator_boundary_mass, 1.0); }, "one-ULP ladder gap");
    rejectMutation([](auto &x) { x.physical_boundary_mass = 0.04; }, "wrong physical endpoint");
    rejectMutation([](auto &x) { x.pv_boundary_mass = 1.1; }, "wrong PV endpoint");
    rejectMutation([](auto &x) { x.factors[0].numerator_boundary_mass = 0.3; }, "reversed factor");
    rejectMutation([](auto &x) { x.physical_boundary_mass = 2.0; }, "reversed endpoints");
    rejectMutation([](auto &x) { x.factors[0].numerator_boundary_mass =
        std::numeric_limits<double>::quiet_NaN(); }, "NaN numerator");
    rejectMutation([](auto &x) { x.factors[0].denominator_boundary_mass =
        std::numeric_limits<double>::infinity(); }, "infinite denominator");
    rejectMutation([](auto &x) { x.pv_boundary_mass =
        std::numeric_limits<double>::infinity(); }, "infinite PV endpoint");
    rejectMutation([](auto &x) { x.physical_boundary_mass =
        std::numeric_limits<double>::quiet_NaN(); }, "NaN physical endpoint");
    rejectMutation([](auto &x) { x.determinant_exponent = 0.0; }, "zero sector exponent");
    rejectMutation([](auto &x) { x.factors[0].determinant_exponent = -0.5; }, "negative factor exponent");
    rejectMutation([](auto &x) { x.factors[0].determinant_exponent = 0.5; }, "factor exponent mismatch");
    rejectMutation([](auto &x) { x.factors[0].determinant_exponent =
        std::numeric_limits<double>::quiet_NaN(); }, "NaN factor exponent");
    rejectMutation([](auto &x) { x.determinant_exponent =
        std::numeric_limits<double>::infinity(); }, "infinite sector exponent");
    rejectMutation([](auto &x) { x.factors[0].representation =
        static_cast<MDWFDeterminantFactorRepresentation>(99); }, "unknown representation");
    requireRejected([&] { validateMDWFMassLadderMetadata(baseline,
        static_cast<MDWFMassLadderPolicy>(99)); }, "unknown policy", rejected);
    requireRejected([&] { validateMDWFMassLadderMetadata(makeLadder({0.6, 0.4}, 1.0),
        MDWFMassLadderPolicy::AllowIdentityFactorsForTests); }, "reversed relaxed ladder", rejected);
    requireRejected([] { validateMDWFDeterminantFactorMetadata(MDWFDeterminantFactorMetadata{}); },
                    "default factor", rejected);
    requireRejected([] { validateMDWFMassLadderMetadata(MDWFMassLadderMetadata{}); },
                    "default ladder", rejected);
    require(diagnostics.ladders == 8 && diagnostics.identityFactors == 2 && rejected == 29,
            "expected test coverage");

    rootLogger.info("MDWF determinant factorization mock test passed with ladders = ",
        diagnostics.ladders, ", diagonalComponents = 4, testedExponents = [0.5, 1, 1.5]",
        ", identityFactors = ", diagnostics.identityFactors, ", rejectedMetadataCases = ", rejected,
        ", maxTelescopeAbsDiff = ", diagnostics.maxTelescopeDifference,
        ", maxHeatbathActionAbsDiff = ", diagnostics.maxHeatbathActionDifference,
        ", representation = diagonal-mock-only; physical operator unchanged");
    std::cout << "MDWF determinant factorization mock test passed" << std::endl;
}
} // namespace

int main() {
    try {
        stdLogger.setVerbosity(INFO);
        runTest();
        return 0;
    } catch (const std::exception &error) {
        rootLogger.error(error.what());
        return -1;
    }
}
