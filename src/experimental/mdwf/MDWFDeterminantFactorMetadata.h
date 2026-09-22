#pragma once

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only contract for [det N(a) / det N(b)]^p. These mass labels do
// not map to MDWFLinearOperator::setMass(), and the opaque common identity
// is not proof of physical/PV operator equivalence. No production ratio,
// flavor inference, coefficient selection, or integrator policy is provided.
enum class MDWFDeterminantFactorRepresentation {
    DiagonalMockOnly
};

enum class MDWFMassLadderPolicy {
    StrictIncreasing,
    AllowIdentityFactorsForTests
};

struct MDWFDeterminantFactorMetadata {
    std::string sector_id;
    std::string factor_id;
    std::string common_operator_identity;
    double numerator_boundary_mass = 0.0;
    double denominator_boundary_mass = 0.0;
    double determinant_exponent = 0.0; // Incomplete metadata must fail validation.
    MDWFDeterminantFactorRepresentation representation =
        MDWFDeterminantFactorRepresentation::DiagonalMockOnly;
};

struct MDWFMassLadderMetadata {
    std::string sector_id;
    std::string common_operator_identity;
    double physical_boundary_mass = 0.0;
    double pv_boundary_mass = 0.0;
    double determinant_exponent = 0.0;
    std::vector<MDWFDeterminantFactorMetadata> factors;
};

namespace mdwf_determinant_metadata_detail {
inline void require(bool valid, const std::string &message) {
    if (!valid) {
        throw std::invalid_argument("MDWF determinant metadata: " + message);
    }
}

inline bool hasLabel(const std::string &label) {
    return label.find_first_not_of(" \t\r\n") != std::string::npos;
}

inline void validatePolicy(MDWFMassLadderPolicy policy) {
    require(policy == MDWFMassLadderPolicy::StrictIncreasing
            || policy == MDWFMassLadderPolicy::AllowIdentityFactorsForTests,
            "unsupported mass-ladder policy");
}

inline void validateMassPair(double numerator, double denominator,
                             MDWFMassLadderPolicy policy) {
    require(std::isfinite(numerator) && std::isfinite(denominator),
            "mass endpoints must be finite");
    require(policy == MDWFMassLadderPolicy::StrictIncreasing
                ? numerator < denominator : numerator <= denominator,
            "mass endpoints violate the increasing-ladder policy");
}
} // namespace mdwf_determinant_metadata_detail

inline void validateMDWFDeterminantFactorMetadata(
    const MDWFDeterminantFactorMetadata &factor,
    MDWFMassLadderPolicy policy = MDWFMassLadderPolicy::StrictIncreasing) {
    using namespace mdwf_determinant_metadata_detail;
    validatePolicy(policy);
    require(hasLabel(factor.sector_id) && hasLabel(factor.factor_id)
            && hasLabel(factor.common_operator_identity), "factor labels must be nonempty");
    require(factor.representation == MDWFDeterminantFactorRepresentation::DiagonalMockOnly,
            "only the diagonal mock representation is supported");
    require(std::isfinite(factor.determinant_exponent)
            && factor.determinant_exponent > 0.0, "factor exponent must be finite and positive");
    validateMassPair(factor.numerator_boundary_mass, factor.denominator_boundary_mass, policy);
}

inline void validateMDWFMassLadderMetadata(
    const MDWFMassLadderMetadata &ladder,
    MDWFMassLadderPolicy policy = MDWFMassLadderPolicy::StrictIncreasing) {
    using namespace mdwf_determinant_metadata_detail;
    validatePolicy(policy);
    require(hasLabel(ladder.sector_id) && hasLabel(ladder.common_operator_identity),
            "ladder labels must be nonempty");
    require(std::isfinite(ladder.determinant_exponent)
            && ladder.determinant_exponent > 0.0, "ladder exponent must be finite and positive");
    validateMassPair(ladder.physical_boundary_mass, ladder.pv_boundary_mass, policy);
    require(!ladder.factors.empty(), "ladder must contain at least one factor");

    std::set<std::string> factorIds;
    double nextNumerator = ladder.physical_boundary_mass;
    for (const auto &factor : ladder.factors) {
        validateMDWFDeterminantFactorMetadata(factor, policy);
        require(factor.sector_id == ladder.sector_id, "factor belongs to a different sector");
        require(factor.common_operator_identity == ladder.common_operator_identity,
                "factor has a different common operator identity");
        require(factor.determinant_exponent == ladder.determinant_exponent,
                "factor exponent differs from the sector exponent");
        require(factorIds.insert(factor.factor_id).second, "duplicate factor identifier");
        // Exact metadata continuity: do not silently repair gaps with a tolerance.
        require(factor.numerator_boundary_mass == nextNumerator,
                "disconnected ladder or wrong physical endpoint");
        nextNumerator = factor.denominator_boundary_mass;
    }
    require(nextNumerator == ladder.pv_boundary_mass, "wrong PV endpoint");
}
