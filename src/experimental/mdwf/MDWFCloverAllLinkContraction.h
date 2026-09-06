/*
 * MDWF test-only all-link clover contraction helpers.
 *
 * These helpers mirror the validated selected-link clover path derivative,
 * attribute each active plaquette factor to its periodic physical bulk link,
 * and provide a selected-link left-oriented raw matrix for the isolated
 * projection test.  They do not define a production force field, ipdot
 * convention, or HMC sign.
 */

#pragma once

#include "MDWFAllLinkRandomDirection.h"

#include <array>
#include <stdexcept>
#include <vector>

template<class floatT>
struct MDWFAllLinkCloverPathFactor {
    gSiteMu site_mu;
    bool dagger_link;
};

template<class floatT>
SU3<floatT> mdwfAllLinkCloverPathFactorValue(
    SU3Accessor<floatT, R18> gauge_acc,
    const MDWFAllLinkCloverPathFactor<floatT> &factor) {

    if (factor.dagger_link) {
        return gauge_acc.getLinkDagger(factor.site_mu);
    }
    return gauge_acc.getLink(factor.site_mu);
}

template<class floatT, size_t HaloDepth>
SU3<floatT> mdwfAllLinkCloverPathFactorDerivative(
    SU3Accessor<floatT, R18> gauge_acc,
    const MDWFAllLinkCloverPathFactor<floatT> &factor) {

    const SU3<floatT> link = gauge_acc.getLink(factor.site_mu);
    const SU3<floatT> direction
        = mdwfAllLinkDeterministicDirection<floatT, HaloDepth>(factor.site_mu);
    if (factor.dagger_link) {
        return static_cast<floatT>(-1.0) * dagger(link) * direction;
    }
    return direction * link;
}

inline void mdwfAllLinkCloverAddFmunuDerivativeToBlocks(
    const SU3<double> &dFmunu,
    int mu,
    int nu,
    Matrix6x6<double> &upper,
    Matrix6x6<double> &lower) {

    const COMPLEX(double) ii(0.0, 1.0);

    for (int colorRow = 0; colorRow < 3; colorRow++) {
        for (int colorColumn = 0; colorColumn < 3; colorColumn++) {
            const COMPLEX(double) value = dFmunu(colorRow, colorColumn);

            if (mu == 0 && nu == 1) {
                upper.val[colorRow][colorColumn] += value;
                upper.val[colorRow + 3][colorColumn + 3] -= value;
                lower.val[colorRow][colorColumn] += value;
                lower.val[colorRow + 3][colorColumn + 3] -= value;
            } else if (mu == 0 && nu == 2) {
                upper.val[colorRow][colorColumn + 3] += -ii * value;
                upper.val[colorRow + 3][colorColumn] += ii * value;
                lower.val[colorRow][colorColumn + 3] += -ii * value;
                lower.val[colorRow + 3][colorColumn] += ii * value;
            } else if (mu == 0 && nu == 3) {
                upper.val[colorRow][colorColumn + 3] += -value;
                upper.val[colorRow + 3][colorColumn] += -value;
                lower.val[colorRow][colorColumn + 3] += value;
                lower.val[colorRow + 3][colorColumn] += value;
            } else if (mu == 1 && nu == 2) {
                upper.val[colorRow][colorColumn + 3] += value;
                upper.val[colorRow + 3][colorColumn] += value;
                lower.val[colorRow][colorColumn + 3] += value;
                lower.val[colorRow + 3][colorColumn] += value;
            } else if (mu == 1 && nu == 3) {
                upper.val[colorRow][colorColumn + 3] += -ii * value;
                upper.val[colorRow + 3][colorColumn] += ii * value;
                lower.val[colorRow][colorColumn + 3] += ii * value;
                lower.val[colorRow + 3][colorColumn] += -ii * value;
            } else if (mu == 2 && nu == 3) {
                upper.val[colorRow][colorColumn] += -value;
                upper.val[colorRow + 3][colorColumn + 3] += value;
                lower.val[colorRow][colorColumn] += value;
                lower.val[colorRow + 3][colorColumn + 3] += -value;
            }
        }
    }
}

inline void mdwfAllLinkCloverScaleMatrix(
    Matrix6x6<double> &matrix,
    double factor) {

    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 6; column++) {
            matrix.val[row][column] *= factor;
        }
    }
}

inline Vect12<double> mdwfAllLinkCloverApplyDerivative(
    Matrix6x6<double> &upper,
    Matrix6x6<double> &lower,
    const Vect12<double> &spinor) {

    Vect18<double> upperStored = upper.ConvertHermitianToVect18();
    Vect18<double> lowerStored = lower.ConvertHermitianToVect18();
    Matrix6x6<double> storedUpper(upperStored);
    Matrix6x6<double> storedLower(lowerStored);

    Vect12<double> out = storedUpper.MatrixXVect12UpDown(spinor, 0);
    out = storedLower.MatrixXVect12UpDown(out, 1);
    return out;
}

template<size_t HaloDepth>
size_t mdwfAllLinkCloverPhysicalLinkIndex(const gSiteMu &site_mu) {
    typedef GIndexer<All, HaloDepth> GInd;
    const sitexyzt global = GInd::getLatData().globalPos(site_mu.coord);
    const gSite physicalSite = GInd::getSite(
        global.x, global.y, global.z, global.t);
    return physicalSite.isite * 4 + site_mu.mu;
}

template<size_t HaloDepth, size_t Ls>
void mdwfAllLinkCloverAccumulatePath(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &clover_site,
    int mu,
    int nu,
    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> &path,
    double csw,
    double rational_numerator,
    std::vector<double> &clover_link_derivatives) {

    typedef GIndexer<All, HaloDepth> GInd;

    for (size_t active = 0; active < path.size(); active++) {
        SU3<double> dQ = su3_one<double>();
        for (size_t factor = 0; factor < path.size(); factor++) {
            if (factor == active) {
                dQ *= mdwfAllLinkCloverPathFactorDerivative<double, HaloDepth>(
                    gauge_acc, path[factor]);
            } else {
                dQ *= mdwfAllLinkCloverPathFactorValue(
                    gauge_acc, path[factor]);
            }
        }

        SU3<double> dF
            = (COMPLEX(double)(0.0, -1.0) / 8.0) * (dQ - dagger(dQ));
        dF = dF - (1.0 / 3.0) * tr_c(dF) * su3_one<double>();

        Matrix6x6<double> upper;
        Matrix6x6<double> lower;
        mdwfAllLinkCloverAddFmunuDerivativeToBlocks(
            dF, mu, nu, upper, lower);
        mdwfAllLinkCloverScaleMatrix(upper, -0.5 * csw);
        mdwfAllLinkCloverScaleMatrix(lower, -0.5 * csw);

        COMPLEX(double) contraction(0.0, 0.0);
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack siteStack = GInd::getSiteStack(
                clover_site, stack);
            const Vect12<double> dCloverChi
                = mdwfAllLinkCloverApplyDerivative(
                    upper, lower, chi_acc.getElement(siteStack));
            contraction += eta_acc.getElement(siteStack) * dCloverChi;
        }

        const size_t physicalLink
            = mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(
                path[active].site_mu);
        clover_link_derivatives[physicalLink]
            += -2.0 * rational_numerator * real(contraction);
    }
}

template<size_t HaloDepth, size_t Ls>
void mdwfAllLinkCloverAccumulateSite(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &site,
    double csw,
    double rational_numerator,
    std::vector<double> &clover_link_derivatives) {

    typedef GIndexer<All, HaloDepth> GInd;

    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            const std::array<MDWFAllLinkCloverPathFactor<double>, 4> pathP = {{
                {GInd::getSiteMu(site, mu), false},
                {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
                {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
                {GInd::getSiteMu(site, nu), true}
            }};
            const std::array<MDWFAllLinkCloverPathFactor<double>, 4> pathQ = {{
                {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
                {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
                {GInd::getSiteMu(GInd::site_up_dn(site, mu, nu), nu), false},
                {GInd::getSiteMu(site, mu), true}
            }};
            const std::array<MDWFAllLinkCloverPathFactor<double>, 4> pathR = {{
                {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
                {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), nu), true},
                {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), mu), false},
                {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}
            }};
            const std::array<MDWFAllLinkCloverPathFactor<double>, 4> pathS = {{
                {GInd::getSiteMu(site, nu), false},
                {GInd::getSiteMu(GInd::site_up_dn(site, nu, mu), mu), true},
                {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
                {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}
            }};

            mdwfAllLinkCloverAccumulatePath<HaloDepth, Ls>(
                gauge_acc, chi_acc, eta_acc, site, mu, nu, pathP, csw,
                rational_numerator, clover_link_derivatives);
            mdwfAllLinkCloverAccumulatePath<HaloDepth, Ls>(
                gauge_acc, chi_acc, eta_acc, site, mu, nu, pathQ, csw,
                rational_numerator, clover_link_derivatives);
            mdwfAllLinkCloverAccumulatePath<HaloDepth, Ls>(
                gauge_acc, chi_acc, eta_acc, site, mu, nu, pathR, csw,
                rational_numerator, clover_link_derivatives);
            mdwfAllLinkCloverAccumulatePath<HaloDepth, Ls>(
                gauge_acc, chi_acc, eta_acc, site, mu, nu, pathS, csw,
                rational_numerator, clover_link_derivatives);
        }
    }
}

template<class floatT>
SU3<floatT> mdwfSelectedLinkCloverPathFactorDerivative(
    SU3Accessor<floatT, R18> gauge_acc,
    const MDWFAllLinkCloverPathFactor<floatT> &factor,
    const SU3<floatT> &direction,
    MDWFFiniteDifferenceMultiplicationSide multiplication_side) {

    const SU3<floatT> link = gauge_acc.getLink(factor.site_mu);
    if (factor.dagger_link) {
        if (multiplication_side
            == MDWFFiniteDifferenceMultiplicationSide::Right) {
            return static_cast<floatT>(-1.0) * direction * dagger(link);
        }
        return static_cast<floatT>(-1.0) * dagger(link) * direction;
    }
    if (multiplication_side
        == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return link * direction;
    }
    return direction * link;
}

template<size_t HaloDepth, size_t Ls>
double mdwfSelectedLinkCloverContractionPath(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &clover_site,
    int mu,
    int nu,
    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> &path,
    const gSiteMu &selected_link,
    const SU3<double> &direction,
    MDWFFiniteDifferenceMultiplicationSide multiplication_side,
    double csw) {

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t selected_physical_link
        = mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(selected_link);
    COMPLEX(double) contraction(0.0, 0.0);

    for (size_t active = 0; active < path.size(); active++) {
        if (mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(
                path[active].site_mu)
            != selected_physical_link) {
            continue;
        }

        SU3<double> dQ = su3_one<double>();
        for (size_t factor = 0; factor < path.size(); factor++) {
            if (factor == active) {
                dQ *= mdwfSelectedLinkCloverPathFactorDerivative(
                    gauge_acc, path[factor], direction, multiplication_side);
            } else {
                dQ *= mdwfAllLinkCloverPathFactorValue(
                    gauge_acc, path[factor]);
            }
        }

        SU3<double> dF
            = (COMPLEX(double)(0.0, -1.0) / 8.0) * (dQ - dagger(dQ));
        dF = dF - (1.0 / 3.0) * tr_c(dF) * su3_one<double>();

        Matrix6x6<double> upper;
        Matrix6x6<double> lower;
        mdwfAllLinkCloverAddFmunuDerivativeToBlocks(
            dF, mu, nu, upper, lower);
        mdwfAllLinkCloverScaleMatrix(upper, -0.5 * csw);
        mdwfAllLinkCloverScaleMatrix(lower, -0.5 * csw);

        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site_stack
                = GInd::getSiteStack(clover_site, stack);
            const Vect12<double> d_clover_chi
                = mdwfAllLinkCloverApplyDerivative(
                    upper, lower, chi_acc.getElement(site_stack));
            contraction += eta_acc.getElement(site_stack) * d_clover_chi;
        }
    }

    return real(contraction);
}

template<size_t HaloDepth, size_t Ls>
double mdwfSelectedLinkCloverContractionTerm(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSiteMu &selected_link,
    const SU3<double> &direction,
    MDWFFiniteDifferenceMultiplicationSide multiplication_side,
    double csw) {

    typedef GIndexer<All, HaloDepth> GInd;
    double contraction = 0.0;
    const size_t volume = GInd::getLatData().vol4;

    for (size_t site_index = 0; site_index < volume; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_p = {{
                    {GInd::getSiteMu(site, mu), false},
                    {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
                    {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
                    {GInd::getSiteMu(site, nu), true}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_q = {{
                    {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
                    {GInd::getSiteMu(
                        GInd::site_up_dn(site, mu, nu), nu), false},
                    {GInd::getSiteMu(site, mu), true}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_r = {{
                    {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
                    {GInd::getSiteMu(
                        GInd::site_dn_dn(site, mu, nu), nu), true},
                    {GInd::getSiteMu(
                        GInd::site_dn_dn(site, mu, nu), mu), false},
                    {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_s = {{
                    {GInd::getSiteMu(site, nu), false},
                    {GInd::getSiteMu(
                        GInd::site_up_dn(site, nu, mu), mu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}
                }};

                contraction
                    += mdwfSelectedLinkCloverContractionPath<
                        HaloDepth, Ls>(
                        gauge_acc, chi_acc, eta_acc, site, mu, nu,
                        path_p, selected_link, direction,
                        multiplication_side, csw);
                contraction
                    += mdwfSelectedLinkCloverContractionPath<
                        HaloDepth, Ls>(
                        gauge_acc, chi_acc, eta_acc, site, mu, nu,
                        path_q, selected_link, direction,
                        multiplication_side, csw);
                contraction
                    += mdwfSelectedLinkCloverContractionPath<
                        HaloDepth, Ls>(
                        gauge_acc, chi_acc, eta_acc, site, mu, nu,
                        path_r, selected_link, direction,
                        multiplication_side, csw);
                contraction
                    += mdwfSelectedLinkCloverContractionPath<
                        HaloDepth, Ls>(
                        gauge_acc, chi_acc, eta_acc, site, mu, nu,
                        path_s, selected_link, direction,
                        multiplication_side, csw);
            }
        }
    }

    return contraction;
}

inline SU3<double> mdwfCloverHermitianSensitivityBasis(
    size_t basis_index) {

    const COMPLEX(double) zero(0.0, 0.0);
    const COMPLEX(double) one(1.0, 0.0);
    const COMPLEX(double) plus_i(0.0, 1.0);
    const COMPLEX(double) minus_i(0.0, -1.0);

    if (basis_index == 0) {
        return SU3<double>(
            one, zero, zero,
            zero, zero, zero,
            zero, zero, zero);
    }
    if (basis_index == 1) {
        return SU3<double>(
            zero, zero, zero,
            zero, one, zero,
            zero, zero, zero);
    }
    if (basis_index == 2) {
        return SU3<double>(
            zero, zero, zero,
            zero, zero, zero,
            zero, zero, one);
    }
    if (basis_index == 3) {
        return SU3<double>(
            zero, one, zero,
            one, zero, zero,
            zero, zero, zero);
    }
    if (basis_index == 4) {
        return SU3<double>(
            zero, plus_i, zero,
            minus_i, zero, zero,
            zero, zero, zero);
    }
    if (basis_index == 5) {
        return SU3<double>(
            zero, zero, one,
            zero, zero, zero,
            one, zero, zero);
    }
    if (basis_index == 6) {
        return SU3<double>(
            zero, zero, plus_i,
            zero, zero, zero,
            minus_i, zero, zero);
    }
    if (basis_index == 7) {
        return SU3<double>(
            zero, zero, zero,
            zero, zero, one,
            zero, one, zero);
    }
    if (basis_index == 8) {
        return SU3<double>(
            zero, zero, zero,
            zero, zero, plus_i,
            zero, minus_i, zero);
    }

    throw std::runtime_error(stdLogger.fatal(
        "MDWF clover Hermitian sensitivity basis index must be in [0, 8], "
        "got ", basis_index));
}

template<size_t HaloDepth, size_t Ls>
double mdwfCloverFieldStrengthContraction(
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &clover_site,
    int mu,
    int nu,
    const SU3<double> &field_strength,
    double csw) {

    typedef GIndexer<All, HaloDepth> GInd;

    Matrix6x6<double> upper;
    Matrix6x6<double> lower;
    mdwfAllLinkCloverAddFmunuDerivativeToBlocks(
        field_strength, mu, nu, upper, lower);
    mdwfAllLinkCloverScaleMatrix(upper, -0.5 * csw);
    mdwfAllLinkCloverScaleMatrix(lower, -0.5 * csw);

    COMPLEX(double) contraction(0.0, 0.0);
    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack site_stack
            = GInd::getSiteStack(clover_site, stack);
        const Vect12<double> clover_chi
            = mdwfAllLinkCloverApplyDerivative(
                upper, lower, chi_acc.getElement(site_stack));
        contraction += eta_acc.getElement(site_stack) * clover_chi;
    }
    return real(contraction);
}

template<size_t HaloDepth, size_t Ls>
SU3<double> mdwfCloverFieldStrengthSensitivity(
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &clover_site,
    int mu,
    int nu,
    double csw) {

    SU3<double> sensitivity = su3_zero<double>();
    for (size_t basis_index = 0; basis_index < 9; basis_index++) {
        const SU3<double> basis
            = mdwfCloverHermitianSensitivityBasis(basis_index);
        const double basis_norm
            = static_cast<double>(real(tr_c(basis, basis)));
        const double response
            = mdwfCloverFieldStrengthContraction<HaloDepth, Ls>(
                chi_acc, eta_acc, clover_site, mu, nu, basis, csw);
        sensitivity += (response / basis_norm) * basis;
    }

    return sensitivity
           - (1.0 / 3.0) * tr_c(sensitivity) * su3_one<double>();
}

template<size_t HaloDepth>
bool mdwfCloverPathContainsPhysicalLink(
    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> &path,
    size_t selected_physical_link) {

    for (const auto &factor : path) {
        if (mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(factor.site_mu)
            == selected_physical_link) {
            return true;
        }
    }
    return false;
}

template<size_t HaloDepth>
SU3<double> mdwfSelectedLinkCloverLeftRawContractionPath(
    SU3Accessor<double, R18> gauge_acc,
    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> &path,
    const gSiteMu &selected_link,
    const SU3<double> &field_strength_sensitivity) {

    /*
     * For dQ = A H B and dF = (-i/8)(dQ - dQ^dagger), Hermitian
     * field_strength_sensitivity Y gives
     *
     *   Re tr(dF Y) = Re tr(H B (-iY/4) A).
     *
     * The returned matrix is therefore raw and left-oriented; no TA()
     * projection or rational weight is applied here.
     */
    const size_t selected_physical_link
        = mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(selected_link);
    const SU3<double> field_strength_weight
        = COMPLEX(double)(0.0, -0.25) * field_strength_sensitivity;
    SU3<double> raw_matrix = su3_zero<double>();

    for (size_t active = 0; active < path.size(); active++) {
        if (mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(
                path[active].site_mu)
            != selected_physical_link) {
            continue;
        }

        SU3<double> prefix = su3_one<double>();
        for (size_t factor = 0; factor < active; factor++) {
            prefix *= mdwfAllLinkCloverPathFactorValue(
                gauge_acc, path[factor]);
        }
        SU3<double> suffix = su3_one<double>();
        for (size_t factor = active + 1; factor < path.size(); factor++) {
            suffix *= mdwfAllLinkCloverPathFactorValue(
                gauge_acc, path[factor]);
        }

        const SU3<double> link
            = gauge_acc.getLink(path[active].site_mu);
        if (path[active].dagger_link) {
            const SU3<double> left_factor
                = static_cast<double>(-1.0) * prefix * dagger(link);
            raw_matrix += suffix * field_strength_weight * left_factor;
        } else {
            const SU3<double> right_factor = link * suffix;
            raw_matrix += right_factor * field_strength_weight * prefix;
        }
    }

    return raw_matrix;
}

template<size_t HaloDepth, size_t Ls>
SU3<double> mdwfSelectedLinkCloverLeftRawContractionMatrixTerm(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSiteMu &selected_link,
    double csw) {

    typedef GIndexer<All, HaloDepth> GInd;

    SU3<double> raw_matrix = su3_zero<double>();
    const size_t volume = GInd::getLatData().vol4;
    const size_t selected_physical_link
        = mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(selected_link);
    for (size_t site_index = 0; site_index < volume; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_p = {{
                    {GInd::getSiteMu(site, mu), false},
                    {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
                    {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
                    {GInd::getSiteMu(site, nu), true}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_q = {{
                    {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
                    {GInd::getSiteMu(
                        GInd::site_up_dn(site, mu, nu), nu), false},
                    {GInd::getSiteMu(site, mu), true}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_r = {{
                    {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
                    {GInd::getSiteMu(
                        GInd::site_dn_dn(site, mu, nu), nu), true},
                    {GInd::getSiteMu(
                        GInd::site_dn_dn(site, mu, nu), mu), false},
                    {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}
                }};
                const std::array<
                    MDWFAllLinkCloverPathFactor<double>, 4> path_s = {{
                    {GInd::getSiteMu(site, nu), false},
                    {GInd::getSiteMu(
                        GInd::site_up_dn(site, nu, mu), mu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
                    {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}
                }};

                if (!mdwfCloverPathContainsPhysicalLink<HaloDepth>(
                        path_p, selected_physical_link)
                    && !mdwfCloverPathContainsPhysicalLink<HaloDepth>(
                        path_q, selected_physical_link)
                    && !mdwfCloverPathContainsPhysicalLink<HaloDepth>(
                        path_r, selected_physical_link)
                    && !mdwfCloverPathContainsPhysicalLink<HaloDepth>(
                        path_s, selected_physical_link)) {
                    continue;
                }

                const SU3<double> sensitivity
                    = mdwfCloverFieldStrengthSensitivity<HaloDepth, Ls>(
                        chi_acc, eta_acc, site, mu, nu, csw);
                raw_matrix
                    += mdwfSelectedLinkCloverLeftRawContractionPath<
                        HaloDepth>(
                        gauge_acc, path_p, selected_link, sensitivity);
                raw_matrix
                    += mdwfSelectedLinkCloverLeftRawContractionPath<
                        HaloDepth>(
                        gauge_acc, path_q, selected_link, sensitivity);
                raw_matrix
                    += mdwfSelectedLinkCloverLeftRawContractionPath<
                        HaloDepth>(
                        gauge_acc, path_r, selected_link, sensitivity);
                raw_matrix
                    += mdwfSelectedLinkCloverLeftRawContractionPath<
                        HaloDepth>(
                        gauge_acc, path_s, selected_link, sensitivity);
            }
        }
    }

    return raw_matrix;
}

template<size_t HaloDepth>
void mdwfAllLinkCloverAccumulateLeftRawContractionPath(
    SU3Accessor<double, R18> gauge_acc,
    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> &path,
    const SU3<double> &field_strength_sensitivity,
    double rational_weight,
    std::vector<SU3<double>> &raw_clover_matrices,
    std::vector<size_t> &path_additions) {

    /*
     * This is the all-link attribution of the selected-link derivation above.
     * Each active factor contributes its raw left-oriented matrix directly to
     * the periodic physical bulk link represented by that factor.  No TA()
     * projection is applied here.
     */
    const SU3<double> field_strength_weight
        = COMPLEX(double)(0.0, -0.25) * field_strength_sensitivity;

    for (size_t active = 0; active < path.size(); active++) {
        SU3<double> prefix = su3_one<double>();
        for (size_t factor = 0; factor < active; factor++) {
            prefix *= mdwfAllLinkCloverPathFactorValue(
                gauge_acc, path[factor]);
        }
        SU3<double> suffix = su3_one<double>();
        for (size_t factor = active + 1; factor < path.size(); factor++) {
            suffix *= mdwfAllLinkCloverPathFactorValue(
                gauge_acc, path[factor]);
        }

        const SU3<double> link
            = gauge_acc.getLink(path[active].site_mu);
        SU3<double> raw_contribution = su3_zero<double>();
        if (path[active].dagger_link) {
            const SU3<double> left_factor
                = static_cast<double>(-1.0) * prefix * dagger(link);
            raw_contribution
                = suffix * field_strength_weight * left_factor;
        } else {
            const SU3<double> right_factor = link * suffix;
            raw_contribution
                = right_factor * field_strength_weight * prefix;
        }

        const size_t physical_link
            = mdwfAllLinkCloverPhysicalLinkIndex<HaloDepth>(
                path[active].site_mu);
        raw_clover_matrices[physical_link]
            += rational_weight * raw_contribution;
        path_additions[physical_link]++;
    }
}

template<size_t HaloDepth, size_t Ls>
void mdwfAllLinkCloverAccumulateLeftRawContractionSite(
    SU3Accessor<double, R18> gauge_acc,
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    const gSite &site,
    double csw,
    double rational_weight,
    std::vector<SU3<double>> &raw_clover_matrices,
    std::vector<size_t> &path_additions) {

    typedef GIndexer<All, HaloDepth> GInd;

    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            const std::array<
                MDWFAllLinkCloverPathFactor<double>, 4> path_p = {{
                {GInd::getSiteMu(site, mu), false},
                {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
                {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
                {GInd::getSiteMu(site, nu), true}
            }};
            const std::array<
                MDWFAllLinkCloverPathFactor<double>, 4> path_q = {{
                {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
                {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
                {GInd::getSiteMu(
                    GInd::site_up_dn(site, mu, nu), nu), false},
                {GInd::getSiteMu(site, mu), true}
            }};
            const std::array<
                MDWFAllLinkCloverPathFactor<double>, 4> path_r = {{
                {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
                {GInd::getSiteMu(
                    GInd::site_dn_dn(site, mu, nu), nu), true},
                {GInd::getSiteMu(
                    GInd::site_dn_dn(site, mu, nu), mu), false},
                {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}
            }};
            const std::array<
                MDWFAllLinkCloverPathFactor<double>, 4> path_s = {{
                {GInd::getSiteMu(site, nu), false},
                {GInd::getSiteMu(
                    GInd::site_up_dn(site, nu, mu), mu), true},
                {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
                {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}
            }};

            const SU3<double> sensitivity
                = mdwfCloverFieldStrengthSensitivity<HaloDepth, Ls>(
                    chi_acc, eta_acc, site, mu, nu, csw);
            mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                gauge_acc, path_p, sensitivity, rational_weight,
                raw_clover_matrices, path_additions);
            mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                gauge_acc, path_q, sensitivity, rational_weight,
                raw_clover_matrices, path_additions);
            mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                gauge_acc, path_r, sensitivity, rational_weight,
                raw_clover_matrices, path_additions);
            mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                gauge_acc, path_s, sensitivity, rational_weight,
                raw_clover_matrices, path_additions);
        }
    }
}
