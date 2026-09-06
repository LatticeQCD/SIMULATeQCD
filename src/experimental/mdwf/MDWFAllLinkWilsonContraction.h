/*
 * MDWF test-only all-link Wilson contraction helper.
 *
 * This mirrors the validated Wilson hopping variation for an explicit
 * anti-Hermitian link direction and provides a selected-link, left-oriented
 * raw bilinear for the isolated projection test.  It does not define a
 * production force field, ipdot convention, or HMC sign.
 */

#pragma once

#include "MDWFAllLinkRandomDirection.h"

template<class floatT>
ColorVect<floatT> mdwfAllLinkWilsonGamma(
    uint8_t mu,
    const ColorVect<floatT> &spinor) {

    if (mu == 0) {
        return GammaXMultVec(spinor);
    }
    if (mu == 1) {
        return GammaYMultVec(spinor);
    }
    if (mu == 2) {
        return GammaZMultVec(spinor);
    }
    return GammaTMultVec(spinor);
}

template<class floatT>
COMPLEX(double) mdwfAllLinkWilsonColorVectDot(
    const ColorVect<floatT> &left,
    const ColorVect<floatT> &right) {

    COMPLEX(double) result(0.0, 0.0);
    for (size_t spin = 0; spin < 4; spin++) {
        result += left[spin] * right[spin];
    }
    return result;
}

template<size_t HaloDepth, size_t Ls>
double mdwfAllLinkWilsonContractionTerm(
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    SU3Accessor<double, R18> gauge_acc,
    const gSite &link_site,
    uint8_t mu,
    const SU3<double> &direction) {

    typedef GIndexer<All, HaloDepth> GInd;

    const SU3<double> link = gauge_acc.getLink(
        GInd::getSiteMu(link_site, mu));
    const SU3<double> dLink = direction * link;
    const SU3<double> dLinkDagger
        = static_cast<double>(-1.0) * dagger(link) * direction;
    const gSite forwardOutputSite = GInd::site_up(link_site, mu);
    COMPLEX(double) contraction(0.0, 0.0);

    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack outputForward = GInd::getSiteStack(link_site, stack);
        const gSiteStack inputForward = GInd::site_up(outputForward, mu);
        const gSiteStack outputBackward = GInd::getSiteStack(
            forwardOutputSite, stack);
        const gSiteStack inputBackward = GInd::site_dn(outputBackward, mu);

        Vect12<double> chiForwardVect = chi_acc.getElement(inputForward);
        Vect12<double> chiBackwardVect = chi_acc.getElement(inputBackward);
        Vect12<double> etaForwardVect = eta_acc.getElement(outputForward);
        Vect12<double> etaBackwardVect = eta_acc.getElement(outputBackward);

        const ColorVect<double> chiForward = convertVect12ToColorVect(
            chiForwardVect);
        const ColorVect<double> chiBackward = convertVect12ToColorVect(
            chiBackwardVect);
        const ColorVect<double> etaForward = convertVect12ToColorVect(
            etaForwardVect);
        const ColorVect<double> etaBackward = convertVect12ToColorVect(
            etaBackwardVect);

        const ColorVect<double> forwardHop = dLink * chiForward;
        const ColorVect<double> backwardHop = dLinkDagger * chiBackward;
        const ColorVect<double> dMForward
            = static_cast<double>(0.5)
              * (mdwfAllLinkWilsonGamma(mu, forwardHop) - forwardHop);
        const ColorVect<double> dMBackward
            = static_cast<double>(-0.5)
              * (backwardHop + mdwfAllLinkWilsonGamma(mu, backwardHop));

        contraction += mdwfAllLinkWilsonColorVectDot(
            etaForward, dMForward);
        contraction += mdwfAllLinkWilsonColorVectDot(
            etaBackward, dMBackward);
    }

    return real(contraction);
}

/*
 * Return the unprojected matrix B for one rational term and one link in the
 * left-variation convention
 *
 *   dU = H U,  dS(H) = Re tr(H B).
 *
 * No traceless anti-Hermitian projection, rational numerator, force sign, or
 * HMC convention is applied here.
 */
template<size_t HaloDepth, size_t Ls>
SU3<double> mdwfWilsonLeftRawContractionMatrixTerm(
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    SU3Accessor<double, R18> gauge_acc,
    const gSite &link_site,
    uint8_t mu) {

    typedef GIndexer<All, HaloDepth> GInd;

    const SU3<double> link = gauge_acc.getLink(
        GInd::getSiteMu(link_site, mu));
    const gSite forward_output_site = GInd::site_up(link_site, mu);
    SU3<double> raw_matrix = su3_zero<double>();

    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack output_forward
            = GInd::getSiteStack(link_site, stack);
        const gSiteStack input_forward
            = GInd::site_up(output_forward, mu);
        const gSiteStack output_backward
            = GInd::getSiteStack(forward_output_site, stack);
        const gSiteStack input_backward
            = GInd::site_dn(output_backward, mu);

        Vect12<double> chi_forward_vect
            = chi_acc.getElement(input_forward);
        Vect12<double> chi_backward_vect
            = chi_acc.getElement(input_backward);
        Vect12<double> eta_forward_vect
            = eta_acc.getElement(output_forward);
        Vect12<double> eta_backward_vect
            = eta_acc.getElement(output_backward);

        const ColorVect<double> chi_forward
            = convertVect12ToColorVect(chi_forward_vect);
        const ColorVect<double> chi_backward
            = convertVect12ToColorVect(chi_backward_vect);
        const ColorVect<double> eta_forward
            = convertVect12ToColorVect(eta_forward_vect);
        const ColorVect<double> eta_backward
            = convertVect12ToColorVect(eta_backward_vect);

        const ColorVect<double> transported_chi_forward
            = link * chi_forward;
        const ColorVect<double> projected_eta_forward
            = static_cast<double>(0.5)
              * (mdwfAllLinkWilsonGamma(mu, eta_forward) - eta_forward);
        const ColorVect<double> projected_eta_backward
            = static_cast<double>(0.5)
              * (eta_backward
                 + mdwfAllLinkWilsonGamma(mu, eta_backward));
        const ColorVect<double> transported_eta_backward
            = link * projected_eta_backward;

        for (size_t spin = 0; spin < 4; spin++) {
            raw_matrix += tensor_prod(
                transported_chi_forward[spin],
                conj(projected_eta_forward[spin]));
            raw_matrix += tensor_prod(
                chi_backward[spin],
                conj(transported_eta_backward[spin]));
        }
    }

    return raw_matrix;
}
