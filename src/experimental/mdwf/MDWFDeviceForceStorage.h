/*
 * Device-accelerated all-link MDWF force storage (E3a, EVEN_ODD_DESIGN.md).
 *
 * Computes exactly what overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0
 * and overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero
 * (MDWFAllLinkDirectionIndependentStorage.h) compute,
 *
 *   destination_l = TA( sum_t w_t [ B_W,t(l) + B_C,t(l) ] ),   w_t = -2 numerator_t,
 *
 * but restructured so that the per-term work runs on the device and the
 * expensive clover-leaf products are done once per call instead of once per
 * term:
 *
 *   Wilson:  B_W,t(x, mu) = sum_s sum_spin [U chi_t(x+mu)] (P eta_t(x))^+ + chi_t(x) (U P' eta_t(x+mu))^+,
 *            accumulated per term on the device (same formula as
 *            mdwfWilsonLeftRawContractionMatrixTerm).
 *   Clover:  B_C,t is linear in the per-site, per-plane field-strength
 *            sensitivity Y_t(x, mu nu) (mdwfCloverFieldStrengthSensitivity) and
 *            otherwise depends on the gauge field only
 *            (mdwfAllLinkCloverAccumulateLeftRawContractionPath), so
 *            sum_t w_t B_C(Y_t) = B_C(sum_t w_t Y_t). The weighted sensitivity
 *            sum is accumulated on the device per term (cheap, site local);
 *            the leaf-path products run once, on the host, with the summed Y.
 *
 * The device functions below are __host__ __device__ copies of the validated
 * host helpers (the host originals are untouched); mdwfDeviceForceStorageTest
 * requires agreement with the host storage to rounding.
 *
 * Single rank, like the host storage.
 */

#pragma once

#include "MDWFAllLinkDirectionIndependentStorage.h"

#include <string>
#include <vector>

// ---- Wilson part (device copy of mdwfWilsonLeftRawContractionMatrixTerm). ----

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfDevWilsonGamma(uint8_t mu, const ColorVect<floatT> &spinor) {
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

template<size_t HaloDepth, size_t Ls>
__host__ __device__ inline SU3<double> mdwfDevWilsonRawTerm(Vect12ArrayAcc<double> chi_acc,
                                                           Vect12ArrayAcc<double> eta_acc,
                                                           SU3Accessor<double, R18> gauge_acc,
                                                           const gSite &link_site, uint8_t mu) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3<double> link = gauge_acc.getLink(GInd::getSiteMu(link_site, mu));
    const gSite forward_output_site = GInd::site_up(link_site, mu);
    SU3<double> raw_matrix = su3_zero<double>();
    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack output_forward = GInd::getSiteStack(link_site, stack);
        const gSiteStack input_forward = GInd::site_up(output_forward, mu);
        const gSiteStack output_backward = GInd::getSiteStack(forward_output_site, stack);
        const gSiteStack input_backward = GInd::site_dn(output_backward, mu);
        Vect12<double> chi_forward_vect = chi_acc.getElement(input_forward);
        Vect12<double> chi_backward_vect = chi_acc.getElement(input_backward);
        Vect12<double> eta_forward_vect = eta_acc.getElement(output_forward);
        Vect12<double> eta_backward_vect = eta_acc.getElement(output_backward);
        const ColorVect<double> chi_forward = convertVect12ToColorVect(chi_forward_vect);
        const ColorVect<double> chi_backward = convertVect12ToColorVect(chi_backward_vect);
        const ColorVect<double> eta_forward = convertVect12ToColorVect(eta_forward_vect);
        const ColorVect<double> eta_backward = convertVect12ToColorVect(eta_backward_vect);
        const ColorVect<double> transported_chi_forward = link * chi_forward;
        const ColorVect<double> projected_eta_forward
            = static_cast<double>(0.5) * (mdwfDevWilsonGamma(mu, eta_forward) - eta_forward);
        const ColorVect<double> projected_eta_backward
            = static_cast<double>(0.5) * (eta_backward + mdwfDevWilsonGamma(mu, eta_backward));
        const ColorVect<double> transported_eta_backward = link * projected_eta_backward;
        for (size_t spin = 0; spin < 4; spin++) {
            raw_matrix += tensor_prod(transported_chi_forward[spin], conj(projected_eta_forward[spin]));
            raw_matrix += tensor_prod(chi_backward[spin], conj(transported_eta_backward[spin]));
        }
    }
    return raw_matrix;
}

// accumulator(x, mu) += weight * B_W(x, mu)
template<size_t HaloDepth, size_t Ls>
struct MDWFDevWilsonAccumulate {
    SU3Accessor<double, R18> _acc;
    SU3Accessor<double, R18> _gauge;
    Vect12ArrayAcc<double> _chi;
    Vect12ArrayAcc<double> _eta;
    double _weight;

    MDWFDevWilsonAccumulate(Gaugefield<double, true, HaloDepth, R18> &acc, Gaugefield<double, true, HaloDepth, R18> &gauge,
                            const MDWFSpinor<double, true, All, HaloDepth, Ls> &chi,
                            const MDWFSpinor<double, true, All, HaloDepth, Ls> &eta, double weight)
        : _acc(acc.getAccessor()), _gauge(gauge.getAccessor()), _chi(chi.getAccessor()), _eta(eta.getAccessor()),
          _weight(weight) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;
        const gSite site = GInd::getSite(siteMu.isite);
        return _acc.getLink(siteMu)
               + _weight * mdwfDevWilsonRawTerm<HaloDepth, Ls>(_chi, _eta, _gauge, site, siteMu.mu);
    }
};

// ---- Clover sensitivity (device copies of the mdwfCloverFieldStrengthSensitivity chain). ----

__host__ __device__ inline void mdwfDevCloverAddFmunuDerivativeToBlocks(const SU3<double> &dFmunu, int mu, int nu,
                                                                       Matrix6x6<double> &upper,
                                                                       Matrix6x6<double> &lower) {
    const COMPLEX(double) ii(0.0, 1.0);
    // Non-const copy: SU3's const operator()(i, j) is __host__ only (src/base/math/su3.h), so reading a
    // const SU3 element by element in device code is undefined (it silently gave zeros, job 34576938).
    SU3<double> f = dFmunu;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            const COMPLEX(double) value = f(r, c);
            if (mu == 0 && nu == 1) {
                upper.val[r][c] += value;
                upper.val[r + 3][c + 3] -= value;
                lower.val[r][c] += value;
                lower.val[r + 3][c + 3] -= value;
            } else if (mu == 0 && nu == 2) {
                upper.val[r][c + 3] += -ii * value;
                upper.val[r + 3][c] += ii * value;
                lower.val[r][c + 3] += -ii * value;
                lower.val[r + 3][c] += ii * value;
            } else if (mu == 0 && nu == 3) {
                upper.val[r][c + 3] += -value;
                upper.val[r + 3][c] += -value;
                lower.val[r][c + 3] += value;
                lower.val[r + 3][c] += value;
            } else if (mu == 1 && nu == 2) {
                upper.val[r][c + 3] += value;
                upper.val[r + 3][c] += value;
                lower.val[r][c + 3] += value;
                lower.val[r + 3][c] += value;
            } else if (mu == 1 && nu == 3) {
                upper.val[r][c + 3] += -ii * value;
                upper.val[r + 3][c] += ii * value;
                lower.val[r][c + 3] += ii * value;
                lower.val[r + 3][c] += -ii * value;
            } else if (mu == 2 && nu == 3) {
                upper.val[r][c] += -value;
                upper.val[r + 3][c + 3] += value;
                lower.val[r][c] += value;
                lower.val[r + 3][c + 3] += -value;
            }
        }
    }
}

__host__ __device__ inline SU3<double> mdwfDevCloverSensitivityBasis(int index) {
    const COMPLEX(double) zero(0.0, 0.0);
    const COMPLEX(double) one(1.0, 0.0);
    const COMPLEX(double) pi(0.0, 1.0);
    const COMPLEX(double) mi(0.0, -1.0);
    switch (index) {
    case 0: return SU3<double>(one, zero, zero, zero, zero, zero, zero, zero, zero);
    case 1: return SU3<double>(zero, zero, zero, zero, one, zero, zero, zero, zero);
    case 2: return SU3<double>(zero, zero, zero, zero, zero, zero, zero, zero, one);
    case 3: return SU3<double>(zero, one, zero, one, zero, zero, zero, zero, zero);
    case 4: return SU3<double>(zero, pi, zero, mi, zero, zero, zero, zero, zero);
    case 5: return SU3<double>(zero, zero, one, zero, zero, zero, one, zero, zero);
    case 6: return SU3<double>(zero, zero, pi, zero, zero, zero, mi, zero, zero);
    case 7: return SU3<double>(zero, zero, zero, zero, zero, one, zero, one, zero);
    default: return SU3<double>(zero, zero, zero, zero, zero, pi, zero, mi, zero);
    }
}

// Re sum_s eta^+ (-csw/2 sigma_{mu nu} F) chi for a given F (device copy of mdwfCloverFieldStrengthContraction).
template<size_t HaloDepth, size_t Ls>
__host__ __device__ inline double mdwfDevCloverFieldStrengthContraction(Vect12ArrayAcc<double> chi_acc,
                                                                       Vect12ArrayAcc<double> eta_acc,
                                                                       const gSite &site, int mu, int nu,
                                                                       const SU3<double> &field_strength, double csw) {
    typedef GIndexer<All, HaloDepth> GInd;
    Matrix6x6<double> upper;
    Matrix6x6<double> lower;
    mdwfDevCloverAddFmunuDerivativeToBlocks(field_strength, mu, nu, upper, lower);
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            upper.val[r][c] *= -0.5 * csw;
            lower.val[r][c] *= -0.5 * csw;
        }
    }
    // Same Hermitian packing round trip as mdwfAllLinkCloverApplyDerivative.
    Vect18<double> upperStored = upper.ConvertHermitianToVect18();
    Vect18<double> lowerStored = lower.ConvertHermitianToVect18();
    Matrix6x6<double> storedUpper(upperStored);
    Matrix6x6<double> storedLower(lowerStored);
    COMPLEX(double) contraction(0.0, 0.0);
    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack siteStack = GInd::getSiteStack(site, stack);
        Vect12<double> out = storedUpper.MatrixXVect12UpDown(chi_acc.getElement(siteStack), 0);
        out = storedLower.MatrixXVect12UpDown(out, 1);
        contraction += eta_acc.getElement(siteStack) * out;
    }
    return real(contraction);
}

template<size_t HaloDepth, size_t Ls>
__host__ __device__ inline SU3<double> mdwfDevCloverSensitivity(Vect12ArrayAcc<double> chi_acc,
                                                               Vect12ArrayAcc<double> eta_acc, const gSite &site,
                                                               int mu, int nu, double csw) {
    SU3<double> sensitivity = su3_zero<double>();
    for (int b = 0; b < 9; b++) {
        const SU3<double> basis = mdwfDevCloverSensitivityBasis(b);
        const double norm = static_cast<double>(real(tr_c(basis, basis)));
        const double response
            = mdwfDevCloverFieldStrengthContraction<HaloDepth, Ls>(chi_acc, eta_acc, site, mu, nu, basis, csw);
        sensitivity += (response / norm) * basis;
    }
    return sensitivity - (1.0 / 3.0) * tr_c(sensitivity) * su3_one<double>();
}

// Plane index p = 0..5 for (mu, nu) = (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), the host loop order.
__host__ __device__ inline void mdwfDevPlane(int p, int &mu, int &nu) {
    const int mus[6] = {0, 0, 0, 1, 1, 2};
    const int nus[6] = {1, 2, 3, 2, 3, 3};
    mu = mus[p];
    nu = nus[p];
}

// Y_first holds planes 0..3 in its four link slots, Y_second planes 4, 5 in slots 0, 1: Y(x, p) += weight * Y_t(x, p).
template<size_t HaloDepth, size_t Ls>
struct MDWFDevSensitivityAccumulate {
    SU3Accessor<double, R18> _acc;
    Vect12ArrayAcc<double> _chi;
    Vect12ArrayAcc<double> _eta;
    double _weight;
    double _csw;
    int _planeOffset;

    MDWFDevSensitivityAccumulate(Gaugefield<double, true, HaloDepth, R18> &acc,
                                 const MDWFSpinor<double, true, All, HaloDepth, Ls> &chi,
                                 const MDWFSpinor<double, true, All, HaloDepth, Ls> &eta, double weight, double csw,
                                 int planeOffset)
        : _acc(acc.getAccessor()), _chi(chi.getAccessor()), _eta(eta.getAccessor()), _weight(weight), _csw(csw),
          _planeOffset(planeOffset) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;
        const int p = _planeOffset + siteMu.mu;
        if (p > 5) {
            return _acc.getLink(siteMu);
        }
        int mu = 0;
        int nu = 0;
        mdwfDevPlane(p, mu, nu);
        const gSite site = GInd::getSite(siteMu.isite);
        return _acc.getLink(siteMu)
               + _weight * mdwfDevCloverSensitivity<HaloDepth, Ls>(_chi, _eta, site, mu, nu, _csw);
    }
};

template<size_t HaloDepth>
struct MDWFDevZeroLinks {
    __host__ __device__ SU3<double> operator()(gSiteMu) {
        return su3_zero<double>();
    }
};

/*
 * Drop-in replacement for the two host storage functions (same arguments; the device gauge field
 * is copied from gaugeHost). csw = 0 skips the clover part.
 */
template<size_t HaloDepth, size_t Ls, class Terms>
void overwriteMDWFAllLinkStorageDevice(Gaugefield<double, false, HaloDepth, R18> &destination,
                                       const Gaugefield<double, false, HaloDepth, R18> &gaugeHost,
                                       const Terms &terms,
                                       const MDWFRationalCoefficients<double> &force_coefficients,
                                       double csw,
                                       CommunicationBase &comm_base,
                                       const std::string &name) {
    typedef GIndexer<All, HaloDepth> GInd;
    using DevGauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF device all-link storage is single-rank only"));
    }
    if (terms.size() != force_coefficients.numerator.size()) {
        throw std::runtime_error(stdLogger.fatal("MDWF device all-link storage: ", terms.size(), " terms but ",
                                                 force_coefficients.numerator.size(), " numerators"));
    }
    const bool clover = (csw != 0.0);

    DevGauge gauge(comm_base, name + "_dgauge");
    DevGauge wilsonAcc(comm_base, name + "_dwilson");
    DevGauge yFirst(comm_base, name + "_dyfirst");
    DevGauge ySecond(comm_base, name + "_dysecond");
    gauge = gaugeHost;
    gauge.updateAll();
    wilsonAcc.iterateOverBulkAllMu(MDWFDevZeroLinks<HaloDepth>());
    if (clover) {
        yFirst.iterateOverBulkAllMu(MDWFDevZeroLinks<HaloDepth>());
        ySecond.iterateOverBulkAllMu(MDWFDevZeroLinks<HaloDepth>());
    }

    for (size_t term = 0; term < terms.size(); term++) {
        const auto &chi = terms.chi(term);
        const auto &eta = terms.eta(term);
        const double weight = -2.0 * force_coefficients.numerator[term];
        wilsonAcc.iterateOverBulkAllMu(MDWFDevWilsonAccumulate<HaloDepth, Ls>(wilsonAcc, gauge, chi, eta, weight));
        if (clover) {
            yFirst.iterateOverBulkAllMu(MDWFDevSensitivityAccumulate<HaloDepth, Ls>(yFirst, chi, eta, weight, csw, 0));
            ySecond.iterateOverBulkAllMu(MDWFDevSensitivityAccumulate<HaloDepth, Ls>(ySecond, chi, eta, weight, csw, 4));
        }
    }

    HostGauge wilsonHost(comm_base, name + "_hwilson");
    wilsonHost = wilsonAcc;
    const size_t bulk_links = 4 * lat.vol4;
    std::vector<SU3<double>> raw_clover(bulk_links, su3_zero<double>());
    if (clover) {
        HostGauge yFirstHost(comm_base, name + "_hyfirst");
        HostGauge ySecondHost(comm_base, name + "_hysecond");
        yFirstHost = yFirst;
        ySecondHost = ySecond;
        const SU3Accessor<double, R18> gauge_acc = gaugeHost.getAccessor();
        const SU3Accessor<double, R18> y1 = yFirstHost.getAccessor();
        const SU3Accessor<double, R18> y2 = ySecondHost.getAccessor();
        std::vector<size_t> path_additions(bulk_links, 0);
        for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
            const gSite site = GInd::getSite(site_index);
            int p = 0;
            for (int mu = 0; mu < 4; mu++) {
                for (int nu = mu + 1; nu < 4; nu++, p++) {
                    const SU3<double> sensitivity = (p < 4) ? y1.getLink(GInd::getSiteMu(site, p))
                                                            : y2.getLink(GInd::getSiteMu(site, p - 4));
                    // The four leaves of mdwfAllLinkCloverAccumulateLeftRawContractionSite, rational weight
                    // already contained in the summed sensitivity.
                    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> path_p = {{
                        {GInd::getSiteMu(site, mu), false},
                        {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
                        {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
                        {GInd::getSiteMu(site, nu), true}}};
                    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> path_q = {{
                        {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
                        {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
                        {GInd::getSiteMu(GInd::site_up_dn(site, mu, nu), nu), false},
                        {GInd::getSiteMu(site, mu), true}}};
                    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> path_r = {{
                        {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
                        {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), nu), true},
                        {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), mu), false},
                        {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}}};
                    const std::array<MDWFAllLinkCloverPathFactor<double>, 4> path_s = {{
                        {GInd::getSiteMu(site, nu), false},
                        {GInd::getSiteMu(GInd::site_up_dn(site, nu, mu), mu), true},
                        {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
                        {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}}};
                    mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                        gauge_acc, path_p, sensitivity, 1.0, raw_clover, path_additions);
                    mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                        gauge_acc, path_q, sensitivity, 1.0, raw_clover, path_additions);
                    mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                        gauge_acc, path_r, sensitivity, 1.0, raw_clover, path_additions);
                    mdwfAllLinkCloverAccumulateLeftRawContractionPath<HaloDepth>(
                        gauge_acc, path_s, sensitivity, 1.0, raw_clover, path_additions);
                }
            }
        }
    }

    const SU3Accessor<double, R18> wAcc = wilsonHost.getAccessor();
    SU3Accessor<double, R18> destination_acc = destination.getAccessor();
    for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            SU3<double> total = wAcc.getLink(siteMu) + raw_clover[site.isite * 4 + mu];
            total.TA();
            destination_acc.setLink(siteMu, total);
        }
    }
}
