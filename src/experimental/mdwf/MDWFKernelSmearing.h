/*
 * Smeared links for the Wilson/clover kernel of the Mobius operator (to reduce m_res on coarse lattices), built from
 * SIMULATeQCD's HISQ smearing chain (src/modules/hisq/hisqSmearing.h):
 *
 *   fat7_u3   W = P_U3[ fat7(U) ]   HISQ level 1 (HisqSmearing::SmearLvl1, coefficients 1/8, 1/16, 1/64,
 *                                   1/384, no Lepage term, no staggered phases), then the U(3) projection
 *                                   P_U3 V = V (V^+ V)^(-1/2) (HisqSmearing::ProjectU3)
 *   fat7_su3  W exp(-i arg(det W) / 3)   the same, with the U(1) phase removed (SU(3) links)
 *
 * applied steps times (steps = 2: the W of the previous step is smeared again).
 * No Naik term: it does not belong in a Wilson-type kernel. Stout smearing is
 * deliberately not offered. The cube root of the determinant phase uses the
 * principal branch; mdwfKernelLinkReport logs the largest |arg det| before
 * the fix, which stays far from pi for smooth fat links.
 *
 * The smeared field replaces the thin links in the fermion operator only
 * (valence measurements; an HMC with smeared links also needs the smearing
 * force, WP6). Boundary phases are applied afterwards, to the smeared field.
 */

#pragma once

#include "../../gauge/gaugefield.h"
#include "../../gauge/gaugeAction.h"
#include "../../modules/hisq/hisqSmearing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

enum class MDWFKernelSmearingType { None, Fat7U3, Fat7SU3 };

inline MDWFKernelSmearingType mdwfKernelSmearingFromString(const std::string &name) {
    if (name == "none") {
        return MDWFKernelSmearingType::None;
    }
    if (name == "fat7_u3") {
        return MDWFKernelSmearingType::Fat7U3;
    }
    if (name == "fat7_su3") {
        return MDWFKernelSmearingType::Fat7SU3;
    }
    throw std::runtime_error(stdLogger.fatal("MDWF kernel smearing must be none, fat7_u3 or fat7_su3, got ", name));
}

// W -> W exp(-i arg(det W) / 3): a U(3) link to SU(3) (principal branch of the cube root).
template<size_t HaloDepth>
struct MDWFRemoveDeterminantPhase {
    SU3Accessor<double, R18> _acc;

    explicit MDWFRemoveDeterminantPhase(Gaugefield<double, true, HaloDepth, R18> &field)
        : _acc(field.getAccessor()) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        SU3<double> link = _acc.getLink(siteMu);
        const COMPLEX(double) d = det(link);
        const double phase = atan2(imag(d), real(d)) / 3.0;
        return COMPLEX(double)(cos(phase), -sin(phase)) * link;
    }
};

struct MDWFKernelLinkReport {
    double plaquette;
    double maxUnitarityViolation;   // max_links infnorm(W^+ W - 1)
    double maxDetDeviation;         // max_links |det W - 1|
    double maxDetPhase;             // max_links |arg det W|
};

// Host diagnostics of a link field (single rank).
template<size_t HaloDepth>
MDWFKernelLinkReport mdwfKernelLinkReport(Gaugefield<double, true, HaloDepth, R18> &field, const std::string &name) {
    typedef GIndexer<All, HaloDepth> GInd;
    GaugeAction<double, true, HaloDepth, R18> action(field);
    MDWFKernelLinkReport report{static_cast<double>(action.plaquette()), 0.0, 0.0, 0.0};
    Gaugefield<double, false, HaloDepth, R18> host(field.getComm(), name);
    host = field;
    const SU3Accessor<double, R18> acc = host.getAccessor();
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            SU3<double> w = acc.getLink(GInd::getSiteMu(site, mu));
            const SU3<double> check = dagger(w) * w - su3_one<double>();
            const COMPLEX(double) d = det(w);
            report.maxUnitarityViolation = std::max(report.maxUnitarityViolation,
                                                    static_cast<double>(infnorm(check)));
            report.maxDetDeviation = std::max(report.maxDetDeviation, static_cast<double>(abs(d - COMPLEX(double)(1.0, 0.0))));
            report.maxDetPhase = std::max(report.maxDetPhase, std::abs(std::atan2(imag(d), real(d))));
        }
    }
    return report;
}

/*
 * smeared = steps applications of fat7 + U(3) projection (+ determinant phase removal for fat7_su3) to thin;
 * smeared = thin for MDWFKernelSmearingType::None. thin is not changed.
 */
template<size_t HaloDepth>
void mdwfSmearKernelLinks(Gaugefield<double, true, HaloDepth, R18> &smeared,
                          Gaugefield<double, true, HaloDepth, R18> &thin, MDWFKernelSmearingType type, int steps,
                          const std::string &name) {
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    if (type == MDWFKernelSmearingType::None) {
        smeared = thin;
        smeared.updateAll();
        return;
    }
    if (steps < 1) {
        throw std::runtime_error(stdLogger.fatal("MDWF kernel smearing needs steps >= 1, got ", steps));
    }
    CommunicationBase &comm = thin.getComm();
    Gauge work(comm, name + "_work");
    Gauge fat(comm, name + "_fat7");
    Gauge unusedLvl2(comm, name + "_nolvl2");
    Gauge unusedNaik(comm, name + "_nonaik");
    work = thin;
    work.updateAll();
    // The staples read work through its accessor, so later steps only refill work.
    HisqSmearing<double, true, HaloDepth, R18, R18, R18, R18> smearing(work, unusedLvl2, unusedNaik);
    for (int step = 0; step < steps; step++) {
        smearing.SmearLvl1(fat);
        smearing.ProjectU3(fat, smeared);
        if (type == MDWFKernelSmearingType::Fat7SU3) {
            const MDWFKernelLinkReport beforeFix = mdwfKernelLinkReport<HaloDepth>(smeared, name + "_rep_u3");
            rootLogger.info("MDWF kernel smearing step ", step + 1, ": U(3) links, max |arg det| = ",
                            beforeFix.maxDetPhase, " (removed)");
            smeared.iterateOverBulkAllMu(MDWFRemoveDeterminantPhase<HaloDepth>(smeared));
            smeared.updateAll();
        }
        if (step + 1 < steps) {
            work = smeared;
            work.updateAll();
        }
    }
}
