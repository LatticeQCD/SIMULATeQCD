/*
 * Device-accelerated MDWF all-link force storage test (E3a,
 * MDWFDeviceForceStorage.h).
 *
 * Random gauge field (6^4, Ls = 8) and random terms (chi_t, eta_t Gaussian
 * MDWF spinors, numerators of both signs). For c_sw = 0 and c_sw = 0.5,
 * overwriteMDWFAllLinkStorageDevice must reproduce the validated host storage
 * (overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0 /
 * overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero) link by link:
 * max |K_device - K_host| <= 1e-12 max |K_host|. Wall times of both are
 * reported for 5 and for 40 terms (the host cost grows with the number of
 * terms, the device version's clover-leaf part does not).
 *
 * Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDeviceForceStorage.h"
#include "../experimental/mdwf/MDWFHmcFermionActions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

template<size_t HaloDepth>
double mdwfDevStorageMaxDiff(const Gaugefield<double, false, HaloDepth, R18> &a,
                             const Gaugefield<double, false, HaloDepth, R18> &b, double &maxNorm) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> aAcc = a.getAccessor();
    const SU3Accessor<double, R18> bAcc = b.getAccessor();
    double maxDiff = 0.0;
    maxNorm = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            maxDiff = std::max(maxDiff, static_cast<double>(infnorm(aAcc.getLink(siteMu) - bAcc.getLink(siteMu))));
            maxNorm = std::max(maxNorm, static_cast<double>(infnorm(bAcc.getLink(siteMu))));
        }
    }
    return maxDiff;
}

double mdwfDevStorageSeconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const size_t HaloDepth = 2;
        const size_t Ls = 8;
        initIndexer(HaloDepth, param, commBase);
        using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
        using DevGauge = Gaugefield<double, true, HaloDepth, R18>;
        using HostGauge = Gaugefield<double, false, HaloDepth, R18>;

        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261006);
        d_rand = h_rand;

        DevGauge gauge(commBase, "MDWF_devst_gauge");
        gauge.random(d_rand.state);
        gauge.updateAll();
        HostGauge gaugeHost(commBase, "MDWF_devst_ghost");
        gaugeHost = gauge;

        const size_t nFields = 5;
        std::vector<std::unique_ptr<Spinor>> chis;
        std::vector<std::unique_ptr<Spinor>> etas;
        for (size_t t = 0; t < nFields; t++) {
            chis.emplace_back(new Spinor(commBase, "MDWF_devst_c" + std::to_string(t) + "x"));
            etas.emplace_back(new Spinor(commBase, "MDWF_devst_e" + std::to_string(t) + "x"));
            chis.back()->gauss(d_rand.state);
            etas.back()->gauss(d_rand.state);
            chis.back()->updateAll();
            etas.back()->updateAll();
        }

        HostGauge kHost(commBase, "MDWF_devst_khost");
        HostGauge kDevice(commBase, "MDWF_devst_kdev");
        bool passed = true;
        for (int c = 0; c < 2; c++) {
            const double csw = (c == 0) ? 0.5 : 0.0;
            for (size_t nTerms : {static_cast<size_t>(5), static_cast<size_t>(40)}) {
                MDWFExplicitForceTerms<Spinor> terms;
                std::vector<double> numerators;
                for (size_t t = 0; t < nTerms; t++) {
                    terms.add(*chis[t % nFields], *etas[(3 * t + 1) % nFields]);
                    numerators.push_back(0.7 - 0.37 * static_cast<double>(t % 7) + 0.05 * static_cast<double>(t));
                }
                const MDWFRationalCoefficients<double> coefficients{0.0, numerators,
                                                                     std::vector<double>(nTerms, 0.0)};
                auto start = std::chrono::steady_clock::now();
                if (csw != 0.0) {
                    overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
                        kHost, gaugeHost, terms, coefficients, csw, commBase, "MDWF_devst_hst");
                } else {
                    overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<HaloDepth, Ls>(
                        kHost, gaugeHost, terms, coefficients, commBase, "MDWF_devst_hst");
                }
                const double hostSeconds = mdwfDevStorageSeconds(start);
                start = std::chrono::steady_clock::now();
                overwriteMDWFAllLinkStorageDevice<HaloDepth, Ls>(kDevice, gaugeHost, terms, coefficients, csw, commBase,
                                                                 "MDWF_devst_dst");
                const double deviceSeconds = mdwfDevStorageSeconds(start);
                double maxNorm = 0.0;
                const double maxDiff = mdwfDevStorageMaxDiff<HaloDepth>(kDevice, kHost, maxNorm);
                const bool ok = maxNorm > 0.0 && maxDiff <= 1e-12 * maxNorm;
                passed = passed && ok;
                rootLogger.info("MDWF device force storage test (c_sw = ", csw, ", ", nTerms,
                                " terms): max |K_device - K_host| = ", maxDiff, " (max |K| = ", maxNorm,
                                ", relative ", maxDiff / maxNorm, "); host ", hostSeconds, " s, device ",
                                deviceSeconds, " s, speedup ", hostSeconds / deviceSeconds, ", passed = ", ok);
            }
        }
        if (!passed) {
            throw std::runtime_error(stdLogger.fatal("MDWF device force storage test failed"));
        }
        rootLogger.info("MDWF device force storage test passed");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
