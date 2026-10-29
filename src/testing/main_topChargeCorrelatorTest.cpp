/*
 * main_topChargeCorrelatorTest.cpp
 *
 * Jonas Winter
 *
 * Test the topological-charge correlator (topChargeCorrGeneralTau) against a brute-force computation on the host:
 *     G_Q(r)       = (1/V) sum_x q(x + r) q^*(x)                                        (Matsubara mode off)
 *     G_Q(r, w_n)  = (1/V_3) sum_y q^(y + r, w_n) q^^*(y, w_n)                          (Matsubara mode on)
 * with q^ being the unitary time-only DFT (sign +1), followed by the average over r^2 shells for tau (or n) in [0, Nt/2].
 *
 * Covered functions of TopologicalChargeCorrelator:
 *     - topChargeCorrGeneralTau<improved, improved_O6>(gauge, result, matsubara)
 *           for (improved, improved_O6) = (false, false), (true, false), (true, true)
 *           and matsubara = false, true; all tau (or n) in [0, Nt/2] and all r^2 in [0, r2max]
 *
 * Not covered: the topological-charge density q(x) is taken from Topology and used as input on both sides,
 * so it is not tested itself. checkLatticeExtents() is called during setup but not tested. Single rank only.
 *
 */

#include "../simulateqcd.h"
#include "../modules/observables/topology.h"
#include "../modules/observables/topologicalChargeCorrelator.h"
#include <array>
#include <complex>

#define PREC double
#define HaloDepth 3

typedef std::complex<double> cplx;
typedef GIndexer<All, HaloDepth> GInd;

constexpr double tolerance = 1e-10;


struct Lattice {
    std::array<int, 4> L;
    size_t vol3, vol4;

    Lattice() {
        sitexyzt globL = GInd::getLatData().globalLatticeXYZT();
        L = {globL[0], globL[1], globL[2], globL[3]};
        vol3 = (size_t) L[0] * L[1] * L[2];
        vol4 = vol3 * L[3];
    }

    size_t index(int x, int y, int z, int t) const {
        return (size_t) x + L[0] * ((size_t) y + L[1] * ((size_t) z + L[2] * (size_t) t));
    }

    size_t shifted(int x, int y, int z, int t, int dx, int dy, int dz, int dt) const {
        return index((x + dx) % L[0], (y + dy) % L[1], (z + dz) % L[2], (t + dt) % L[3]);
    }

    // spatial r^2 with coordinates relative to the origin, in (-L/2, L/2]
    int r2(int x, int y, int z) const {
        int c[3] = {x, y, z};
        int result = 0;
        for (int i = 0; i < 3; i++) {
            if (c[i] > L[i] / 2) c[i] -= L[i];
            result += c[i] * c[i];
        }
        return result;
    }
};

typedef std::vector<cplx> HostScalar;


template<bool improved, bool improved_O6>
HostScalar topChargeDensity(Gaugefield<PREC, true, HaloDepth> &gauge, CommunicationBase &commBase, const Lattice &lat) {
    LatticeContainer<true, COMPLEX(PREC)> device(commBase, "topCorrTestQ", "topCorrTestQB", "topCorrTestQC", "topCorrTestQD");
    LatticeContainer<false, COMPLEX(PREC)> host(commBase, "topCorrTestQHost", "topCorrTestQHostB", "topCorrTestQHostC", "topCorrTestQHostD");
    device.adjustSize(GInd::getLatData().vol4);
    host.adjustSize(GInd::getLatData().vol4);
    device.template iterateOverBulk<All, HaloDepth>(topChargeDensKernelComplex<PREC, HaloDepth, true, improved, improved_O6>(gauge));
    host.copyFromLatticeContainer(device);
    LatticeContainerAccessor accessor = host.getAccessor();

    HostScalar q(lat.vol4);
    for (int t = 0; t < lat.L[3]; t++)
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++) {
        COMPLEX(PREC) value = accessor.getElement<COMPLEX(PREC)>(GInd::getSite(x, y, z, t));
        q[lat.index(x, y, z, t)] = cplx(real(value), imag(value));
    }
    return q;
}

// G_Q(r) = (1/V) sum_x q(x + r) q^*(x)
HostScalar bruteForce4D(const HostScalar &q, const Lattice &lat) {
    HostScalar G(lat.vol4);
    for (int rt = 0; rt < lat.L[3]; rt++)
    for (int rz = 0; rz < lat.L[2]; rz++)
    for (int ry = 0; ry < lat.L[1]; ry++)
    for (int rx = 0; rx < lat.L[0]; rx++) {
        cplx sum(0.0, 0.0);
        for (int t = 0; t < lat.L[3]; t++)
        for (int z = 0; z < lat.L[2]; z++)
        for (int y = 0; y < lat.L[1]; y++)
        for (int x = 0; x < lat.L[0]; x++) {
            sum += q[lat.shifted(x, y, z, t, rx, ry, rz, rt)] * std::conj(q[lat.index(x, y, z, t)]);
        }
        G[lat.index(rx, ry, rz, rt)] = sum / (double) lat.vol4;
    }
    return G;
}

// G_Q(r, w_n) = (1/V_3) sum_y q^(y + r, w_n) q^^*(y, w_n), q^(x, w_n) = Nt^{-1/2} sum_t exp(+2 pi i n t / Nt) q(x, t)
HostScalar bruteForceMatsubara(const HostScalar &q, const Lattice &lat) {
    const int Nt = lat.L[3];
    HostScalar qhat(lat.vol4);
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++)
    for (int n = 0; n < Nt; n++) {
        cplx sum(0.0, 0.0);
        for (int t = 0; t < Nt; t++) {
            const double angle = 2.0 * M_PI * n * t / Nt;
            sum += cplx(std::cos(angle), std::sin(angle)) * q[lat.index(x, y, z, t)];
        }
        qhat[lat.index(x, y, z, n)] = sum / std::sqrt((double) Nt);
    }

    HostScalar G(lat.vol4);
    for (int n = 0; n < Nt; n++)
    for (int rz = 0; rz < lat.L[2]; rz++)
    for (int ry = 0; ry < lat.L[1]; ry++)
    for (int rx = 0; rx < lat.L[0]; rx++) {
        cplx sum(0.0, 0.0);
        for (int z = 0; z < lat.L[2]; z++)
        for (int y = 0; y < lat.L[1]; y++)
        for (int x = 0; x < lat.L[0]; x++) {
            sum += qhat[lat.shifted(x, y, z, n, rx, ry, rz, 0)] * std::conj(qhat[lat.index(x, y, z, n)]);
        }
        G[lat.index(rx, ry, rz, n)] = sum / (double) lat.vol3;
    }
    return G;
}

// average over the r^2 shells for tau in [0, tauMax] and r^2 in [0, r2max]; empty shells stay 0
std::vector<std::vector<cplx>> shellAverage(const HostScalar &G, const Lattice &lat, int tauMax, int r2max) {
    std::vector<std::vector<cplx>> result(tauMax + 1, std::vector<cplx>(r2max + 1, cplx(0.0, 0.0)));
    std::vector<int> counts(r2max + 1, 0);
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++) {
        const int r2 = lat.r2(x, y, z);
        if (r2 > r2max) continue;
        counts[r2]++;
        for (int tau = 0; tau <= tauMax; tau++) result[tau][r2] += G[lat.index(x, y, z, tau)];
    }
    for (int tau = 0; tau <= tauMax; tau++)
    for (int r2 = 0; r2 <= r2max; r2++) {
        if (counts[r2] != 0) result[tau][r2] /= (double) counts[r2];
    }
    return result;
}

template<bool improved, bool improved_O6>
bool testDiscretisation(Gaugefield<PREC, true, HaloDepth> &gauge, CommunicationBase &commBase,
                        TopologicalChargeCorrelator<PREC, HaloDepth> &topChargeCorr, const std::string &name) {
    const Lattice lat;
    const int tauMax = topChargeCorr.getTauMax();
    const int r2max = topChargeCorr.getR2max();
    const HostScalar q = topChargeDensity<improved, improved_O6>(gauge, commBase, lat);

    bool lerror = false;
    for (bool matsubara : {false, true}) {
        const std::string text = "topChargeCorrGeneralTau<" + name + ">, " + (matsubara ? "Matsubara mode on" : "Matsubara mode off") + " vs brute force";

        std::vector<std::vector<COMPLEX(PREC)>> result(tauMax + 1, std::vector<COMPLEX(PREC)>(r2max + 1));
        topChargeCorr.template topChargeCorrGeneralTau<improved, improved_O6>(gauge, result, matsubara);

        const std::vector<std::vector<cplx>> reference = shellAverage(matsubara ? bruteForceMatsubara(q, lat) : bruteForce4D(q, lat), lat, tauMax, r2max);

        double maxRef = 0.0, maxDiff = 0.0;
        for (int tau = 0; tau <= tauMax; tau++)
        for (int r2 = 0; r2 <= r2max; r2++) {
            maxRef = std::max(maxRef, std::abs(reference[tau][r2]));
            maxDiff = std::max(maxDiff, std::abs(cplx(real(result[tau][r2]), imag(result[tau][r2])) - reference[tau][r2]));
        }
        const double deviation = maxDiff / std::max(maxRef, 1e-300);

        if (deviation <= tolerance) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text, " (rel. dev. ", deviation, ")");
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text, " (rel. dev. ", deviation, ")");
            lerror = true;
        }
    }
    return lerror;
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/topChargeCorrelatorTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth, param, commBase);

    if (commBase.getNumberProcesses() != 1) {
        rootLogger.error("topChargeCorrelatorTest only supports a single rank.");
        return -1;
    }

    // HaloDepth 3 covers the 1x3 rectangles of the O(a^6)-improved density (finding B5)
    Gaugefield<PREC, true, HaloDepth> gauge(commBase);
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    TopologicalChargeCorrelator<PREC, HaloDepth> topChargeCorr(commBase);
    topChargeCorr.checkLatticeExtents();

    bool lerror = false;
    lerror |= testDiscretisation<false, false>(gauge, commBase, topChargeCorr, "plain");
    lerror |= testDiscretisation<true, false>(gauge, commBase, topChargeCorr, "improved");
    lerror |= testDiscretisation<true, true>(gauge, commBase, topChargeCorr, "improved_O6");

    if (lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green, "passed!", CoutColors::reset);
    }

    return 0;
}
