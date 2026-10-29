/*
 * main_emtCorrelatorTest.cpp
 *
 * Jonas Winter
 *
 * Test the EMT correlator tensor (EMTCorrGTensor) against a brute-force computation on the host:
 *     G_ab(τ, r)   = (1/V_4) sum_x T_a(t + τ, x + r) T_b^*(t, x)                         (Matsubara mode off)
 *     G_ab(ω_n, r) = (1/V_3) sum_y T^_a(ω_n, y + r)  T^_b^*(ω_n, y)                      (Matsubara mode on)
 * with a, b being the 10 index pairs of the symmetric EMT and T^ being the unitary time-only DFT (sign +1).
 *
 * Covered functions of EnergyMomentumTensorCorrelator:
 *     - EMTCorrGTensor(gauge, G, matsubara)       for matsubara = false, true; all 10 x 10 components at all r
 *
 * Not covered: EMTFullComplex, which builds the EMT field T, is used as input on both sides and not tested itself.
 * checkLatticeExtents() is called during setup but not tested.
 * Single rank only.
 *
 */

#include "../simulateqcd.h"
#include "../modules/observables/blocking.h"
#include "../modules/observables/energyMomentumTensorCorrelator.h"
#include <array>
#include <complex>

#define PREC double
#define HaloDepth 2

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
};

// field with 10 complex components per site, indexed by Lattice::index
typedef std::vector<std::array<cplx, 10>> HostEMT;
// field with 10x10 complex components per site
typedef std::vector<std::array<cplx, 100>> HostCorr;


HostEMT downloadEMT(LatticeContainer<true, Matrix4x4SymComplex<PREC>> &device, CommunicationBase &commBase, const Lattice &lat) {
    LatticeContainer<false, Matrix4x4SymComplex<PREC>> host(commBase, "emtCorrTestEMTHost", "emtCorrTestEMTHostB", "emtCorrTestEMTHostC", "emtCorrTestEMTHostD");
    host.adjustSize(GInd::getLatData().vol4);
    host.copyFromLatticeContainer(device);
    LatticeContainerAccessor accessor = host.getAccessor();

    HostEMT result(lat.vol4);
    for (int t = 0; t < lat.L[3]; t++)
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++) {
        Matrix4x4SymComplex<PREC> m = accessor.getElement<Matrix4x4SymComplex<PREC>>(GInd::getSite(x, y, z, t));
        for (int a = 0; a < 10; a++) result[lat.index(x, y, z, t)][a] = cplx(real(m.elems[a]), imag(m.elems[a]));
    }
    return result;
}

HostCorr downloadCorr(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> &device, CommunicationBase &commBase, const Lattice &lat) {
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> host(commBase, "emtCorrTestGHost", "emtCorrTestGHostB", "emtCorrTestGHostC", "emtCorrTestGHostD");
    host.adjustSize(GInd::getLatData().vol4);
    host.copyFromLatticeContainer(device);
    LatticeContainerAccessor accessor = host.getAccessor();

    HostCorr result(lat.vol4);
    for (int t = 0; t < lat.L[3]; t++)
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++) {
        Tensor4x4Symx4x4SymComplex<PREC> G = accessor.getElement<Tensor4x4Symx4x4SymComplex<PREC>>(GInd::getSite(x, y, z, t));
        for (int a = 0; a < 10; a++)
        for (int b = 0; b < 10; b++) {
            result[lat.index(x, y, z, t)][a * 10 + b] = cplx(real(G.elems[a][b]), imag(G.elems[a][b]));
        }
    }
    return result;
}

// G_ab(τ, r) = (1/V_4) sum_x T_a(t + τ, x + r) T_b^*(t, x)
HostCorr bruteForce4D(const HostEMT &T, const Lattice &lat) {
    HostCorr G(lat.vol4);
    for (int rt = 0; rt < lat.L[3]; rt++)
    for (int rz = 0; rz < lat.L[2]; rz++)
    for (int ry = 0; ry < lat.L[1]; ry++)
    for (int rx = 0; rx < lat.L[0]; rx++) {
        std::array<cplx, 100> sum;
        sum.fill(cplx(0.0, 0.0));
        for (int t = 0; t < lat.L[3]; t++)
        for (int z = 0; z < lat.L[2]; z++)
        for (int y = 0; y < lat.L[1]; y++)
        for (int x = 0; x < lat.L[0]; x++) {
            const std::array<cplx, 10> &Tr = T[lat.shifted(x, y, z, t, rx, ry, rz, rt)];
            const std::array<cplx, 10> &T0 = T[lat.index(x, y, z, t)];
            for (int a = 0; a < 10; a++)
            for (int b = 0; b < 10; b++) {
                sum[a * 10 + b] += Tr[a] * std::conj(T0[b]);
            }
        }
        for (cplx &value : sum) value /= (double) lat.vol4;
        G[lat.index(rx, ry, rz, rt)] = sum;
    }
    return G;
}

// G_ab(ω_n, r) = (1/V_3) sum_y T^_a(ω_n, y + r) T^_b^*(ω_n, y), T^(w_n, x) = N_τ^{-1/2} sum_t exp(+2 pi i n t / N_τ) T(t, x)
HostCorr bruteForceMatsubara(const HostEMT &T, const Lattice &lat) {
    const int Nt = lat.L[3];
    HostEMT That(lat.vol4);
    for (int z = 0; z < lat.L[2]; z++)
    for (int y = 0; y < lat.L[1]; y++)
    for (int x = 0; x < lat.L[0]; x++)
    for (int n = 0; n < Nt; n++) {
        std::array<cplx, 10> sum;
        sum.fill(cplx(0.0, 0.0));
        for (int t = 0; t < Nt; t++) {
            const double angle = 2.0 * M_PI * n * t / Nt;
            const cplx phase(std::cos(angle), std::sin(angle));
            for (int a = 0; a < 10; a++) sum[a] += phase * T[lat.index(x, y, z, t)][a];
        }
        for (cplx &value : sum) value /= std::sqrt((double) Nt);
        That[lat.index(x, y, z, n)] = sum;
    }

    HostCorr G(lat.vol4);
    for (int n = 0; n < Nt; n++)
    for (int rz = 0; rz < lat.L[2]; rz++)
    for (int ry = 0; ry < lat.L[1]; ry++)
    for (int rx = 0; rx < lat.L[0]; rx++) {
        std::array<cplx, 100> sum;
        sum.fill(cplx(0.0, 0.0));
        for (int z = 0; z < lat.L[2]; z++)
        for (int y = 0; y < lat.L[1]; y++)
        for (int x = 0; x < lat.L[0]; x++) {
            const std::array<cplx, 10> &Tr = That[lat.shifted(x, y, z, n, rx, ry, rz, 0)];
            const std::array<cplx, 10> &T0 = That[lat.index(x, y, z, n)];
            for (int a = 0; a < 10; a++)
            for (int b = 0; b < 10; b++) {
                sum[a * 10 + b] += Tr[a] * std::conj(T0[b]);
            }
        }
        for (cplx &value : sum) value /= (double) lat.vol3;
        G[lat.index(rx, ry, rz, n)] = sum;
    }
    return G;
}

// max |G - G_ref| / max |G_ref|; logs the worst site and component pair
double relativeDeviation(const HostCorr &G, const HostCorr &reference, const Lattice &lat, bool verbose) {
    double maxRef = 0.0, maxDiff = 0.0;
    size_t worstSite = 0;
    int worstComponent = 0;
    for (size_t site = 0; site < lat.vol4; site++)
    for (int c = 0; c < 100; c++) {
        maxRef = std::max(maxRef, std::abs(reference[site][c]));
        const double diff = std::abs(G[site][c] - reference[site][c]);
        if (diff > maxDiff) {
            maxDiff = diff;
            worstSite = site;
            worstComponent = c;
        }
    }
    if (verbose) {
        rootLogger.info("   worst site index ", worstSite, ", index pairs (", worstComponent / 10, ", ", worstComponent % 10, "): G = ",
            G[worstSite][worstComponent], ", reference = ", reference[worstSite][worstComponent]);
    }
    return maxDiff / std::max(maxRef, 1e-300);
}

// the other ordering, (1/V) sum_x T_a(t, x) T_b^*(t + τ, x + r), equals conj(G_ba(τ, r)); only used as a diagnostic
HostCorr otherOrdering(const HostCorr &G) {
    HostCorr result(G.size());
    for (size_t site = 0; site < G.size(); site++)
    for (int a = 0; a < 10; a++)
    for (int b = 0; b < 10; b++) {
        result[site][a * 10 + b] = std::conj(G[site][b * 10 + a]);
    }
    return result;
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/emtCorrelatorTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth, param, commBase);

    if (commBase.getNumberProcesses() != 1) {
        rootLogger.error("emtCorrelatorTest only supports a single rank.");
        return -1;
    }

    Gaugefield<PREC, true, HaloDepth> gauge(commBase);
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    const Lattice lat;

    // the EMT field that the correlator uses internally
    LatticeContainer<true, Matrix4x4SymComplex<PREC>> emt(commBase, "emtCorrTestEMT", "emtCorrTestEMTB", "emtCorrTestEMTC", "emtCorrTestEMTD");
    emt.adjustSize(GInd::getLatData().vol4);
    emt.template iterateOverBulk<All, HaloDepth>(EMTFullComplex<PREC, true, HaloDepth>(gauge.getAccessor()));
    const HostEMT T = downloadEMT(emt, commBase, lat);

    EnergyMomentumTensorCorrelator<PREC, HaloDepth> emtCorr(commBase);
    emtCorr.checkLatticeExtents();

    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> G(commBase, "emtCorrTestG", "emtCorrTestGB", "emtCorrTestGC", "emtCorrTestGD");
    G.adjustSize(GInd::getLatData().vol4);

    bool lerror = false;

    for (bool matsubara : {false, true}) {
        const std::string mode = matsubara ? "Matsubara mode on" : "Matsubara mode off";

        emtCorr.EMTCorrGTensor(gauge, G, matsubara);
        const HostCorr result = downloadCorr(G, commBase, lat);
        const HostCorr reference = matsubara ? bruteForceMatsubara(T, lat) : bruteForce4D(T, lat);

        const double deviation = relativeDeviation(result, reference, lat, true);
        if (deviation <= tolerance) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, "EMTCorrGTensor, ", mode, " vs brute force (rel. dev. ", deviation, ")");
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, "EMTCorrGTensor, ", mode, " vs brute force (rel. dev. ", deviation, ")");
            rootLogger.info(
                "   diagnostic: rel. dev. from the other ordering T(t, x) T^*(t + τ, x + r): ",
                relativeDeviation(result, otherOrdering(reference), lat, false)
            );
            lerror = true;
        }
    }

    if (lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green, "passed!", CoutColors::reset);
    }

    return 0;
}
