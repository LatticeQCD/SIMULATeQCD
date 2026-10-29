/*
 * main_tensorDecompositionTest.cpp
 *
 * Jonas Winter
 *
 * Test the tensor decomposition of the correlator tensor:
 *   1. projector algebra on the host (symmetry, orthogonality, idempotence, completeness, r = 0 limits),
 *   2. recovery of known component functions g_X from a synthetic G = sum_X g_X(r^2, tau) P_X(r),
 *   3. the r^2 bookkeeping (r2max, shell counts, number of non-empty shells).
 *
 * Test the tensor decomposition of the correlator tensor:
 *   1. projector algebra on the host (symmetry, orthogonality, idempotence, completeness, r = 0 limits),
 *   2. recovery of known component functions g_X from a synthetic G = sum_X g_X(r^2, tau) P_X(r),
 *   3. the r^2 bookkeeping (r2max, shell counts, number of non-empty shells).
 *
 * Covered functions:
 *   1. projector algebra, for all 14 projectors (TT, RT, MT, UT, ss, ll, ww, mm, sl, sw, sm, lw, lm, wm):
 *     - projectorFunction<floatT, Projector>(r, mu, nu, rho, sigma)
 *   2. recovery of the component functions, for the projectors listed in allProjectorsAveragedTau and
 *      allProjectorsGeneralTau, for r^2 > 0 and at r^2 = 0:
 *     - TensorDecomposition::getComponentFunctionsAveragedTau<true>
 *     - TensorDecomposition::getComponentFunctionsGeneralTau<true>
 *   3. r^2 bookkeeping, against brute-force counts over the global lattice:
 *     - TensorDecomposition::getR2max()            = min(Lx, Ly, Lz)^2 / 4
 *     - TensorDecomposition::getTauMax()           = Nt / 2
 *     - TensorDecomposition::getR2Counts(counts)
 *     - TensorDecomposition::getNumberOfHitR2()
 *
 * Not covered: multiple ranks.
 * The test is single rank only, and getR2Counts loops over the local lattice extents,
 * so it gives wrong counts on a split lattice.
 *
 */

#include "../simulateqcd.h"
#include "../modules/tensor_decomposition/tensorDecomposition.h"
#include <array>
#include <complex>

#define PREC double
#define HaloDepth 0

typedef std::complex<double> cplx;
typedef GIndexer<All, HaloDepth> GInd;
typedef std::array<double, 256> Tensor4;  // index ((mu*4 + nu)*4 + rho)*4 + sigma

constexpr double tolerance = 1e-12;
constexpr double toleranceRecovery = 1e-10;

constexpr std::array<Projector, 7> diagonalProjectors{{
    Projector::TT, Projector::RT, Projector::UT, Projector::ss, Projector::ll, Projector::ww, Projector::mm
}};

// mixed projectors P_ab and the diagonal projectors P_aa, P_bb with P_ab . P_ab = P_aa + P_bb
constexpr std::array<std::array<Projector, 3>, 7> mixedProjectors{{
    {{Projector::MT, Projector::RT, Projector::UT}},
    {{Projector::sl, Projector::ss, Projector::ll}},
    {{Projector::sw, Projector::ss, Projector::ww}},
    {{Projector::sm, Projector::ss, Projector::mm}},
    {{Projector::lw, Projector::ll, Projector::ww}},
    {{Projector::lm, Projector::ll, Projector::mm}},
    {{Projector::wm, Projector::ww, Projector::mm}}
}};

std::string projectorName(Projector projector) {
    static const std::array<std::string, 14> names{{
        "TT", "RT", "MT", "UT", "ss", "ll", "ww", "mm", "sl", "sw", "sm", "lw", "lm", "wm"
    }};
    return names[(int) projector];
}

int idx(int mu, int nu, int rho, int sigma) {
    return ((mu * 4 + nu) * 4 + rho) * 4 + sigma;
}


// ------------------------------------------------------------------------------- projectors on the host

double projectorValue(Projector projector, sitexyzt r, int mu, int nu, int rho, int sigma) {
    switch (projector) {
        case Projector::TT: return projectorFunction<PREC, Projector::TT>(r, mu, nu, rho, sigma);
        case Projector::RT: return projectorFunction<PREC, Projector::RT>(r, mu, nu, rho, sigma);
        case Projector::MT: return projectorFunction<PREC, Projector::MT>(r, mu, nu, rho, sigma);
        case Projector::UT: return projectorFunction<PREC, Projector::UT>(r, mu, nu, rho, sigma);
        case Projector::ss: return projectorFunction<PREC, Projector::ss>(r, mu, nu, rho, sigma);
        case Projector::ll: return projectorFunction<PREC, Projector::ll>(r, mu, nu, rho, sigma);
        case Projector::ww: return projectorFunction<PREC, Projector::ww>(r, mu, nu, rho, sigma);
        case Projector::mm: return projectorFunction<PREC, Projector::mm>(r, mu, nu, rho, sigma);
        case Projector::sl: return projectorFunction<PREC, Projector::sl>(r, mu, nu, rho, sigma);
        case Projector::sw: return projectorFunction<PREC, Projector::sw>(r, mu, nu, rho, sigma);
        case Projector::sm: return projectorFunction<PREC, Projector::sm>(r, mu, nu, rho, sigma);
        case Projector::lw: return projectorFunction<PREC, Projector::lw>(r, mu, nu, rho, sigma);
        case Projector::lm: return projectorFunction<PREC, Projector::lm>(r, mu, nu, rho, sigma);
        case Projector::wm: return projectorFunction<PREC, Projector::wm>(r, mu, nu, rho, sigma);
    }
    return 0.0;
}

Tensor4 projectorTensor(Projector projector, sitexyzt r) {
    Tensor4 result;
    for (int mu = 0; mu < 4; mu++)
    for (int nu = 0; nu < 4; nu++)
    for (int rho = 0; rho < 4; rho++)
    for (int sigma = 0; sigma < 4; sigma++) {
        result[idx(mu, nu, rho, sigma)] = projectorValue(projector, r, mu, nu, rho, sigma);
    }
    return result;
}

// full contraction A : B over all four indices
double contract(const Tensor4 &A, const Tensor4 &B) {
    double result = 0.0;
    for (int i = 0; i < 256; i++) result += A[i] * B[i];
    return result;
}

// (A . B)(mu nu, rho sigma) = sum_{alpha beta} A(mu nu, alpha beta) B(alpha beta, rho sigma)
Tensor4 product(const Tensor4 &A, const Tensor4 &B) {
    Tensor4 result;
    result.fill(0.0);
    for (int mu = 0; mu < 4; mu++)
    for (int nu = 0; nu < 4; nu++)
    for (int rho = 0; rho < 4; rho++)
    for (int sigma = 0; sigma < 4; sigma++)
    for (int alpha = 0; alpha < 4; alpha++)
    for (int beta = 0; beta < 4; beta++) {
        result[idx(mu, nu, rho, sigma)] += A[idx(mu, nu, alpha, beta)] * B[idx(alpha, beta, rho, sigma)];
    }
    return result;
}

Tensor4 addTensors(const Tensor4 &A, const Tensor4 &B, double factorB = 1.0) {
    Tensor4 result;
    for (int i = 0; i < 256; i++) result[i] = A[i] + factorB * B[i];
    return result;
}

double maxAbsDifference(const Tensor4 &A, const Tensor4 &B) {
    double result = 0.0;
    for (int i = 0; i < 256; i++) result = std::max(result, std::abs(A[i] - B[i]));
    return result;
}

// identity on symmetric 4x4 matrices: (delta_{mu rho} delta_{nu sigma} + delta_{mu sigma} delta_{nu rho}) / 2
Tensor4 symmetricIdentity() {
    Tensor4 result;
    for (int mu = 0; mu < 4; mu++)
    for (int nu = 0; nu < 4; nu++)
    for (int rho = 0; rho < 4; rho++)
    for (int sigma = 0; sigma < 4; sigma++) {
        result[idx(mu, nu, rho, sigma)] = 0.5 * (delta(mu, rho) * delta(nu, sigma) + delta(mu, sigma) * delta(nu, rho));
    }
    return result;
}

// all integer spatial vectors with components in [-2, 2], except 0
std::vector<sitexyzt> testVectors() {
    std::vector<sitexyzt> vectors;
    for (int x = -2; x <= 2; x++)
    for (int y = -2; y <= 2; y++)
    for (int z = -2; z <= 2; z++) {
        if (x == 0 && y == 0 && z == 0) continue;
        vectors.push_back(sitexyzt(x, y, z, 0));
    }
    return vectors;
}


// ------------------------------------------------------------------------------- bookkeeping

struct TestContext {
    bool lerror = false;

    void checkDeviation(double deviation, double tol, const std::string &text) {
        if (deviation <= tol) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text, " (max. dev. ", deviation, ")");
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text, " (max. dev. ", deviation, ")");
            lerror = true;
        }
    }

    void checkEqual(long expected, long result, const std::string &text) {
        if (expected == result) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text, " (", result, ")");
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text, " (expected ", expected, ", got ", result, ")");
            lerror = true;
        }
    }
};


// ------------------------------------------------------------------------------- 1. projector algebra

void testProjectorAlgebra(TestContext &ctx) {
    const std::vector<sitexyzt> vectors = testVectors();
    const Tensor4 identity = symmetricIdentity();

    for (int X = 0; X < 14; X++) {
        const Projector projector = (Projector) X;
        double symmetryDeviation = 0.0;
        for (const sitexyzt &r : vectors) {
            const Tensor4 P = projectorTensor(projector, r);
            for (int mu = 0; mu < 4; mu++)
            for (int nu = 0; nu < 4; nu++)
            for (int rho = 0; rho < 4; rho++)
            for (int sigma = 0; sigma < 4; sigma++) {
                const double value = P[idx(mu, nu, rho, sigma)];
                symmetryDeviation = std::max(symmetryDeviation, std::abs(value - P[idx(nu, mu, rho, sigma)]));
                symmetryDeviation = std::max(symmetryDeviation, std::abs(value - P[idx(mu, nu, sigma, rho)]));
                symmetryDeviation = std::max(symmetryDeviation, std::abs(value - P[idx(rho, sigma, mu, nu)]));
            }
        }
        ctx.checkDeviation(symmetryDeviation, tolerance, "P_" + projectorName(projector) + " is symmetric in mu<->nu, rho<->sigma and under pair exchange");
    }

    // orthogonality: P_X : P_Y = multiplicity(X) delta_XY
    double orthogonalityDeviation = 0.0;
    for (const sitexyzt &r : vectors) {
        std::array<Tensor4, 14> P;
        for (int X = 0; X < 14; X++) P[X] = projectorTensor((Projector) X, r);
        for (int X = 0; X < 14; X++)
        for (int Y = 0; Y < 14; Y++) {
            const double expected = (X == Y) ? multiplicity((Projector) X) : 0.0;
            orthogonalityDeviation = std::max(orthogonalityDeviation, std::abs(contract(P[X], P[Y]) - expected));
        }
    }
    ctx.checkDeviation(orthogonalityDeviation, tolerance, "P_X : P_Y = multiplicity(X) delta_XY for all 14 projectors");

    // idempotence of the diagonal projectors
    for (Projector projector : diagonalProjectors) {
        double deviation = 0.0;
        for (const sitexyzt &r : vectors) {
            const Tensor4 P = projectorTensor(projector, r);
            deviation = std::max(deviation, maxAbsDifference(product(P, P), P));
        }
        ctx.checkDeviation(deviation, tolerance, "P_" + projectorName(projector) + " . P_" + projectorName(projector) + " = P_" + projectorName(projector));
    }

    // mixed projectors: P_ab . P_ab = P_aa + P_bb
    for (const std::array<Projector, 3> &mixed : mixedProjectors) {
        double deviation = 0.0;
        for (const sitexyzt &r : vectors) {
            const Tensor4 P = projectorTensor(mixed[0], r);
            const Tensor4 expected = addTensors(projectorTensor(mixed[1], r), projectorTensor(mixed[2], r));
            deviation = std::max(deviation, maxAbsDifference(product(P, P), expected));
        }
        ctx.checkDeviation(deviation, tolerance, "P_" + projectorName(mixed[0]) + " . P_" + projectorName(mixed[0])
            + " = P_" + projectorName(mixed[1]) + " + P_" + projectorName(mixed[2]));
    }

    // completeness: the diagonal projectors add up to the identity on symmetric matrices
    double completenessDeviation = 0.0;
    for (const sitexyzt &r : vectors) {
        Tensor4 sum;
        sum.fill(0.0);
        for (Projector projector : diagonalProjectors) sum = addTensors(sum, projectorTensor(projector, r));
        completenessDeviation = std::max(completenessDeviation, maxAbsDifference(sum, identity));
    }
    ctx.checkDeviation(completenessDeviation, tolerance, "sum of the diagonal projectors = identity on symmetric matrices");

    // r = 0: isotropic average of P_X(r_hat). The projectors are polynomials of degree <= 4 in r_hat, and the integer set
    // (2/5) avg(+-e_i) + (3/5) avg((+-1, +-1, +-1)) is an exact spherical 4-design.
    std::vector<sitexyzt> axes, diagonals;
    for (int i = 0; i < 3; i++)
    for (int s : {1, -1}) {
        int c[3] = {0, 0, 0};
        c[i] = s;
        axes.push_back(sitexyzt(c[0], c[1], c[2], 0));
    }
    for (int x : {1, -1})
    for (int y : {1, -1})
    for (int z : {1, -1}) {
        diagonals.push_back(sitexyzt(x, y, z, 0));
    }
    for (int X = 0; X < 14; X++) {
        const Projector projector = (Projector) X;
        Tensor4 average;
        average.fill(0.0);
        for (const sitexyzt &r : axes) average = addTensors(average, projectorTensor(projector, r), 0.4 / axes.size());
        for (const sitexyzt &r : diagonals) average = addTensors(average, projectorTensor(projector, r), 0.6 / diagonals.size());
        const Tensor4 atOrigin = projectorTensor(projector, sitexyzt(0, 0, 0, 0));
        ctx.checkDeviation(maxAbsDifference(atOrigin, average), tolerance, "P_" + projectorName(projector) + "(r=0) = isotropic average of P_" + projectorName(projector) + "(r_hat)");
    }
}


// ------------------------------------------------------------------------------- 2. recovery of component functions

// known component function g_X(r^2, tau), different for every projector
cplx syntheticComponentFunction(Projector projector, int r2, int tau) {
    const int X = (int) projector;
    return cplx(1.0 + X + 0.05 * r2 + 0.1 * tau, 0.2 * (X + 1) - 0.01 * r2 + 0.03 * tau);
}

sitexyzt relativePosition(int x, int y, int z, int t) {
    return GInd::getLatData().globalPosRelativeToOrigin(sitexyzt(x, y, z, t));
}

void testRecovery(TestContext &ctx, CommunicationBase &commBase) {
    TensorDecomposition<PREC, HaloDepth> tensorDecomposition(commBase);
    const int r2max = tensorDecomposition.getR2max();
    const int tauMax = tensorDecomposition.getTauMax();
    const int lt = GInd::getLatData().globLT;

    // build G = sum_X g_X(r^2, tau) P_X(r) on the host
    LatticeContainer<false, Tensor4x4Symx4x4SymComplex<PREC>> GHost(commBase, "tdTestGHost", "tdTestGHostB", "tdTestGHostC", "tdTestGHostD");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> GDevice(commBase, "tdTestGDevice", "tdTestGDeviceB", "tdTestGDeviceC", "tdTestGDeviceD");
    GHost.adjustSize(GInd::getLatData().vol4);
    GDevice.adjustSize(GInd::getLatData().vol4);

    LatticeContainerAccessor accessor = GHost.getAccessor();
    for (int t = 0; t < (int) GInd::getLatData().lt; t++)
    for (int z = 0; z < (int) GInd::getLatData().lz; z++)
    for (int y = 0; y < (int) GInd::getLatData().ly; y++)
    for (int x = 0; x < (int) GInd::getLatData().lx; x++) {
        sitexyzt r = relativePosition(x, y, z, t);
        const int r2 = r.getSpatialPart().getNormSquared();
        const int tau = GInd::getLatData().globalPos(sitexyzt(x, y, z, t))[3];

        Tensor4x4Symx4x4SymComplex<PREC> G;
        for (int X = 0; X < 14; X++) {
            const Projector projector = (Projector) X;
            const cplx g = syntheticComponentFunction(projector, r2, tau);
            for (int mu = 0; mu < 4; mu++)
            for (int nu = 0; nu <= mu; nu++)
            for (int rho = 0; rho < 4; rho++)
            for (int sigma = 0; sigma <= rho; sigma++) {
                const double p = projectorValue(projector, r, mu, nu, rho, sigma);
                G(mu, nu, rho, sigma, G(mu, nu, rho, sigma) + COMPLEX(PREC)(p * g.real(), p * g.imag()));
            }
        }
        accessor.setElement(GInd::getSite(x, y, z, t), G);
    }
    GDevice.copyFromLatticeContainer(GHost);

    std::vector<std::vector<COMPLEX(PREC)>> averagedTau(10, std::vector<COMPLEX(PREC)>(r2max + 1));
    std::vector<std::vector<std::vector<COMPLEX(PREC)>>> generalTau(14, std::vector<std::vector<COMPLEX(PREC)>>(tauMax + 1, std::vector<COMPLEX(PREC)>(r2max + 1)));
    tensorDecomposition.getComponentFunctionsAveragedTau<true>(GDevice, averagedTau);
    tensorDecomposition.getComponentFunctionsGeneralTau<true>(GDevice, generalTau);

    std::vector<int> counts(r2max + 1);
    tensorDecomposition.getR2Counts(counts);

    // at r = 0 the projectors are not orthogonal (e.g. P_TT(0) = P_RT(0)): expected G^X(0) = sum_Y g_Y (P_X(0) : P_Y(0)) / m_X
    const sitexyzt origin(0, 0, 0, 0);
    std::array<std::array<double, 14>, 14> gram;
    for (int X = 0; X < 14; X++)
    for (int Y = 0; Y < 14; Y++) {
        gram[X][Y] = contract(projectorTensor((Projector) X, origin), projectorTensor((Projector) Y, origin));
    }
    auto expected = [&](Projector projector, int r2, int tau) {
        if (r2 != 0) return syntheticComponentFunction(projector, r2, tau);
        cplx result(0.0, 0.0);
        for (int Y = 0; Y < 14; Y++) result += gram[(int) projector][Y] * syntheticComponentFunction((Projector) Y, 0, tau);
        return result / (double) multiplicity(projector);
    };

    for (size_t i = 0; i < allProjectorsGeneralTau.size(); i++) {
        const Projector projector = allProjectorsGeneralTau[i];
        double deviation = 0.0, deviationOrigin = 0.0;
        for (int tau = 0; tau <= tauMax; tau++)
        for (int r2 = 0; r2 <= r2max; r2++) {
            if (counts[r2] == 0) continue;
            const cplx result(real(generalTau[i][tau][r2]), imag(generalTau[i][tau][r2]));
            const double diff = std::abs(result - expected(projector, r2, tau));
            if (r2 == 0) deviationOrigin = std::max(deviationOrigin, diff);
            else deviation = std::max(deviation, diff);
        }
        ctx.checkDeviation(deviation, toleranceRecovery, "general tau: g_" + projectorName(projector) + " recovered for r^2 > 0");
        ctx.checkDeviation(deviationOrigin, toleranceRecovery, "general tau: G^" + projectorName(projector) + "(r^2 = 0) matches the projector overlaps");
    }

    for (size_t i = 0; i < allProjectorsAveragedTau.size(); i++) {
        const Projector projector = allProjectorsAveragedTau[i];
        double deviation = 0.0, deviationOrigin = 0.0;
        for (int r2 = 0; r2 <= r2max; r2++) {
            if (counts[r2] == 0) continue;
            cplx average(0.0, 0.0);
            for (int tau = 0; tau < lt; tau++) average += expected(projector, r2, tau);
            average /= (double) lt;
            const cplx result(real(averagedTau[i][r2]), imag(averagedTau[i][r2]));
            const double diff = std::abs(result - average);
            if (r2 == 0) deviationOrigin = std::max(deviationOrigin, diff);
            else deviation = std::max(deviation, diff);
        }
        ctx.checkDeviation(deviation, toleranceRecovery, "averaged tau: tau-average of g_" + projectorName(projector) + " recovered for r^2 > 0");
        ctx.checkDeviation(deviationOrigin, toleranceRecovery, "averaged tau: G^" + projectorName(projector) + "(r^2 = 0) matches the projector overlaps");
    }
}


// ------------------------------------------------------------------------------- 3. r^2 bookkeeping

void testR2Bookkeeping(TestContext &ctx) {
    typedef TensorDecomposition<PREC, HaloDepth> TD;
    sitexyzt globL = GInd::getLatData().globalLatticeXYZT();

    const int minExtent = std::min({globL[0], globL[1], globL[2]});
    const int expectedR2max = minExtent * minExtent / 4;
    ctx.checkEqual(expectedR2max, TD::getR2max(), "getR2max() = min(Lx, Ly, Lz)^2 / 4");
    ctx.checkEqual(globL[3] / 2, TD::getTauMax(), "getTauMax() = Nt / 2");

    // brute-force shell counts with the expected r2max
    std::vector<int> expectedCounts(std::max(expectedR2max, TD::getR2max()) + 1, 0);
    for (int x = 0; x < globL[0]; x++)
    for (int y = 0; y < globL[1]; y++)
    for (int z = 0; z < globL[2]; z++) {
        int c[3] = {x, y, z};
        int r2 = 0;
        for (int i = 0; i < 3; i++) {
            if (c[i] > globL[i] / 2) c[i] -= globL[i];
            r2 += c[i] * c[i];
        }
        if (r2 <= expectedR2max) expectedCounts[r2]++;
    }

    // getR2Counts writes up to its own r2max, so the vector has to be large enough for both
    std::vector<int> counts(expectedCounts.size());
    TD::getR2Counts(counts);
    long wrongShells = 0;
    for (size_t r2 = 0; r2 < counts.size(); r2++) {
        if (counts[r2] != expectedCounts[r2]) wrongShells++;
    }
    ctx.checkEqual(0, wrongShells, "getR2Counts() matches the brute-force shell counts (number of wrong shells)");

    long hitShells = 0;
    for (int count : expectedCounts) {
        if (count != 0) hitShells++;
    }
    ctx.checkEqual(hitShells, TD::getNumberOfHitR2(), "getNumberOfHitR2() = number of non-empty shells");
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/tensorDecompositionTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth, param, commBase);

    if (commBase.getNumberProcesses() != 1) {
        rootLogger.error("tensorDecompositionTest only supports a single rank.");
        return -1;
    }

    TestContext ctx;

    rootLogger.info("Testing the projector algebra...");
    testProjectorAlgebra(ctx);

    rootLogger.info("Testing the recovery of component functions...");
    testRecovery(ctx, commBase);

    rootLogger.info("Testing the r^2 bookkeeping...");
    testR2Bookkeeping(ctx);

    if (ctx.lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green, "passed!", CoutColors::reset);
    }

    return 0;
}
