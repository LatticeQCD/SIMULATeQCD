/*
 * MDWF even/odd decomposition and Schur-complement solver test
 * (MDWFMobiusEvenOdd.h, EVEN_ODD_DESIGN.md stage E1).
 *
 * Random gauge field, 6^4, Ls = 8, Mobius M5 = 1.8, b5 = 1.5, mf = 0.05, for
 * c_sw = 0.5 and c_sw = 0:
 *
 *   1. split/merge round trip of an All-layout MDWF spinor is exact.
 *   2. Block reconstruction: [M_ee v_e + M_eo v_o, M_oe v_e + M_oo v_o] equals
 *      MDWFMobiusCloverLinearOperator, and [M_ee^+ v_e + M_oe^+ v_o,
 *      M_eo^+ v_e + M_oo^+ v_o] equals MDWFMobiusCloverAdjointLinearOperator
 *      (relative 1e-13).
 *   3. Block inverse: M_oo^-1 M_oo v = v, M_oo M_oo^-1 v = v, and the same for
 *      the adjoint, on odd and even sites (relative 1e-12).
 *   4. Schur complement adjoint identity <x, Mhat y> = <Mhat^+ x, y> (1e-12).
 *   5. Solves against the unpreconditioned CG on M^+ M (MDWFCoupledCG,
 *      MDWFNormalOperator), precision 1e-10 for both:
 *        M x = b:          true residual |M x - b| / |b| <= 1e-7,
 *                          agreement with x_ref = (M^+ M)^-1 M^+ b to 1e-6;
 *        M^+ M x = b:      true residual <= 1e-7, agreement with the
 *                          unpreconditioned normal solve to 1e-6;
 *      with CG iterations, wall-clock times, and the speedup reported.
 *
 * Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusEvenOdd.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

template<class Spinor>
double mdwfEoNorm2(Spinor &s) {
    double sum = 0.0;
    for (const COMPLEX(double) &d : s.dotProductStacked(s)) {
        sum += real<double>(d);
    }
    return sum;
}

template<class Spinor>
COMPLEX(double) mdwfEoDot(Spinor &a, const Spinor &b) {
    COMPLEX(double) sum = 0.0;
    for (const COMPLEX(double) &d : a.dotProductStacked(b)) {
        sum += d;
    }
    return sum;
}

// |a - b| / |b| using work as scratch.
template<class Spinor>
double mdwfEoRelDiff(Spinor &work, const Spinor &a, Spinor &b) {
    work = a;
    work.template axpyThisB<64>(-1.0, b);
    return std::sqrt(mdwfEoNorm2(work) / mdwfEoNorm2(b));
}

double mdwfEoSeconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template<size_t Ls>
bool runMDWFMobiusEvenOddCase(CommunicationBase &commBase, Gaugefield<double, true, 2, R18> &gauge, double csw,
                              uint4 *randState) {
    const size_t HaloDepth = 2;
    using Solver = MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>;
    using EvenOdd = typename Solver::EvenOdd;
    using SpinorAll = typename Solver::SpinorAll;
    using SpinorE = typename Solver::SpinorE;
    using SpinorO = typename Solver::SpinorO;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using CG = MDWFCoupledCG<double, NormalAdapter>;

    const double M5 = 1.8;
    const double mf = 0.05;
    const double b5 = 1.5;
    const int maxIter = 20000;
    const double precision = 1e-10;
    // Field names below avoid being substrings of earlier live names (getSmartName aliasing, AGENTS.md).
    const std::string tag = "MDWF_eo_test_c" + std::to_string(static_cast<int>(csw * 10));

    Forward forward(gauge, M5, mf, b5, csw, tag + "_forward");
    Adjoint adjoint(gauge, M5, mf, b5, csw, tag + "_adjoint");
    Normal normal(commBase, forward, adjoint, tag + "_normal");
    NormalAdapter normalAdapter(normal);
    Solver solver(gauge, M5, mf, b5, csw, tag + "_solver");
    EvenOdd &eo = solver.evenOdd();
    eo.refresh();

    SpinorAll v(commBase, tag + "_v");
    SpinorAll w(commBase, tag + "_w");
    SpinorAll ref(commBase, tag + "_ref");
    SpinorAll blocks(commBase, tag + "_blocks");
    SpinorAll work(commBase, tag + "_scratch");
    SpinorE ve(commBase, tag + "_ve"), ae(commBase, tag + "_ae"), be(commBase, tag + "_be"), we(commBase, tag + "_we");
    SpinorO vo(commBase, tag + "_vo"), ao(commBase, tag + "_ao"), bo(commBase, tag + "_bo"), wo(commBase, tag + "_wo");

    v.gauss(randState);
    v.updateAll();
    rootLogger.info("MDWF even/odd test: M5 = ", M5, ", mf = ", mf, ", b5 = ", b5, ", c_sw = ", csw, ", Ls = ", Ls,
                    ", mass = ", eo.params().mass);

    // --- 1. split / merge round trip. ---
    EvenOdd::split(ve, vo, v);
    EvenOdd::merge(w, ve, vo);
    const double roundTrip = mdwfEoRelDiff(work, w, v);

    // --- 2. block reconstruction of M and M^dagger. ---
    ve.updateAll();
    vo.updateAll();
    eo.Mee(ae, ve);
    eo.Meo(be, vo);
    ae += be;
    eo.Moe(ao, ve);
    eo.Moo(bo, vo);
    ao += bo;
    EvenOdd::merge(blocks, ae, ao);
    forward.apply(ref, v, true);
    const double forwardDiff = mdwfEoRelDiff(work, blocks, ref);

    eo.Mee(ae, ve, true);
    eo.MoeDagger(be, vo);
    ae += be;
    eo.MeoDagger(ao, ve);
    eo.Moo(bo, vo, true);
    ao += bo;
    EvenOdd::merge(blocks, ae, ao);
    adjoint.apply(ref, v, true);
    const double adjointDiff = mdwfEoRelDiff(work, blocks, ref);

    // --- 3. block inverse. ---
    double inverseDiff = 0.0;
    for (int dagger = 0; dagger < 2; dagger++) {
        eo.Moo(ao, vo, dagger);
        eo.MooInv(bo, ao, dagger);
        inverseDiff = std::max(inverseDiff, mdwfEoRelDiff(wo, bo, vo));
        eo.MooInv(ao, vo, dagger);
        eo.Moo(bo, ao, dagger);
        inverseDiff = std::max(inverseDiff, mdwfEoRelDiff(wo, bo, vo));
        eo.Mee(ae, ve, dagger);
        eo.MeeInv(be, ae, dagger);
        inverseDiff = std::max(inverseDiff, mdwfEoRelDiff(we, be, ve));
    }

    // --- 4. Schur adjoint identity. ---
    SpinorE xe(commBase, tag + "_xe"), ye(commBase, tag + "_ye");
    SpinorAll tmpAll(commBase, tag + "_tmp_all");
    tmpAll.gauss(randState);
    EvenOdd::split(xe, wo, tmpAll);
    tmpAll.gauss(randState);
    EvenOdd::split(ye, wo, tmpAll);
    eo.schur(ae, ye, false);
    eo.schur(be, xe, true);
    const COMPLEX(double) left = mdwfEoDot(xe, ae);
    const COMPLEX(double) right = mdwfEoDot(be, ye);
    const double schurAdjointDiff = abs(left - right) / std::max(abs(left), 1e-300);

    const bool algebraPassed = roundTrip == 0.0 && forwardDiff <= 1e-13 && adjointDiff <= 1e-13
                               && inverseDiff <= 1e-12 && schurAdjointDiff <= 1e-12;
    rootLogger.info("MDWF even/odd test algebra (c_sw = ", csw, "): split/merge = ", roundTrip,
                    ", |M_blocks v - M v| / |M v| = ", forwardDiff, ", adjoint = ", adjointDiff,
                    ", max block-inverse error = ", inverseDiff, ", Schur <x, Mhat y> vs <Mhat^+ x, y> = ",
                    schurAdjointDiff, ", passed = ", algebraPassed);

    // --- 5a. M x = b. ---
    SpinorAll b(commBase, tag + "_src");
    SpinorAll x(commBase, tag + "_xsol");
    SpinorAll xRef(commBase, tag + "_xref");
    SpinorAll rhs(commBase, tag + "_rhs");
    b.gauss(randState);
    b.updateAll();

    auto start = std::chrono::steady_clock::now();
    solver.solve(x, b, maxIter, precision);
    const double eoSolveSeconds = mdwfEoSeconds(start);
    const int eoSolveIterations = solver.lastIterations();
    forward.apply(ref, x, true);
    const double solveResidual = mdwfEoRelDiff(work, ref, b);

    adjoint.apply(rhs, b, true);
    start = std::chrono::steady_clock::now();
    CG cg;
    const MDWFCoupledCGResult<double> refResult = cg.invert(normalAdapter, xRef, rhs, maxIter, precision, true);
    const double refSolveSeconds = mdwfEoSeconds(start);
    const double solveAgreement = mdwfEoRelDiff(work, x, xRef);
    const bool solvePassed = refResult.converged && solveResidual <= 1e-7 && solveAgreement <= 1e-6;
    rootLogger.info("MDWF even/odd test M x = b (c_sw = ", csw, "): even/odd CG iterations = ", eoSolveIterations,
                    " (", eoSolveSeconds, " s), true residual = ", solveResidual,
                    "; unpreconditioned M^+M CG iterations = ", refResult.iterations, " (", refSolveSeconds,
                    " s); |x - x_ref| / |x_ref| = ", solveAgreement, ", speedup = ", refSolveSeconds / eoSolveSeconds,
                    ", passed = ", solvePassed);

    // --- 5b. M^+ M x = b. ---
    start = std::chrono::steady_clock::now();
    solver.solveNormal(x, w, b, maxIter, precision);
    const double eoNormalSeconds = mdwfEoSeconds(start);
    const int eoNormalIterations = solver.lastIterations();
    normal.apply(ref, x, true);
    const double normalResidual = mdwfEoRelDiff(work, ref, b);

    start = std::chrono::steady_clock::now();
    const MDWFCoupledCGResult<double> refNormal = cg.invert(normalAdapter, xRef, b, maxIter, precision, true);
    const double refNormalSeconds = mdwfEoSeconds(start);
    const double normalAgreement = mdwfEoRelDiff(work, x, xRef);
    const bool normalPassed = refNormal.converged && normalResidual <= 1e-7 && normalAgreement <= 1e-6;
    rootLogger.info("MDWF even/odd test M^+M x = b (c_sw = ", csw, "): even/odd CG iterations = ", eoNormalIterations,
                    " (two solves, ", eoNormalSeconds, " s), true residual = ", normalResidual,
                    "; unpreconditioned CG iterations = ", refNormal.iterations, " (", refNormalSeconds,
                    " s); |x - x_ref| / |x_ref| = ", normalAgreement, ", speedup = ",
                    refNormalSeconds / eoNormalSeconds, ", passed = ", normalPassed);

    return algebraPassed && solvePassed && normalPassed;
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261002);
        d_rand = h_rand;

        Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_eo_test_gauge");
        gauge.random(d_rand.state);
        gauge.updateAll();

        const bool cloverPassed = runMDWFMobiusEvenOddCase<8>(commBase, gauge, 0.5, d_rand.state);
        const bool wilsonPassed = runMDWFMobiusEvenOddCase<8>(commBase, gauge, 0.0, d_rand.state);
        if (!cloverPassed || !wilsonPassed) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd test failed: c_sw = 0.5 passed = ", cloverPassed,
                                                     ", c_sw = 0 passed = ", wilsonPassed));
        }
        rootLogger.info("MDWF even/odd test passed");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
