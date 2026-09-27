/*
 * MDWF simultaneous multishift CG test.
 *
 * Validates MDWFCoupledMultiShiftCG with MDWFMultiShiftStrategy::Simultaneous
 * (one Krylov sequence for all shifts) against exact solutions and against the
 * validated Independent strategy (one CG per shift):
 *
 *   1. Mock diagonal operator lambda(site, s) = 1 + 0.1 s + 0.001 (isite mod 97)
 *      with shifts {0, 0.01, 0.1, 1, 10}: every solution matches the exact
 *      x_i = b / (lambda + sigma_i) to relative 1e-8.
 *   2. Mobius clover M^\dagger M on a random gauge field (6^4, Ls = 8, c_sw = 0.5,
 *      b5 = 1.5), control M5 = -2 and physical-like M5 = 1.8, mf = 0.1, with 12
 *      shifts geometric in [1e-3, 1e2]: every shifted system's independently
 *      recomputed residual |b - (A + sigma_i) x_i| / |b| is <= 10 * precision,
 *      and the simultaneous and independent solutions agree to relative 1e-5
 *      (both are accurate only to the solver tolerance times the condition
 *      number). Reports operator applications of both strategies.
 *   3. The rational action (computeMDWFRationalAction, 3 terms) with
 *      Simultaneous agrees with Independent to relative 1e-9.
 *
 * Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFPseudofermionAction.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFMscgMockDiagonal {
    Vect12ArrayAcc<floatT> _in;
    floatT _power;

    template<bool onDevice>
    MDWFMscgMockDiagonal(const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &in, floatT power)
        : _in(in.getAccessor()), _power(power) {}

    __host__ __device__ static floatT lambda(const gSiteStack &site) {
        return static_cast<floatT>(1.0) + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack)
               + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite % 97);
    }

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        return lambda(site) * _in.getElement(site);
    }
};

// Exact solution b / (lambda + sigma) of the mock system.
template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFMscgMockExact {
    Vect12ArrayAcc<floatT> _in;
    floatT _sigma;

    template<bool onDevice>
    MDWFMscgMockExact(const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &in, floatT sigma)
        : _in(in.getAccessor()), _sigma(sigma) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        return (static_cast<floatT>(1.0)
                / (MDWFMscgMockDiagonal<floatT, LatLayout, HaloDepth, Ls>::lambda(site) + _sigma))
               * _in.getElement(site);
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFMscgMockOperator {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        out.template iterateOverBulk<>(MDWFMscgMockDiagonal<double, All, HaloDepth, Ls>(in, 1.0));
        if (update) {
            out.updateAll();
        }
    }
};

template<class Adapter, class Spinor>
double mdwfMscgRelDiff(Adapter &adapter, Spinor &a, const Spinor &b, Spinor &work) {
    work = a;
    work -= b;
    return std::sqrt(adapter.norm2(work) / std::max(adapter.norm2(a), 1e-300));
}

struct MDWFMscgComparison {
    double maxTrueResidual;
    double maxSolutionDiff;
    int simultaneousApplications;
    int independentApplications;
    bool converged;
};

template<size_t HaloDepth, size_t Ls, class Adapter>
MDWFMscgComparison mdwfMscgCompare(CommunicationBase &commBase, Adapter &adapter,
                                   MDWFSpinor<double, true, All, HaloDepth, Ls> &source,
                                   const std::vector<double> &sigma, int maxIter, double precision,
                                   const std::string &label) {
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    std::vector<std::unique_ptr<Spinor>> simultaneous;
    std::vector<std::unique_ptr<Spinor>> independent;
    std::vector<Spinor *> simultaneousPtrs;
    std::vector<Spinor *> independentPtrs;
    for (size_t i = 0; i < sigma.size(); i++) {
        simultaneous.emplace_back(new Spinor(commBase, label + "_simultaneous_" + std::to_string(i)));
        independent.emplace_back(new Spinor(commBase, label + "_independent_" + std::to_string(i)));
        simultaneousPtrs.push_back(simultaneous.back().get());
        independentPtrs.push_back(independent.back().get());
    }

    MDWFCoupledMultiShiftCG<double, Adapter> simultaneousCg(MDWFMultiShiftStrategy::Simultaneous);
    MDWFCoupledMultiShiftCG<double, Adapter> independentCg(MDWFMultiShiftStrategy::Independent);
    const MDWFCoupledMultiShiftCGResults<double> simResult
        = simultaneousCg.invert(adapter, simultaneousPtrs, source, sigma, maxIter, precision, true);
    const MDWFCoupledMultiShiftCGResults<double> indResult
        = independentCg.invert(adapter, independentPtrs, source, sigma, maxIter, precision, true);

    Spinor applied(commBase, label + "_check_applied");
    Spinor residual(commBase, label + "_check_residual");
    const double sourceNorm = std::sqrt(adapter.norm2(source));
    MDWFMscgComparison cmp{0.0, 0.0, 0, 0, simResult.converged() && indResult.converged()};
    for (size_t i = 0; i < sigma.size(); i++) {
        adapter.apply(applied, *simultaneous[i], false);
        applied.template axpyThisB<64>(sigma[i], *simultaneous[i]);
        residual = source;
        residual -= applied;
        const double trueResidual = std::sqrt(adapter.norm2(residual)) / sourceNorm;
        const double solutionDiff = mdwfMscgRelDiff(adapter, *independent[i], *simultaneous[i], residual);
        cmp.maxTrueResidual = std::max(cmp.maxTrueResidual, trueResidual);
        cmp.maxSolutionDiff = std::max(cmp.maxSolutionDiff, solutionDiff);
        cmp.simultaneousApplications = std::max(cmp.simultaneousApplications, simResult.shifts[i].iterations);
        cmp.independentApplications += indResult.shifts[i].iterations;
        rootLogger.info("MDWF simultaneous multishift ", label, ": sigma = ", sigma[i],
                        ", simultaneous iterations = ", simResult.shifts[i].iterations,
                        ", recurrence residue = ", simResult.shifts[i].residue,
                        ", true residual = ", trueResidual,
                        ", independent iterations = ", indResult.shifts[i].iterations,
                        ", solution relDiff = ", solutionDiff);
    }
    rootLogger.info("MDWF simultaneous multishift ", label, " summary: operator applications simultaneous = ",
                    cmp.simultaneousApplications, " versus independent = ", cmp.independentApplications,
                    " (speedup ", static_cast<double>(cmp.independentApplications)
                                  / std::max(1, cmp.simultaneousApplications),
                    "), max true residual = ", cmp.maxTrueResidual, ", max solution relDiff = ", cmp.maxSolutionDiff,
                    ", converged = ", cmp.converged);
    return cmp;
}

template<size_t Ls>
void runMDWFSimultaneousMultiShiftCGTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MockOperator = MDWFMscgMockOperator<HaloDepth, Ls>;
    using MockAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, MockOperator>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;

    const double precision = 1e-10;
    const int maxIter = 20000;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260928);
    d_rand = h_rand;

    Spinor source(commBase, "MDWF_mscg_test_source");
    Spinor work(commBase, "MDWF_mscg_test_work");
    source.gauss(d_rand.state);
    source.updateAll();

    // --- Part 1: mock operator against exact solutions. ---
    bool mockPassed = true;
    {
        MockOperator mock;
        MockAdapter mockAdapter(mock);
        const std::vector<double> mockSigma = {0.0, 0.01, 0.1, 1.0, 10.0};
        std::vector<std::unique_ptr<Spinor>> solutions;
        std::vector<Spinor *> ptrs;
        for (size_t i = 0; i < mockSigma.size(); i++) {
            solutions.emplace_back(new Spinor(commBase, "MDWF_mscg_test_mock_sol_" + std::to_string(i)));
            ptrs.push_back(solutions.back().get());
        }
        MDWFCoupledMultiShiftCG<double, MockAdapter> cg(MDWFMultiShiftStrategy::Simultaneous);
        const MDWFCoupledMultiShiftCGResults<double> result = cg.invert(mockAdapter, ptrs, source, mockSigma,
                                                                        maxIter, 1e-12, true);
        Spinor exact(commBase, "MDWF_mscg_test_mock_exact");
        double maxDiff = 0.0;
        for (size_t i = 0; i < mockSigma.size(); i++) {
            exact.template iterateOverBulk<>(MDWFMscgMockExact<double, All, HaloDepth, Ls>(source, mockSigma[i]));
            const double diff = mdwfMscgRelDiff(mockAdapter, exact, *solutions[i], work);
            maxDiff = std::max(maxDiff, diff);
            rootLogger.info("MDWF simultaneous multishift mock: sigma = ", mockSigma[i],
                            ", iterations = ", result.shifts[i].iterations, ", relDiff to exact = ", diff);
        }
        mockPassed = result.converged() && maxDiff <= 1e-8;
        rootLogger.info("MDWF simultaneous multishift mock: max relDiff to exact = ", maxDiff, ", passed = ", mockPassed);
    }

    // --- Part 2: Mobius clover M^\dagger M, control and physical-like. ---
    Gauge gauge(commBase, "MDWF_mscg_test_gauge");
    gauge.random(d_rand.state);
    gauge.updateAll();

    std::vector<double> sigma;
    for (int k = 0; k < 12; k++) {
        sigma.push_back(1e-3 * std::pow(1e5, static_cast<double>(k) / 11.0));
    }

    bool mobiusPassed = true;
    const double mobiusM5[2] = {-2.0, 1.8};
    const char *mobiusLabel[2] = {"control", "physical"};
    for (int c = 0; c < 2; c++) {
        const std::string label = std::string("MDWF_mscg_test_") + mobiusLabel[c];
        Forward forward(gauge, mobiusM5[c], 0.1, 1.5, 0.5, label + "_forward");
        Adjoint adjoint(gauge, mobiusM5[c], 0.1, 1.5, 0.5, label + "_adjoint");
        Normal normal(commBase, forward, adjoint, label + "_normal");
        NormalAdapter adapter(normal);
        const MDWFMscgComparison cmp = mdwfMscgCompare<HaloDepth, Ls>(commBase, adapter, source, sigma, maxIter,
                                                                       precision, mobiusLabel[c]);
        const bool passed = cmp.converged && cmp.maxTrueResidual <= 10.0 * precision && cmp.maxSolutionDiff <= 1e-5;
        rootLogger.info("MDWF simultaneous multishift ", mobiusLabel[c], " (M5 = ", mobiusM5[c], "): passed = ", passed);
        mobiusPassed = mobiusPassed && passed;
    }

    // --- Part 3: rational action with Simultaneous versus Independent (control). ---
    bool actionPassed = true;
    {
        Forward forward(gauge, -2.0, 0.1, 1.5, 0.5, "MDWF_mscg_test_action_forward");
        Adjoint adjoint(gauge, -2.0, 0.1, 1.5, 0.5, "MDWF_mscg_test_action_adjoint");
        Normal normal(commBase, forward, adjoint, "MDWF_mscg_test_action_normal");
        NormalAdapter adapter(normal);
        MDWFExplicitRationalInput<double> input{
            "mscg_test_action", MDWFRationalCoefficientRole::Action, 0.125, {0.5, 0.25, 0.125}, {0.0, 0.1, 0.3}};
        const MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(input);
        Spinor workspaceSim(commBase, "MDWF_mscg_test_action_ws_sim");
        Spinor workspaceInd(commBase, "MDWF_mscg_test_action_ws_ind");
        const MDWFRationalActionResult<double> sim = computeMDWFRationalAction<double, NormalAdapter>(
            adapter, workspaceSim, source, coefficients, maxIter, precision, "MDWF_mscg_test_action_sim",
            MDWFMultiShiftStrategy::Simultaneous);
        const MDWFRationalActionResult<double> ind = computeMDWFRationalAction<double, NormalAdapter>(
            adapter, workspaceInd, source, coefficients, maxIter, precision, "MDWF_mscg_test_action_ind",
            MDWFMultiShiftStrategy::Independent);
        const double actionRelDiff = std::abs(sim.action_real - ind.action_real) / std::max(1.0, std::abs(ind.action_real));
        actionPassed = sim.rational_result.converged() && ind.rational_result.converged() && actionRelDiff <= 1e-9;
        rootLogger.info("MDWF simultaneous multishift rational action: simultaneous = ", sim.action_real,
                        ", independent = ", ind.action_real, ", relDiff = ", actionRelDiff, ", passed = ", actionPassed);
    }

    if (!mockPassed || !mobiusPassed || !actionPassed) {
        throw std::runtime_error(stdLogger.fatal("MDWF simultaneous multishift CG test failed: mock passed = ",
                                                 mockPassed, ", Mobius passed = ", mobiusPassed,
                                                 ", rational action passed = ", actionPassed));
    }
    rootLogger.info("MDWF simultaneous multishift CG test passed with Ls = ", Ls);
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

        runMDWFSimultaneousMultiShiftCGTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
