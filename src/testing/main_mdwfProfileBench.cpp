/*
 * MDWF profiling benchmark (tests/profile): the even/odd Mobius operator pieces and the CG solves of the light sector,
 * measured the same way as the GPT driver tests/profile/gpt/gpt_profile.py, on 1 or more GPUs (Nodes).
 *
 * Operator part (double and single precision, c_sw = 0, Ls = 16, fused Schur path as in production):
 *   meo     M_eo (hop odd -> even with the fifth-dimension factor Din)            GPT: Meooe.mat (odd -> even)
 *   mhat    Mhat = M_ee - M_eo M_oo^-1 M_oe                                       GPT: MMdag_evenodd.M
 *   mhatdag Mhat^+                                                               GPT: MMdag_evenodd.Mdag
 *   mdagm   Mhat^+ Mhat (F3 / light_form 0 normal operator)
 *   mmdag   Mhat Mhat^+ (light_form 1 = GPT production normal operator)          GPT: MMdag_evenodd.MMdag
 * ms per application after n_warm warm-up calls (mean over n_apply, device synchronised before and after). The
 * flop and byte models are applied by tests/profile/summarize.py, identically for both codes.
 *
 * CG part: for every mass in cg_masses and every tolerance in cg_precisions, solve (Mhat Mhat^+) x = b (mmdag = 1,
 * default) or (Mhat^+ Mhat) x = b from x = 0, with b = the shared hash Gaussian (MDWFHashFields.h, seed noise_seed,
 * vector 0 + s on slice s, scale 1/sqrt(2)) on the even sites; GPT fills the same b, so |b|^2 must agree to rounding
 * and the double-precision CG iteration counts should agree closely.
 *   double: MDWFCoupledCG, stop at |r_iter|^2 <= eps^2 max(|b|^2, 1) (GPT cg: |r_iter|^2 <= eps^2 |b|^2, same here
 *           since |b|^2 >> 1).
 *   mixed:  production solver MDWFEvenOddMixedSchurSolver (single-precision CG with reliable updates, mixed_delta),
 *           stops on the true double residual. GPT production: defect_correcting(mixed_precision(cg single)).
 * Each solve runs n_cg_repeat times (first includes set-up); the true residual |b - A x| / |b| is recomputed in double.
 *
 * Parameters: Lattice, Nodes, Gaugefile (optional; random links otherwise), M5, b5, mf (operator part), cg_masses,
 * cg_precisions (lists: give them in the file only, command-line lists append), max_iter, mixed_delta, mmdag,
 * noise_seed, n_apply, n_warm, n_cg_repeat, run_ops, run_cg, force_halos. Output lines start with "PROF sq ".
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHashFields.h"
#include "../experimental/mdwf/MDWFMixedPrecisionSolver.h"
#include "../experimental/mdwf/MDWFMobiusEvenOdd.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFProfileBenchParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> mf;
    DynamicParameter<double> cg_masses;
    DynamicParameter<double> cg_precisions;
    Parameter<int> max_iter;
    Parameter<double> mixed_delta;
    Parameter<int> mmdag;
    Parameter<std::string> noise_seed;
    Parameter<int> n_apply;
    Parameter<int> n_warm;
    Parameter<int> n_cg_repeat;
    Parameter<int> run_ops;
    Parameter<int> run_cg;
    Parameter<int> force_halos;

    MDWFProfileBenchParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 1.9);
        addDefault(mf, "mf", 0.00084525);
        addOptional(cg_masses, "cg_masses");
        addOptional(cg_precisions, "cg_precisions");
        addDefault(max_iter, "max_iter", 100000);
        addDefault(mixed_delta, "mixed_delta", 0.1);
        addDefault(mmdag, "mmdag", 1);
        addDefault(noise_seed, "noise_seed", std::string("profile"));
        addDefault(n_apply, "n_apply", 200);
        addDefault(n_warm, "n_warm", 10);
        addDefault(n_cg_repeat, "n_cg_repeat", 2);
        addDefault(run_ops, "run_ops", 1);
        addDefault(run_cg, "run_cg", 1);
        addDefault(force_halos, "force_halos", 0);
    }
};

static double profSeconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void profSync(const char *where) {
    if (gpuDeviceSynchronize() != gpuSuccess) {
        throw std::runtime_error(stdLogger.fatal("device synchronize failed ", where));
    }
}

template<class F>
static double profTimePerCall(F &&f, int nWarm, int n) {
    for (int i = 0; i < nWarm; i++) {
        f();
    }
    profSync("before timing");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        f();
    }
    profSync("after timing");
    return profSeconds(t0) / n;
}

template<class floatT, size_t HaloDepth, size_t Ls>
void profileOperators(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gaugeD,
                      MDWFProfileBenchParameters &param, uint4 *randState) {
    using EvenOdd = MDWFMobiusCloverEvenOdd<floatT, HaloDepth, HaloDepth, Ls>;
    using EvenOddD = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;
    constexpr bool isDouble = std::is_same<floatT, double>::value;
    const char *prec = isDouble ? "double" : "single";

    EvenOddD eoD(gaugeD, param.M5(), param.mf(), param.b5(), 0.0, "prof_eoD");
    eoD.refresh();
    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase, std::string("prof_gauge_") + prec);
    gauge.convert_precision(gaugeD);
    mdwfUpdateHalo(gauge);
    EvenOdd eo(gauge, param.M5(), param.mf(), param.b5(), 0.0, std::string("prof_eo_") + prec);
    if constexpr (isDouble) {
        eo.refresh();
    } else {
        eo.refreshFrom(eoD);
    }
    NormalOp normal(eo, commBase, "prof_normal");

    SpinorE in(commBase, std::string("prof_in_") + prec), out(commBase, std::string("prof_out_") + prec);
    SpinorO inO(commBase, std::string("prof_inO_") + prec);
    in.gauss(randState);
    inO.gauss(randState);
    mdwfUpdateHalo(in);
    mdwfUpdateHalo(inO);

    const int nw = param.n_warm(), na = param.n_apply();
    const double tMeo = profTimePerCall([&] { eo.Meo(out, inO); }, nw, na);
    const double tMhat = profTimePerCall([&] { eo.schur(out, in, false); }, nw, na);
    const double tMhatDag = profTimePerCall([&] { eo.schur(out, in, true); }, nw, na);
    normal.setMMdag(false);
    const double tMdagM = profTimePerCall([&] { normal.apply(out, in); }, nw, na);
    normal.setMMdag(true);
    const double tMMdag = profTimePerCall([&] { normal.apply(out, in); }, nw, na);
    const double tCopy = profTimePerCall([&] { out = in; }, nw, na);
    const double fieldMB = 0.5 * GIndexer<All, HaloDepth>::getLatData().globvol4 / commBase.getNumberProcesses() * Ls
                           * 24.0 * sizeof(floatT) / 1e6;
    rootLogger.info("PROF sq op ", prec, " ranks ", commBase.getNumberProcesses(), " ms meo ", 1e3 * tMeo, " mhat ",
                    1e3 * tMhat, " mhatdag ", 1e3 * tMhatDag, " mdagm ", 1e3 * tMdagM, " mmdag ", 1e3 * tMMdag,
                    " copy ", 1e3 * tCopy, " field_MB_per_rank ", fieldMB);
}

template<size_t HaloDepth, size_t Ls>
void profileCG(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gauge,
               MDWFProfileBenchParameters &param) {
    using EvenOdd = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOp>;
    using SpinorE = typename EvenOdd::SpinorE;
    using Mixed = MDWFEvenOddMixedSchurSolver<HaloDepth, Ls>;

    const bool mmdag = param.mmdag() != 0;
    const std::vector<double> masses = param.cg_masses.isSet() ? param.cg_masses.get()
                                                               : std::vector<double>{0.00084525, 0.006};
    const std::vector<double> precs = param.cg_precisions.isSet() ? param.cg_precisions.get()
                                                                  : std::vector<double>{1e-10, 1e-8};

    SpinorE b(commBase, "prof_cg_b"), x(commBase, "prof_cg_x"), ax(commBase, "prof_cg_ax");
    b.template iterateOverBulk<BLOCKSIZE>(
        MDWFHashSpinor<Even, HaloDepth>(mdwfSeedFromString(param.noise_seed()), 0, 1.0 / std::sqrt(2.0)));
    mdwfUpdateHalo(b);

    for (const double m : masses) {
        EvenOdd eo(gauge, param.M5(), m, param.b5(), 0.0, "prof_cg_eo");
        eo.refresh();
        NormalOp normal(eo, commBase, "prof_cg_normal");
        normal.setMMdag(mmdag);
        Adapter adapter(normal);
        Mixed mixed(gauge, param.M5(), param.b5(), 0.0, param.mixed_delta(), "prof_cg_mixed");
        const double b2 = adapter.norm2(b);
        rootLogger.info("PROF sq cgsource mass ", m, " op ", mmdag ? "mmdag" : "mdagm", " b2 ", b2);

        auto trueResidual = [&]() {
            normal.apply(ax, x, false);
            ax.template axpyThisB<64>(-1.0, b);
            return std::sqrt(adapter.norm2(ax) / b2);
        };
        for (const double eps : precs) {
            for (int rep = 0; rep < param.n_cg_repeat(); rep++) {
                profSync("before double CG");
                auto t0 = std::chrono::steady_clock::now();
                MDWFCoupledCG<double, Adapter> cg;
                const MDWFCoupledCGResult<double> rd = cg.invert(adapter, x, b, param.max_iter(), eps, true);
                profSync("after double CG");
                const double sd = profSeconds(t0);
                rootLogger.info("PROF sq cg double mass ", m, " eps ", eps, " rep ", rep, " iters ", rd.iterations,
                                " seconds ", sd, " ms_per_iter ", 1e3 * sd / std::max(rd.iterations, 1),
                                " converged ", rd.converged ? 1 : 0, " true_res ", trueResidual());

                profSync("before mixed CG");
                t0 = std::chrono::steady_clock::now();
                const MDWFCoupledMixedCGResult rm = mixed.solve(eo, adapter, x, b, param.max_iter(), eps, nullptr, mmdag);
                profSync("after mixed CG");
                const double sm = profSeconds(t0);
                rootLogger.info("PROF sq cg mixed mass ", m, " eps ", eps, " rep ", rep, " iters ", rm.iterations,
                                " reliable ", rm.reliableUpdates, " seconds ", sm, " ms_per_iter ",
                                1e3 * sm / std::max(rm.iterations, 1), " converged ", rm.converged ? 1 : 0,
                                " true_res ", trueResidual());
            }
        }
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFProfileBenchParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfProfileBench.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261017);
        d_rand = h_rand;

        Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "prof_gauge");
        if (param.GaugefileName.isSet() && !param.GaugefileName().empty()) {
            gauge.readconf_nersc(param.GaugefileName());
        } else {
            gauge.random(d_rand.state);
        }
        gauge.updateAll();

        rootLogger.info("PROF sq setup lattice ", param.latDim()[0], "x", param.latDim()[1], "x", param.latDim()[2],
                        "x", param.latDim()[3], " nodes ", param.nodeDim()[0], param.nodeDim()[1], param.nodeDim()[2],
                        param.nodeDim()[3], " ranks ", commBase.getNumberProcesses(), " Ls 16 M5 ", param.M5(),
                        " b5 ", param.b5(), " mf ", param.mf(), " n_apply ", param.n_apply());
        if (param.run_ops() != 0) {
            profileOperators<double, HaloDepth, 16>(commBase, gauge, param, d_rand.state);
            profileOperators<float, HaloDepth, 16>(commBase, gauge, param, d_rand.state);
        }
        if (param.run_cg() != 0) {
            profileCG<HaloDepth, 16>(commBase, gauge, param);
        }
        rootLogger.info("PROF sq done");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
