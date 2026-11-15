/*
 * Stage 0 of the MDWF multigrid project (DSDRMDWFruns/SQCDruns/multigrid/DESIGN.md): two-grid study, single rank.
 * Tools: MDWFTwoGrid.h (coarse space, probing, dense coarse solve, Chebyshev smoother, FCG).
 *
 * Operator A on the even sites: normal_form = 0  Mhat^+ Mhat (the measurement solves, mdwfMeasure),
 *                                normal_form = 1  Mhat Mhat^+ (HMC light_form = 1 force solves, the deflation dumps).
 *
 *   mode = selftest  unit tests on the configured (small) lattice: U1 R P = 1, U2 A_c = R A P on random coarse vectors,
 *                    U3 the coarse correction reproduces an error inside the coarse space, U4 two-grid FCG solution =
 *                    plain CG solution. Exit code 1 on failure. Use e.g. Lattice 8 8 8 8, ls 8, block 4,4,4,4.
 *   mode = study     gauge field: Gaugefile (NERSC) or the deflation dump (dump_dir, dump_eval >= 0). Right-hand sides:
 *                      rhs = point  point_source "x,y,z,t", point_components (0..11): the even/odd-prepared right-hand
 *                                   side of the measurement solve M x = D_- source (normal_form 0 only)
 *                      rhs = dump   the dumped force right-hand side of term dump_term at every eval of solve_evals
 *                      rhs = noise  n_noise hash-noise vectors
 *                    Near-null vectors (setup): cheb (Chebyshev low-pass filter of degree setup_n on hash noise),
 *                    inviter (setup_n plain-CG iterations on A x = noise), exact (converted eigenvectors in exact_dir).
 *                    For every ns in ns_list and smoother degree d in degrees: setup, A_c by probing, then the
 *                    two-grid FCG for every right-hand side; plain CG (same stopping rule) for the first plain_cg
 *                    right-hand sides of every eval. With rhs = dump and several solve_evals, the space of the first
 *                    eval is kept and every later eval is solved twice: variant "stale" (old A_c) and "fresh" (A_c
 *                    probed again with the new A, i.e. stale vectors).
 *                    Rows to <out_dir>/twogrid_<tag>.tsv.
 *   mode = convert   (any node layout) eigenvector set <set_base> (mdwfDeflWriteSet format, needs the writer's layout)
 *                    -> <convert_out>/v<k> in the layout-free MDWFFieldDump format, k < set_count, and
 *                    <convert_out>/lambda.tsv.
 *
 * Usage: ./mdwfTwoGridStudy <param file> [key=value ...]   (parameter/tests/mdwfTwoGridStudy.param). List parameters
 * belong in the file (on the command line they are appended).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFTwoGrid.h"
#include "../experimental/mdwf/MDWFDeflationDump.h"
#include "../experimental/mdwf/MDWFResidualMass.h"
#include "../experimental/mdwf/MDWFMobiusEvenOdd.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFTwoGridStudyParameters : public LatticeParameters {
public:
    Parameter<std::string> mode;
    Parameter<int> ls;
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> mass;
    Parameter<int> normal_form;
    Parameter<int> antiperiodic_t;
    Parameter<std::string> start;
    Parameter<int> seed;
    Parameter<std::string> dump_dir;
    Parameter<int> dump_eval;
    Parameter<int> dump_term;
    DynamicParameter<int> solve_evals;
    Parameter<std::string> rhs;
    Parameter<std::string> point_source;
    DynamicParameter<int> point_components;
    Parameter<int> n_noise;
    Parameter<std::string> block;
    DynamicParameter<int> ns_list;
    DynamicParameter<int> degrees;
    Parameter<std::string> setup;
    Parameter<int> setup_n;
    Parameter<double> setup_lo;
    Parameter<double> setup_ratio;
    Parameter<double> smoother_lo;
    Parameter<double> smoother_ratio;
    DynamicParameter<double> smoother_ratios;
    Parameter<std::string> exact_dir;
    Parameter<double> precision;
    Parameter<int> max_iter;
    Parameter<int> plain_cg;
    Parameter<std::string> out_dir;
    Parameter<std::string> tag;
    Parameter<int> max_coarse_dim;
    Parameter<int> force_halos;
    Parameter<std::string> set_base;
    Parameter<int> set_count;
    Parameter<std::string> convert_out;

    MDWFTwoGridStudyParameters() {
        add(mode, "mode");
        addDefault(ls, "ls", 24);
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 2.5);
        addDefault(mass, "mass", 0.01);
        addDefault(normal_form, "normal_form", 0);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(start, "start", std::string("random"));
        addDefault(seed, "seed", 20261115);
        addDefault(dump_dir, "dump_dir", std::string("."));
        addDefault(dump_eval, "dump_eval", -1);
        addDefault(dump_term, "dump_term", 0);
        addOptional(solve_evals, "solve_evals");
        addDefault(rhs, "rhs", std::string("noise"));
        addDefault(point_source, "point_source", std::string("0,0,0,0"));
        addOptional(point_components, "point_components");
        addDefault(n_noise, "n_noise", 2);
        addDefault(block, "block", std::string("4,4,4,8"));
        addOptional(ns_list, "ns_list");
        addOptional(degrees, "degrees");
        addDefault(setup, "setup", std::string("inviter"));
        addDefault(setup_n, "setup_n", 50);
        addDefault(setup_lo, "setup_lo", 0.0);
        addDefault(setup_ratio, "setup_ratio", 1000.0);
        addDefault(smoother_lo, "smoother_lo", 0.0);
        addDefault(smoother_ratio, "smoother_ratio", 30.0);
        addOptional(smoother_ratios, "smoother_ratios");
        addDefault(exact_dir, "exact_dir", std::string(""));
        addDefault(precision, "precision", 1e-8);
        addDefault(max_iter, "max_iter", 200000);
        addDefault(plain_cg, "plain_cg", 1);
        addDefault(out_dir, "out_dir", std::string("."));
        addDefault(tag, "tag", std::string("study"));
        addDefault(max_coarse_dim, "max_coarse_dim", 33000);
        addDefault(force_halos, "force_halos", 0);
        addDefault(set_base, "set_base", std::string(""));
        addDefault(set_count, "set_count", 0);
        addDefault(convert_out, "convert_out", std::string("."));
    }
};

template<class P>
std::vector<int> mdwfTGList(P &parameter, const std::vector<int> &fallback) {
    if (parameter.isSet() && parameter.numberValues() > 0) {
        return parameter.get();
    }
    return fallback;
}

inline std::vector<int> mdwfTGInts(const std::string &s) {
    std::vector<int> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) {
        out.push_back(std::stoi(item));
    }
    return out;
}

template<size_t Ls>
void runMDWFTwoGridStudy(CommunicationBase &commBase, MDWFTwoGridStudyParameters &param) {
    constexpr size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using TG = MDWFTwoGrid<HaloDepth, Ls>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using EvenOdd = typename Op::EvenOdd;
    using C = std::complex<double>;

    const std::string mode = param.mode();
    const bool mmdag = param.normal_form() == 1;
    const bool antiperiodic = param.antiperiodic_t() != 0;
    const uint64_t seed = static_cast<uint64_t>(param.seed());

    // ---------------------------------------------------------------- convert (any layout)
    if (mode == "convert") {
        Set set;
        std::vector<double> lambda;
        const int count = param.set_count();
        mdwfDeflReadSet<HaloDepth, Ls>(commBase, set, count, lambda, param.set_base());
        typename Op::Field field(commBase, "MDWF_tg_conv_field");
        for (int k = 0; k < count; k++) {
            set[k]->unpack(field);
            mdwfDumpEvenSpinor<HaloDepth, Ls>(commBase, field, param.convert_out() + "/v" + std::to_string(k),
                                              "MDWF_tg_conv_stage");
        }
        if (commBase.IamRoot()) {
            std::ofstream out(param.convert_out() + "/lambda.tsv");
            out << std::setprecision(17) << "# k\tlambda\t(from " << param.set_base() << ")\n";
            for (int k = 0; k < count; k++) {
                out << k << "\t" << lambda[k] << "\n";
            }
        }
        rootLogger.info("MDWF two-grid study: converted ", count, " vectors of ", param.set_base(), " to ",
                        param.convert_out());
        return;
    }

    // ---------------------------------------------------------------- gauge field and operator
    Gauge thin(commBase, "MDWF_tg_thin");
    Gauge fermionGauge(commBase, "MDWF_tg_fgauge");
    auto loadGauge = [&](int eval) {
        if (eval >= 0) {
            mdwfReadGauge<HaloDepth>(commBase, thin, mdwfDeflationDumpEval(param.dump_dir(), eval) + "_U",
                                     "MDWF_tg_stage_u");
        } else if (param.GaugefileName.isSet()) {
            rootLogger.info("MDWF two-grid study: reading NERSC configuration ", param.GaugefileName());
            thin.readconf_nersc(param.GaugefileName());
        } else if (param.start() == "unit") {
            thin.one();
        } else {
            grnd_state<false> h_rand;
            grnd_state<true> d_rand;
            h_rand.make_rng_state(param.seed());
            d_rand = h_rand;
            thin.random(d_rand.state);
        }
        thin.updateAll();
        mdwfFermionGaugeField<HaloDepth>(fermionGauge, thin, antiperiodic);
    };
    const int setupEval = param.dump_eval();
    loadGauge(setupEval);
    Op op(fermionGauge, param.M5(), param.mass(), param.b5(), mmdag, "MDWF_tg_op");
    op.refresh();

    Vec start(commBase);
    start.noise(seed ^ 0x5eedULL);
    const double hi = 1.1 * mdwfDeflLambdaMax(op, start, 60);
    const double smLo = param.smoother_lo() > 0.0 ? param.smoother_lo() : hi / param.smoother_ratio();
    rootLogger.info("MDWF two-grid study: mode ", mode, ", Ls ", Ls, ", mass ", param.mass(), ", operator ",
                    mmdag ? "Mhat Mhat^+" : "Mhat^+ Mhat", ", lambda_max <~ ", hi / 1.1, ", smoother [", smLo, ", ",
                    hi, "]");

    // ---------------------------------------------------------------- selftest
    if (mode == "selftest") {
        bool pass = true;
        const std::vector<int> block = mdwfTGInts(param.block());
        const std::vector<int> nsList = mdwfTGList(param.ns_list, {4});
        TG tg(commBase, block);
        Set raw;
        mdwfDeflAllocate(raw, nsList[0], commBase);
        for (int k = 0; k < nsList[0]; k++) {   // cheap near-null vectors: a few CG iterations on noise
            Vec b(commBase);
            b.noise(seed + 17 * k);
            mdwfDeflCGSnapshots<Op, HaloDepth, Ls>(op, b, {10},
                [&](int n, const Vec &v, double) { if (n == 10) { *raw[k] = v; } }, commBase);
        }
        MDWFTwoGridStats st;
        tg.setBasis(raw, nsList[0], st);
        std::vector<C> ac;
        tg.buildCoarse(op, st, &ac);
        const int n = tg.coarseDim();
        // random coarse vector
        std::vector<C> c(n), back(n);
        uint64_t h = seed;
        for (int i = 0; i < n; i++) {
            h = mdwfDeflSplitMix64(h);
            const double a = static_cast<double>(h >> 11) / 9007199254740992.0;
            h = mdwfDeflSplitMix64(h);
            const double b = static_cast<double>(h >> 11) / 9007199254740992.0;
            c[i] = C(a - 0.5, b - 0.5);
        }
        Vec v(commBase), av(commBase), z(commBase);
        gpuMemcpy(tg.c2(), c.data(), sizeof(C) * n, gpuMemcpyHostToDevice);
        tg.prolong(v, tg.c2(), false);
        tg.restrict_(tg.c1(), v);
        gpuMemcpy(back.data(), tg.c1(), sizeof(C) * n, gpuMemcpyDeviceToHost);
        double u1 = 0.0, cmax = 0.0;
        for (int i = 0; i < n; i++) {
            u1 = std::max(u1, std::abs(back[i] - c[i]));
            cmax = std::max(cmax, std::abs(c[i]));
        }
        u1 /= cmax;
        // U2: A_c c vs R A P c
        op.apply(av, v);
        tg.restrict_(tg.c1(), av);
        gpuMemcpy(back.data(), tg.c1(), sizeof(C) * n, gpuMemcpyDeviceToHost);
        double u2 = 0.0, rmax = 0.0;
        for (int i = 0; i < n; i++) {
            C s(0.0, 0.0);
            for (int j = 0; j < n; j++) {
                s += ac[i + static_cast<size_t>(j) * n] * c[j];
            }
            u2 = std::max(u2, std::abs(s - back[i]));
            rmax = std::max(rmax, std::abs(back[i]));
        }
        u2 /= rmax;
        // U3: coarse correction of r = A P c returns P c
        tg.coarseCorrection(z, av);
        z.axpy(MDWFDeflC(-1.0, 0.0), v);
        const double u3 = std::sqrt(z.norm2() / v.norm2());
        // U4: two-grid FCG = plain CG
        const std::vector<int> degrees = mdwfTGList(param.degrees, {4});
        tg.setSmoother(smLo, hi, degrees[0]);
        Vec b(commBase), x1(commBase), x2(commBase);
        b.noise(seed + 99);
        const MDWFTwoGridSolveResult r1 = mdwfTwoGridPlainCG(op, x1, b, 1e-12, param.max_iter(), commBase);
        const MDWFTwoGridSolveResult r2 = mdwfTwoGridFCG<HaloDepth, Ls>(tg, op, x2, b, 1e-12, param.max_iter(), commBase);
        x2.axpy(MDWFDeflC(-1.0, 0.0), x1);
        const double u4 = std::sqrt(x2.norm2() / x1.norm2());
        auto verdict = [&](const char *name, double value, double tol) {
            const bool ok = value <= tol;
            pass = pass && ok;
            rootLogger.info("MDWF two-grid selftest ", name, ": ", value, " (tolerance ", tol, ") ", ok ? "PASS" : "FAIL");
        };
        rootLogger.info("MDWF two-grid selftest: coarse dimension ", n, ", orthonormality defect after pass 1 ",
                        st.orthoDefect, ", Hermiticity defect ", st.hermiticity);
        verdict("U1 |R P c - c| / |c|_max", u1, 1e-12);
        verdict("U2 |A_c c - R A P c| / max", u2, 1e-11);
        verdict("U3 |P A_c^-1 R A P c - P c| / |P c|", u3, 1e-10);
        verdict("U4 |x_twogrid - x_cg| / |x_cg|", u4, 1e-9);
        rootLogger.info("MDWF two-grid selftest U4 cost: plain CG ", r1.applications, " applications (", r1.iterations,
                        " it, res ", r1.trueResidual, "), two-grid ", r2.applications, " applications (", r2.iterations,
                        " outer it, d = ", degrees[0], ", res ", r2.trueResidual, ")");
        rootLogger.info("MDWF two-grid selftest: ", pass ? "ALL PASS" : "FAILED");
        if (!pass) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid selftest failed"));
        }
        return;
    }

    if (mode != "study") {
        throw std::runtime_error(stdLogger.fatal("mdwfTwoGridStudy: mode must be selftest, study or convert"));
    }

    // ---------------------------------------------------------------- right-hand sides (per eval)
    const std::string rhsKind = param.rhs();
    std::vector<int> evals = mdwfTGList(param.solve_evals, {setupEval});
    if (rhsKind != "dump") {
        evals = {setupEval};
    }
    auto makeRhs = [&](int eval, std::vector<std::unique_ptr<Vec>> &out, std::vector<std::string> &names) {
        out.clear();
        names.clear();
        if (rhsKind == "noise") {
            for (int k = 0; k < param.n_noise(); k++) {
                out.emplace_back(new Vec(commBase));
                out.back()->noise(seed + 1000003ULL * (k + 1));
                names.push_back("noise" + std::to_string(k));
            }
        } else if (rhsKind == "dump") {
            typename Op::Field field(commBase, "MDWF_tg_rhs_field");
            mdwfReadEvenSpinor<HaloDepth, Ls>(commBase, field,
                                              mdwfDeflationDumpTerm(param.dump_dir(), eval, param.dump_term(), "rhs"),
                                              "MDWF_tg_stage_s");
            out.emplace_back(new Vec(commBase));
            out.back()->pack(field);
            names.push_back("dump_t" + std::to_string(param.dump_term()));
        } else if (rhsKind == "point") {
            if (mmdag) {
                throw std::runtime_error(stdLogger.fatal("rhs = point needs normal_form = 0 (the measurement solve)"));
            }
            const std::vector<int> x0 = mdwfTGInts(param.point_source());
            const std::vector<int> comps = mdwfTGList(param.point_components, {0});
            using Meas = MDWFResidualMassMeasurement<HaloDepth, Ls, MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>>;
            Meas meas(fermionGauge, param.M5(), param.mass(), param.b5(), 0.0, "MDWF_tg_meas");
            typename EvenOdd::SpinorAll imported(commBase, "MDWF_tg_imp");
            typename EvenOdd::SpinorE be(commBase, "MDWF_tg_be"), te(commBase, "MDWF_tg_te"), ue(commBase, "MDWF_tg_ue");
            typename EvenOdd::SpinorO bo(commBase, "MDWF_tg_bo"), to(commBase, "MDWF_tg_to");
            EvenOdd &eo = op.evenOdd();
            for (const int a : comps) {
                meas.source5D().template iterateOverBulk<BLOCKSIZE>(
                    MDWFPhysicalPointSource<HaloDepth, Ls>(x0[0], x0[1], x0[2], x0[3], a));
                mdwfUpdateHalo(meas.source5D());
                meas.importSource(imported, meas.source5D());
                // the even/odd preparation of MDWFMobiusEvenOddMixedSolver::solveNoRefresh
                EvenOdd::split(be, bo, imported);
                eo.MooInv(to, bo);
                eo.Meo(te, to);
                ue = be;
                ue.template axpyThisB<64>(-1.0, te);
                eo.schur(te, ue, true);
                out.emplace_back(new Vec(commBase));
                out.back()->pack(te);
                names.push_back("point_c" + std::to_string(a));
            }
        } else {
            throw std::runtime_error(stdLogger.fatal("rhs must be point, dump or noise"));
        }
    };

    // ---------------------------------------------------------------- near-null vectors (raw, host copies)
    const std::vector<int> block = mdwfTGInts(param.block());
    const std::vector<int> nsList = mdwfTGList(param.ns_list, {16});
    const std::vector<int> degrees = mdwfTGList(param.degrees, {4});
    std::vector<double> ratios{param.smoother_ratio()};
    if (param.smoother_ratios.isSet() && param.smoother_ratios.numberValues() > 0) {
        ratios = param.smoother_ratios.get();
    }
    const int nsMax = *std::max_element(nsList.begin(), nsList.end());
    const std::string setup = param.setup();
    const double setupLo = param.setup_lo() > 0.0 ? param.setup_lo() : hi / param.setup_ratio();
    std::vector<std::vector<double>> rawHost(nsMax);
    std::vector<double> rawSeconds(nsMax, 0.0);
    std::vector<long> rawApps(nsMax, 0);
    {
        Vec v(commBase), keep(commBase);
        for (int k = 0; k < nsMax; k++) {
            const long a0 = op.applications();
            const auto t0 = std::chrono::steady_clock::now();
            if (setup == "exact") {
                typename Op::Field field(commBase, "MDWF_tg_exact_field");
                mdwfReadEvenSpinor<HaloDepth, Ls>(commBase, field, param.exact_dir() + "/v" + std::to_string(k),
                                                  "MDWF_tg_stage_e");
                keep.pack(field);
            } else {
                v.noise(seed + 7919ULL * (k + 1));
                const int nIt = param.setup_n();
                auto grab = [&](int nn, const Vec &w, double) { if (nn == nIt) { keep = w; } };
                if (setup == "cheb") {
                    mdwfDeflChebyshevSnapshots<Op, HaloDepth, Ls>(op, v, setupLo, hi, {nIt}, grab, commBase);
                } else if (setup == "inviter") {
                    mdwfDeflCGSnapshots<Op, HaloDepth, Ls>(op, v, {nIt}, grab, commBase);
                } else {
                    throw std::runtime_error(stdLogger.fatal("setup must be cheb, inviter or exact"));
                }
            }
            keep *= COMPLEX(double)(1.0 / std::sqrt(keep.norm2()), 0.0);
            keep.toHost(rawHost[k]);
            rawApps[k] = op.applications() - a0;
            rawSeconds[k] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        long total = 0;
        for (long a : rawApps) {
            total += a;
        }
        rootLogger.info("MDWF two-grid study: ", nsMax, " near-null vectors (", setup, ", n = ", param.setup_n(),
                        setup == "cheb" ? ", filter [" + std::to_string(setupLo) + ", " + std::to_string(hi) + "]" : "",
                        "), ", total, " applications of A");
    }

    // ---------------------------------------------------------------- output
    const std::string outFile = param.out_dir() + "/twogrid_" + param.tag() + ".tsv";
    std::ofstream out;
    if (commBase.IamRoot()) {
        const bool fresh = !std::ifstream(outFile).good();
        out.open(outFile, std::ios::out | std::ios::app);
        out << std::setprecision(10);
        if (fresh) {
            out << "tag\tsetup_eval\teval\trhs\tvariant\tsetup\tsetup_n\tblock\tns\tcoarse_dim\tdegree\tsmoother_lo\t"
                   "smoother_hi\tsetup_apps\tprobe_apps\tsetup_s\tortho_s\tprobe_s\tfactor_s\thermiticity\touter_it\t"
                   "apps\tsolve_s\ttrue_res\tconverged\tplain_apps\tplain_s\tplain_res\tgain_apps\tdiff_vs_plain"
                << std::endl;
        }
    }

    // plain-CG baselines per (eval, rhs index), computed once
    struct Plain { MDWFTwoGridSolveResult r; std::vector<double> x; };
    std::map<std::pair<int, int>, Plain> plain;

    for (const int ns : nsList) {
        TG tg(commBase, block);
        if (2 * tg.nBlocks() * ns > param.max_coarse_dim()) {
            rootLogger.info("MDWF two-grid study: skip ns = ", ns, " (coarse dimension ", 2 * tg.nBlocks() * ns,
                            " > max_coarse_dim)");
            continue;
        }
        if (setupEval >= 0) {
            loadGauge(setupEval);
            op.refresh();
        }
        Set raw;
        mdwfDeflAllocate(raw, ns, commBase);
        long setupApps = 0;
        double setupSeconds = 0.0;
        for (int k = 0; k < ns; k++) {
            raw[k]->fromHost(rawHost[k]);
            setupApps += rawApps[k];
            setupSeconds += rawSeconds[k];
        }
        MDWFTwoGridStats st;
        tg.setBasis(raw, ns, st);
        std::vector<C> acSetup;
        const bool laterEvals = evals.size() > 1 || (evals.size() == 1 && evals[0] != setupEval);
        tg.buildCoarse(op, st, laterEvals ? &acSetup : nullptr);
        const MDWFTwoGridStats setupStats = st;
        bool coarseIsSetup = true;

        for (size_t e = 0; e < evals.size(); e++) {
            const int eval = evals[e];
            if (eval != setupEval) {
                loadGauge(eval);
                op.refresh();
            }
            std::vector<std::unique_ptr<Vec>> rhs;
            std::vector<std::string> names;
            makeRhs(eval, rhs, names);
            const std::vector<std::string> variants = (eval == setupEval) ? std::vector<std::string>{"setup"}
                                                                          : std::vector<std::string>{"stale", "fresh"};
            for (const std::string &variant : variants) {
                MDWFTwoGridStats vs = setupStats;
                if (variant == "fresh") {
                    vs.probeApplications = 0;
                    vs.probeSeconds = 0.0;
                    vs.factorSeconds = 0.0;
                    tg.buildCoarse(op, vs);
                    coarseIsSetup = false;
                } else if (!coarseIsSetup) {   // stale: the A_c of the setup eval again
                    tg.setCoarse(acSetup);
                    coarseIsSetup = true;
                }
                for (const double ratio : ratios) {
                for (const int d : degrees) {
                    const double lo = param.smoother_lo() > 0.0 ? param.smoother_lo() : hi / ratio;
                    tg.setSmoother(lo, hi, d);
                    for (size_t i = 0; i < rhs.size(); i++) {
                        const auto key = std::make_pair(eval, static_cast<int>(i));
                        if (static_cast<int>(i) < param.plain_cg() && plain.find(key) == plain.end()) {
                            Vec x(commBase);
                            Plain p;
                            p.r = mdwfTwoGridPlainCG(op, x, *rhs[i], param.precision(), param.max_iter(), commBase);
                            x.toHost(p.x);
                            rootLogger.info("MDWF two-grid study: plain CG eval ", eval, " ", names[i], ": ",
                                            p.r.applications, " applications, ", p.r.seconds, " s, residual ",
                                            p.r.trueResidual);
                            plain[key] = std::move(p);
                        }
                        Vec x(commBase);
                        const MDWFTwoGridSolveResult r = mdwfTwoGridFCG<HaloDepth, Ls>(
                            tg, op, x, *rhs[i], param.precision(), param.max_iter(), commBase);
                        double diff = -1.0;
                        long plainApps = -1;
                        double plainSeconds = -1.0, plainRes = -1.0;
                        auto it = plain.find(key);
                        if (it != plain.end()) {
                            Vec y(commBase);
                            y.fromHost(it->second.x);
                            const double ny = y.norm2();
                            y.axpy(MDWFDeflC(-1.0, 0.0), x);
                            diff = std::sqrt(y.norm2() / ny);
                            plainApps = it->second.r.applications;
                            plainSeconds = it->second.r.seconds;
                            plainRes = it->second.r.trueResidual;
                        }
                        rootLogger.info("MDWF two-grid study: eval ", eval, " ", names[i], " ", variant, " ns ", ns,
                                        " d ", d, " lo ", lo, ": ", r.iterations, " outer it, ", r.applications,
                                        " applications, ", r.seconds, " s, residual ", r.trueResidual,
                                        plainApps > 0 ? ", gain " + std::to_string(double(plainApps) / r.applications)
                                                      : std::string(""));
                        if (commBase.IamRoot()) {
                            out << param.tag() << "\t" << setupEval << "\t" << eval << "\t" << names[i] << "\t"
                                << variant << "\t" << setup << "\t" << param.setup_n() << "\t" << param.block() << "\t"
                                << ns << "\t" << tg.coarseDim() << "\t" << d << "\t" << lo << "\t" << hi << "\t"
                                << setupApps << "\t" << vs.probeApplications << "\t" << setupSeconds << "\t"
                                << vs.orthoSeconds << "\t" << vs.probeSeconds << "\t" << vs.factorSeconds << "\t"
                                << vs.hermiticity << "\t" << r.iterations << "\t" << r.applications << "\t"
                                << r.seconds << "\t" << r.trueResidual << "\t" << (r.converged ? 1 : 0) << "\t"
                                << plainApps << "\t" << plainSeconds << "\t" << plainRes << "\t"
                                << (plainApps > 0 ? double(plainApps) / r.applications : -1.0) << "\t" << diff
                                << std::endl;
                        }
                    }
                }
                }
            }
        }
    }
    rootLogger.info("MDWF two-grid study: done, rows in ", outFile);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFTwoGridStudyParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfTwoGridStudy.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        if (param.ls() == 8) {
            runMDWFTwoGridStudy<8>(commBase, param);
        } else if (param.ls() == 16) {
            runMDWFTwoGridStudy<16>(commBase, param);
        } else if (param.ls() == 24) {
            runMDWFTwoGridStudy<24>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfTwoGridStudy supports ls = 8, 16 or 24, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
