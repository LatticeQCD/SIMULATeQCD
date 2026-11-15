/*
 * Step 0 of the deflation study for the light MDWF force solves (DSDRMDWFruns/SQCDruns/deflation/DESIGN.md): analysis
 * of the MD positions dumped by mdwfTwoPlusOneHmcRun with defl_dump = 1 (MDWFDeflationDump.h). Tools: MDWFDeflation.h.
 *
 * The operator of force term t is A = Mhat(m_t) Mhat(m_t)^+ on the even sites (light_form = 1; Mhat^+ Mhat for
 * light_form = 0), m_0 = mf, m_t = hasenbusch_masses[t - 1], with the fermion boundary conditions of the run.
 *
 *   mode = eig     for every evaluation in `evals` and every term in `terms`: the k_max lowest eigenpairs of A
 *                  (Chebyshev-filtered thick-restart Lanczos + Rayleigh-Ritz), written to <out_dir>/vec_eNNN_t<t>.r<rank>
 *                  and <out_dir>/eig_eNNN_t<t>.tsv; also checks the dumped force solution, |A sol - rhs| / |rhs|.
 *                  An evaluation whose .tsv exists is skipped (restart).
 *   mode = angles  cosines of the principal angles between the exact K-dimensional subspaces of evaluations
 *                  evals[i] and evals[i + n], n in `lags`, and of the extra `pairs` (a b a b ...), K in k_list:
 *                  <out_dir>/angles_t<t>.tsv
 *   mode = track   chain evals[0] -> evals[1] -> ...; at lag n (position in the chain) the subspaces
 *                    T0  stale V_0 (the exact vectors of evals[0])
 *                    T1  the same span, Rayleigh-Ritz in A_n (K applications)
 *                    T2  lowest K Ritz vectors of span{V, (A_m - A_{m-1}) V}, step by step m = 1 ... n (3K applications
 *                        per step)
 *                    T3  lowest K Ritz vectors of span{V, A_m V}, step by step (2K applications per step)
 *                  are compared with the exact V_n: cosines of the principal angles (..._sigma.tsv), Ritz values and
 *                  residuals (..._ritz.tsv), and, for n in cg_lags, CG iterations to cg_precision (..._cg.tsv) for
 *                    none        plain CG
 *                    exact       deflated CG with the exact V_n
 *                    T0          initial guess sum_k v_k (v_k^+ b) / lambda_k with the stale pairs, then plain CG
 *                                (free: no application of A)
 *                    T1, T2, T3  deflated CG with the Galerkin matrix in A_n. T0 and T1 span the same space, so T1 is
 *                                what the stale space gives when it is used correctly.
 *                  Deflated CG = Saad et al. (search directions A-orthogonal to the subspace in every iteration);
 *                  cg_init = 1 adds the cheap variant "deflated initial guess, then plain CG" for exact, T1, T2, T3.
 *                  Right-hand sides: hash noise (rhs_random) and the dumped one of the force solve (rhs_actual).
 *                  Files <out_dir>/track_t<t>_c<evals[0]>_{sigma,ritz,cg}.tsv, rows appended as they are measured.
 *   mode = calib   timing on one NERSC configuration (Gaugefile): eigenpairs and none / exact CG with a noise
 *                  right-hand side, <out_dir>/calib_t<t>.tsv. Needs no dump.
 *   mode = refresh (2026-11-13) the cost of refreshing a stale set: the first refresh_ns exact vectors v_j of evaluation
 *                  refresh_source (read from eig_dir, default out_dir) are improved with the operator A of every
 *                  evaluation e1 in `evals` (gauge field from the dump), each vector on its own, by
 *                    cg    x = n iterations of plain CG on A x = v_j from x = 0 (no convergence stop: inverse
 *                          iteration with an inexact solve), n in refresh_n_cg
 *                    cheb  x = T_n(L(A)) v_j, Chebyshev low-pass filter on [cheb_lo, cheb_hi] (the recurrence of the
 *                          eigensolver's filter), n in refresh_n_cheb; cheb_lo <= 0: the largest eigenvalue stored in
 *                          the source set file (lambda_63 for a K = 64 set); cheb_hi <= 0: 1.15 x a 60-step Lanczos
 *                          estimate of lambda_max(A)
 *                  (refresh_methods: cg, cheb or both). Both recurrences are nested in n, so one run per vector
 *                  gives every n of the list (MDWFDeflation.h, mdwfDeflCGSnapshots / mdwfDeflChebyshevSnapshots).
 *                  Per snapshot n > 0 the normalised vectors are written as a set (same format as eig; eigenvalue
 *                  slot j = Rayleigh quotient of vector j under A(e1)):
 *                    <out_dir>/refresh/vec_e<e0>_t<t>_to_e<e1>_<method><n>.r<rank>   (refresh_write = 0: none)
 *                  and one row per (n, j), n = 0 (the stale vector itself) included, to
 *                    <out_dir>/refresh/refresh_e<e0>_t<t>_to_e<e1>_<method>.tsv
 *                  (method, n, applications of A = n, seconds, j, Rayleigh quotient, |A x - rq x| / |x|, stale
 *                  eigenvalue). The .tsv is renamed from .part after the sets: its existence marks the pair done
 *                  (skipped on a rerun). <out_dir>/refresh must exist. GPU memory: (number of n > 0 + 1) x
 *                  refresh_ns compact vectors (151 MB each per rank at 32^3 x 8, Ls 24, 4 ranks).
 *
 * Usage: ./mdwfDeflationStep0 <param file> [key=value ...]   (parameter/tests/mdwfDeflationStep0.param). List
 * parameters given on the command line are appended to those of the file. The node layout must be the same in all
 * modes (the eigenvector files are per rank); the dump itself may come from any layout.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDeflation.h"
#include "../experimental/mdwf/MDWFDeflationDump.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFDeflationStep0Parameters : public LatticeParameters {
public:
    Parameter<std::string> mode;
    Parameter<int> ls;
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> mf;
    DynamicParameter<double> hasenbusch_masses;
    Parameter<int> antiperiodic_t;
    Parameter<int> light_form;
    Parameter<int> force_halos;
    Parameter<std::string> dump_dir;
    Parameter<std::string> out_dir;
    DynamicParameter<int> terms;
    DynamicParameter<int> evals;
    DynamicParameter<int> pairs;
    DynamicParameter<int> lags;
    DynamicParameter<int> cg_lags;
    DynamicParameter<int> k_list;
    Parameter<int> k_max;
    Parameter<double> eig_tol;
    Parameter<double> cheb_strength;
    Parameter<double> cheb_lo;
    Parameter<int> eig_restarts;
    Parameter<int> nbasis;
    Parameter<int> nkeep;
    Parameter<double> cg_precision;
    Parameter<int> cg_max_iter;
    Parameter<int> cg_init;
    Parameter<int> rhs_random;
    Parameter<int> rhs_actual;
    Parameter<int> seed;
    // mode = refresh (all unused by the other modes)
    Parameter<int> refresh_source;
    Parameter<int> refresh_ns;
    DynamicParameter<std::string> refresh_methods;
    DynamicParameter<int> refresh_n_cg;
    DynamicParameter<int> refresh_n_cheb;
    Parameter<int> refresh_write;
    Parameter<double> cheb_hi;
    Parameter<std::string> eig_dir;

    MDWFDeflationStep0Parameters() {
        add(mode, "mode");
        addDefault(ls, "Ls", 24);
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 2.5);
        add(mf, "mf");
        addOptional(hasenbusch_masses, "hasenbusch_masses");
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(light_form, "light_form", 1);
        addDefault(force_halos, "force_halos", 0);
        addDefault(dump_dir, "dump_dir", std::string("."));
        addDefault(out_dir, "out_dir", std::string("."));
        addOptional(terms, "terms");
        addOptional(evals, "evals");
        addOptional(pairs, "pairs");
        addOptional(lags, "lags");
        addOptional(cg_lags, "cg_lags");
        addOptional(k_list, "k_list");
        addDefault(k_max, "k_max", 64);
        addDefault(eig_tol, "eig_tol", 1e-11);
        addDefault(cheb_strength, "cheb_strength", 4.0);
        addDefault(cheb_lo, "cheb_lo", 0.0);
        addDefault(eig_restarts, "eig_restarts", 60);
        addDefault(nbasis, "nbasis", 0);
        addDefault(nkeep, "nkeep", 0);
        addDefault(cg_precision, "cg_precision", 1e-8);
        addDefault(cg_max_iter, "cg_max_iter", 200000);
        addDefault(cg_init, "cg_init", 0);
        addDefault(rhs_random, "rhs_random", 1);
        addDefault(rhs_actual, "rhs_actual", 1);
        addDefault(seed, "seed", 20261110);
        addDefault(refresh_source, "refresh_source", -1);
        addDefault(refresh_ns, "refresh_ns", 32);
        addOptional(refresh_methods, "refresh_methods");
        addOptional(refresh_n_cg, "refresh_n_cg");
        addOptional(refresh_n_cheb, "refresh_n_cheb");
        addDefault(refresh_write, "refresh_write", 1);
        addDefault(cheb_hi, "cheb_hi", 0.0);
        addOptional(eig_dir, "eig_dir");
    }
};

template<class P>
std::vector<int> mdwfStep0List(P &parameter, const std::vector<int> &fallback) {
    if (parameter.isSet() && parameter.numberValues() > 0) {
        return parameter.get();
    }
    return fallback;
}

inline std::string mdwfStep0Name(const std::string &dir, const char *what, int eval, int term) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%s_e%03d_t%d", what, eval, term);
    return dir + "/" + buffer;
}

inline bool mdwfStep0Exists(const std::string &path) {
    return std::ifstream(path).good();
}

// Appends rows to a TSV on the root rank; the header is written when the file is new.
class MDWFStep0Table {
    std::ofstream _out;
    bool _root;

public:
    MDWFStep0Table(CommunicationBase &comm, const std::string &path, const std::string &header) : _root(comm.IamRoot()) {
        if (_root) {
            const bool fresh = !mdwfStep0Exists(path);
            _out.open(path, std::ios::out | std::ios::app);
            if (!_out) {
                throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: cannot open ", path));
            }
            _out << std::setprecision(17);
            if (fresh) {
                _out << header << std::endl;
            }
        }
    }

    template<class... Args>
    void row(const Args &...args) {
        if (_root) {
            const char *sep = "";
            ((_out << sep << args, sep = "\t"), ...);
            _out << std::endl;
        }
    }
};

// Last column block of an eig_*.tsv: the eigenvalues (column 2 of the data rows). Empty if the file is missing.
inline std::vector<double> mdwfStep0ReadLambda(const std::string &path) {
    std::vector<double> lambda;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream row(line);
        int k;
        double l;
        if (row >> k >> l) {
            lambda.push_back(l);
        }
    }
    return lambda;
}

template<size_t Ls>
void runMDWFDeflationStep0(CommunicationBase &commBase, MDWFDeflationStep0Parameters &param) {
    constexpr size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using Field = typename Op::Field;
    using Subspace = MDWFDeflSubspace<HaloDepth, Ls>;

    const std::string mode = param.mode();
    const std::string dumpDir = param.dump_dir(), outDir = param.out_dir();
    std::vector<double> ladder{param.mf()};
    if (param.hasenbusch_masses.isSet()) {
        const std::vector<double> h = param.hasenbusch_masses.get();
        ladder.insert(ladder.end(), h.begin(), h.end());
    }
    const std::vector<int> terms = mdwfStep0List(param.terms, {0, 1});
    for (const int t : terms) {
        if (t < 0 || t >= static_cast<int>(ladder.size())) {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: term ", t, " needs ", t,
                                                     " hasenbusch_masses"));
        }
    }
    const std::vector<int> evals = mdwfStep0List(param.evals, {});
    const std::vector<int> lags = mdwfStep0List(param.lags, {1, 2, 4, 8});
    const std::vector<int> cgLags = mdwfStep0List(param.cg_lags, lags);
    const std::vector<int> kList = mdwfStep0List(param.k_list, {16, 32, 64});
    const std::vector<int> pairs = mdwfStep0List(param.pairs, {});
    const int kMax = param.k_max();
    const bool mmdag = param.light_form() == 1;
    const bool antiperiodic = param.antiperiodic_t() != 0;
    const double precision = param.cg_precision();
    const int maxIter = param.cg_max_iter();
    const uint64_t seed = static_cast<uint64_t>(param.seed());
    for (const int k : kList) {
        if (k < 1 || (mode != "calib" && k > kMax)) {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: k_list entry ", k, " outside 1 ... k_max"));
        }
    }
    rootLogger.info("MDWF deflation step 0: mode ", mode, ", Ls ", Ls, ", M5 ", param.M5(), ", b5 ", param.b5(),
                    ", operator ", mmdag ? "Mhat Mhat^+" : "Mhat^+ Mhat", ", antiperiodic_t ", antiperiodic,
                    ", dump ", dumpDir, ", output ", outDir);

    Gauge thin(commBase, "MDWF_step0_thin");
    Gauge fermionGaugeA(commBase, "MDWF_step0_fga"), fermionGaugeB(commBase, "MDWF_step0_fgb");
    Gauge *fermionGauge[2] = {&fermionGaugeA, &fermionGaugeB};

    auto loadGauge = [&](int eval, int slot) {
        mdwfReadGauge<HaloDepth>(commBase, thin, mdwfDeflationDumpEval(dumpDir, eval) + "_U", "MDWF_step0_stage_u");
        thin.updateAll();
        mdwfFermionGaugeField<HaloDepth>(*fermionGauge[slot], thin, antiperiodic);
    };
    auto readField = [&](Vec &v, const std::string &base) {
        Field field(commBase, Op::EvenOdd::sharedEvenName("defl_io"));
        mdwfReadEvenSpinor<HaloDepth, Ls>(commBase, field, base, "MDWF_step0_stage_s");
        v.pack(field);
    };
    auto makeOperator = [&](int slot, int term, const char *tag) {
        return std::unique_ptr<Op>(new Op(*fermionGauge[slot], param.M5(), ladder[term], param.b5(), mmdag,
                                          std::string("MDWF_step0_op") + tag + std::to_string(term)));
    };
    auto eigenParameters = [&](int nev, double lo) {
        MDWFDeflEigenParameters p;
        p.nev = nev;
        p.nBasis = param.nbasis();
        p.nKeep = param.nkeep();
        p.tol = param.eig_tol();
        p.maxRestarts = param.eig_restarts();
        p.strength = param.cheb_strength();
        p.lo = lo;
        return p;
    };
    // |A sol - rhs| / |rhs| of the dumped force solve; < 0 if the dump has no solution.
    auto dumpedResidual = [&](Op &op, int eval, int term) {
        const std::string rhsBase = mdwfDeflationDumpTerm(dumpDir, eval, term, "rhs");
        const std::string solBase = mdwfDeflationDumpTerm(dumpDir, eval, term, "sol");
        if (!mdwfStep0Exists(mdwfFieldDumpRankFile(rhsBase, 0)) || !mdwfStep0Exists(mdwfFieldDumpRankFile(solBase, 0))) {
            return -1.0;
        }
        Vec rhs(commBase), sol(commBase), asol(commBase);
        readField(rhs, rhsBase);
        readField(sol, solBase);
        op.apply(asol, sol);
        asol.axpyThisB(-1.0, rhs);
        return std::sqrt(asol.norm2() / rhs.norm2());
    };
    auto writeEigen = [&](const std::string &path, const MDWFDeflEigenResult &r, int eval, int term,
                          double solveResidual) {
        if (!commBase.IamRoot()) {
            return;
        }
        std::ofstream out(path + ".part", std::ios::out | std::ios::trunc);
        out << std::setprecision(17);
        out << "# mdwfDeflationStep0 eig: evaluation " << eval << ", term " << term << ", mass " << ladder[term]
            << ", operator " << (mmdag ? "MMdag" : "MdagM") << "\n# converged " << r.converged << ", attempts "
            << r.attempts << ", Chebyshev [" << r.lo << ", " << r.hi << "] degree " << r.degree << ", applications "
            << r.applications << ", seconds " << r.seconds << "\n# dumped force solve |A sol - rhs| / |rhs| = "
            << solveResidual << " (-1: not dumped)\n# k\tlambda\tresidual\n";
        for (size_t k = 0; k < r.lambda.size(); k++) {
            out << k << "\t" << r.lambda[k] << "\t" << r.residual[k] << "\n";
        }
        out.close();
        if (!out || std::rename((path + ".part").c_str(), path.c_str()) != 0) {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: cannot write ", path));
        }
    };

    // ------------------------------------------------------------------------------------------------------ calib
    if (mode == "calib") {
        if (!param.GaugefileName.isSet()) {
            throw std::runtime_error(stdLogger.fatal("mode = calib needs Gaugefile"));
        }
        thin.readconf_nersc(param.GaugefileName());
        thin.updateAll();
        mdwfFermionGaugeField<HaloDepth>(*fermionGauge[0], thin, antiperiodic);
        for (const int term : terms) {
            std::unique_ptr<Op> op = makeOperator(0, term, "c");
            op->refresh();
            Vec b(commBase), x(commBase), y(commBase);
            b.noise(seed + 7919ULL * term);
            x.zero();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; i++) {
                op->apply(y, b);
            }
            const double perApplication = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                                          / 20.0;
            const auto t1 = std::chrono::steady_clock::now();
            double sink = 0.0;
            for (int i = 0; i < 20; i++) {
                sink += real<double>(y.dot(b));
                x.axpyThisB(0.0, y);
            }
            const double perDotAxpy = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count()
                                      / 20.0;
            rootLogger.info("MDWF deflation calib: term ", term, ", mass ", ladder[term], ": ", perApplication,
                            " s per application of A, ", perDotAxpy, " s per dot + axpy (", sink != sink, ")");
            MDWFStep0Table table(commBase, outDir + "/calib_t" + std::to_string(term) + ".tsv",
                                 "# what\tK\tvalue\titerations\ttrue_residual\tseconds");
            table.row("seconds_per_application", 0, perApplication, 0, 0, 0);
            table.row("seconds_per_dot_axpy", 0, perDotAxpy, 0, 0, 0);
            const MDWFDeflCGResult none = mdwfDeflCG<Op, HaloDepth, Ls>(*op, x, b, nullptr, MDWFDeflNone, precision,
                                                                       maxIter, commBase);
            rootLogger.info("MDWF deflation calib: term ", term, " CG none: ", none.iterations, " iterations, true "
                            "residual ", none.trueResidual, ", ", none.seconds, " s");
            table.row("cg_none", 0, 0, none.iterations, none.trueResidual, none.seconds);
            Set v;
            const MDWFDeflEigenResult eig = mdwfDeflLowModes(*op, v, eigenParameters(kMax, param.cheb_lo()),
                                                             seed + 31ULL * term, commBase);
            writeEigen(outDir + "/calib_eig_t" + std::to_string(term) + ".tsv", eig, -1, term, -1.0);
            double worst = 0.0;
            for (size_t k = 0; k < eig.lambda.size(); k++) {
                worst = std::max(worst, eig.residual[k] / eig.lambda[k]);
            }
            rootLogger.info("MDWF deflation calib: term ", term, " eigensolver K = ", kMax, ": lambda ", eig.lambda.front(),
                            " ... ", eig.lambda.back(), ", lambda_max bound ", eig.hi, ", worst residual / lambda ",
                            worst, ", ", eig.applications, " applications, ", eig.seconds, " s, converged ",
                            eig.converged);
            table.row("eig_seconds", kMax, eig.seconds, eig.applications, worst, eig.seconds);
            table.row("lambda_first", kMax, eig.lambda.front(), 0, 0, 0);
            table.row("lambda_last", kMax, eig.lambda.back(), 0, 0, 0);
            table.row("lambda_max_bound", kMax, eig.hi, 0, 0, 0);
            Subspace sub;
            for (const int k : kList) {
                if (k > kMax) {
                    continue;
                }
                sub.setup(*op, mdwfDeflPointers(v, k), commBase);
                const MDWFDeflCGResult proj = mdwfDeflCG(*op, x, b, &sub, MDWFDeflProject, precision, maxIter, commBase);
                const MDWFDeflCGResult init = mdwfDeflCG(*op, x, b, &sub, MDWFDeflInit, precision, maxIter, commBase);
                rootLogger.info("MDWF deflation calib: term ", term, " K = ", k, ": deflated CG ", proj.iterations,
                                " iterations (", proj.seconds, " s, true residual ", proj.trueResidual,
                                "), deflated initial guess + CG ", init.iterations, " iterations (", init.seconds,
                                " s, true residual ", init.trueResidual, ")");
                table.row("cg_exact_project", k, 0, proj.iterations, proj.trueResidual, proj.seconds);
                table.row("cg_exact_init", k, 0, init.iterations, init.trueResidual, init.seconds);
            }
        }
        return;
    }

    if (evals.empty()) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: mode ", mode, " needs evals"));
    }

    // -------------------------------------------------------------------------------------------------------- eig
    if (mode == "eig") {
        for (const int term : terms) {
            std::unique_ptr<Op> op = makeOperator(0, term, "e");
            double hint = param.cheb_lo();
            for (const int eval : evals) {
                const std::string tsv = mdwfStep0Name(outDir, "eig", eval, term) + ".tsv";
                if (mdwfStep0Exists(tsv)) {
                    const std::vector<double> done = mdwfStep0ReadLambda(tsv);
                    if (static_cast<int>(done.size()) >= kMax) {
                        hint = 2.5 * done[kMax - 1];
                        rootLogger.info("MDWF deflation eig: evaluation ", eval, " term ", term, " exists, skipped");
                        continue;
                    }
                }
                if (!(hint > 0.0)) {
                    // Filter edge from any finished evaluation of this term (the spectrum moves slowly).
                    for (int other = 0; other < 1000 && !(hint > 0.0); other++) {
                        const std::vector<double> l = mdwfStep0ReadLambda(mdwfStep0Name(outDir, "eig", other, term) + ".tsv");
                        if (static_cast<int>(l.size()) >= kMax) {
                            hint = 2.5 * l[kMax - 1];
                        }
                    }
                }
                loadGauge(eval, 0);
                op->refresh();
                const double solveResidual = dumpedResidual(*op, eval, term);
                rootLogger.info("MDWF deflation eig: evaluation ", eval, " term ", term, " (mass ", ladder[term],
                                "): dumped force solve |A sol - rhs| / |rhs| = ", solveResidual);
                Set v;
                const MDWFDeflEigenResult eig = mdwfDeflLowModes(*op, v, eigenParameters(kMax, hint),
                                                                 seed + 1000003ULL * eval + 31ULL * term, commBase);
                double worst = 0.0;
                for (size_t k = 0; k < eig.lambda.size(); k++) {
                    worst = std::max(worst, eig.residual[k] / eig.lambda[k]);
                }
                rootLogger.info("MDWF deflation eig: evaluation ", eval, " term ", term, ": lambda ", eig.lambda.front(),
                                " ... ", eig.lambda.back(), ", worst residual / lambda ", worst, ", ", eig.applications,
                                " applications, ", eig.seconds, " s, converged ", eig.converged);
                mdwfDeflWriteSet<HaloDepth, Ls>(commBase, v, kMax, eig.lambda, mdwfStep0Name(outDir, "vec", eval, term));
                writeEigen(tsv, eig, eval, term, solveResidual);   // after the vectors: its existence marks "done"
                hint = 2.5 * eig.lambda.back();
            }
        }
        return;
    }

    // ----------------------------------------------------------------------------------------------------- angles
    if (mode == "angles") {
        for (const int term : terms) {
            MDWFStep0Table table(commBase, outDir + "/angles_t" + std::to_string(term) + ".tsv",
                                 "# term\tK\teval_i\teval_j\tlag\tk\tcosine   (lag -1: extra pair)");
            // (eval_i, eval_j, lag), grouped by eval_i so that its set is read once.
            std::map<int, std::vector<std::pair<int, int>>> partners;
            for (size_t i = 0; i < evals.size(); i++) {
                for (const int n : lags) {
                    if (i + n < evals.size()) {
                        partners[evals[i]].push_back({evals[i + n], n});
                    }
                }
            }
            for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
                partners[pairs[i]].push_back({pairs[i + 1], -1});
            }
            for (const auto &entry : partners) {
                Set a, b;
                std::vector<double> la, lb;
                mdwfDeflReadSet<HaloDepth, Ls>(commBase, a, kMax, la, mdwfStep0Name(outDir, "vec", entry.first, term));
                for (const auto &partner : entry.second) {
                    mdwfDeflReadSet<HaloDepth, Ls>(commBase, b, kMax, lb,
                                                  mdwfStep0Name(outDir, "vec", partner.first, term));
                    const std::vector<MDWFDeflC> g = mdwfDeflGram(mdwfDeflPointers(a, kMax), mdwfDeflPointers(b, kMax));
                    for (const int k : kList) {
                        std::vector<MDWFDeflC> block(static_cast<size_t>(k) * k);
                        for (int r = 0; r < k; r++) {
                            for (int c = 0; c < k; c++) {
                                block[r * k + c] = g[r * kMax + c];
                            }
                        }
                        const std::vector<double> sigma = mdwfSingularValues(block, k, k);
                        double capture = 0.0;
                        for (int j = 0; j < k; j++) {
                            capture += sigma[j] * sigma[j] / k;
                            table.row(term, k, entry.first, partner.first, partner.second, j, sigma[j]);
                        }
                        rootLogger.info("MDWF deflation angles: term ", term, " K ", k, " evaluations ", entry.first,
                                        " -> ", partner.first, " (lag ", partner.second, "): mean cos^2 = ", capture,
                                        ", smallest cosine ", sigma.back());
                    }
                }
            }
        }
        return;
    }

    // ---------------------------------------------------------------------------------------------------- refresh
    if (mode == "refresh") {
        const int source = param.refresh_source();
        const int ns = param.refresh_ns();
        const std::string eigDir = param.eig_dir.isSet() ? param.eig_dir() : outDir;
        const std::string refreshDir = outDir + "/refresh";
        std::vector<std::string> methods{"cg", "cheb"};
        if (param.refresh_methods.isSet() && param.refresh_methods.numberValues() > 0) {
            methods = param.refresh_methods.get();
        }
        std::map<std::string, std::vector<int>> nList;
        nList["cg"] = mdwfStep0List(param.refresh_n_cg, {0, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
        nList["cheb"] = mdwfStep0List(param.refresh_n_cheb, {0, 16, 32, 64, 128, 256, 512, 1024, 2048});
        if (source < 0 || ns < 1) {
            throw std::runtime_error(stdLogger.fatal("mode = refresh needs refresh_source >= 0 and refresh_ns >= 1"));
        }
        for (const std::string &method : methods) {
            if (nList.count(method) == 0) {
                throw std::runtime_error(stdLogger.fatal("refresh_methods: cg or cheb, got ", method));
            }
            mdwfDeflCheckSnapshotList(nList[method]);
        }
        for (const int term : terms) {
            const std::string sourceBase = mdwfStep0Name(eigDir, "vec", source, term);
            Set stale;
            std::vector<double> lambdaStale;
            mdwfDeflReadSet<HaloDepth, Ls>(commBase, stale, ns, lambdaStale, sourceBase);
            double lo = param.cheb_lo();
            if (!(lo > 0.0)) {
                const std::vector<double> all = mdwfDeflReadSetLambdas(sourceBase);
                lo = *std::max_element(all.begin(), all.end());
            }
            std::unique_ptr<Op> op = makeOperator(0, term, "r");
            for (const int target : evals) {
                bool loaded = false;
                double hi = param.cheb_hi();
                for (const std::string &method : methods) {
                    char tag[96];
                    std::snprintf(tag, sizeof(tag), "e%03d_t%d_to_e%03d_%s", source, term, target, method.c_str());
                    const std::string tsv = refreshDir + "/refresh_" + tag + ".tsv";
                    if (mdwfStep0Exists(tsv)) {
                        rootLogger.info("MDWF deflation refresh: ", tsv, " exists, skipped");
                        continue;
                    }
                    if (!loaded) {
                        loadGauge(target, 0);
                        op->refresh();
                        loaded = true;
                    }
                    const bool cheb = method == "cheb";
                    if (cheb && !(hi > 0.0)) {
                        Vec start(commBase);
                        start.noise(seed + 1000003ULL * target + 31ULL * term + 17ULL);
                        hi = 1.15 * mdwfDeflLambdaMax(*op, start, 60);
                    }
                    const std::vector<int> &list = nList[method];
                    std::vector<int> written;   // n > 0
                    for (const int n : list) {
                        if (n > 0) {
                            written.push_back(n);
                        }
                    }
                    const double gb = static_cast<double>((written.size() + 1) * ns) * stale[0]->bytes() / 1e9;
                    rootLogger.info("MDWF deflation refresh: term ", term, " evaluation ", source, " -> ", target, ", ",
                                    method, cheb ? " on [" : "", cheb ? std::to_string(lo) + ", " + std::to_string(hi) + "]"
                                    : std::string(""), ", N_s ", ns, ", ", list.size(), " snapshots up to n = ",
                                    list.back(), "; snapshot memory ", gb, " GB per rank");
                    std::vector<Set> snap(written.size());
                    for (Set &s : snap) {
                        mdwfDeflAllocate(s, ns, commBase);
                    }
                    std::vector<std::vector<double>> rq(written.size(), std::vector<double>(ns, 0.0));
                    std::vector<double> sumRq(list.size(), 0.0), sumRes(list.size(), 0.0), sumSec(list.size(), 0.0);
                    if (commBase.IamRoot()) {
                        std::remove((tsv + ".part").c_str());
                    }
                    commBase.globalBarrier();
                    {
                        MDWFStep0Table table(commBase, tsv + ".part",
                                             "# method\tn\tapplications\tseconds\tj\trayleigh\trel_residual\t"
                                             "lambda_source   (refresh of vector j of evaluation " + std::to_string(source)
                                             + " with A of evaluation " + std::to_string(target) + ", term "
                                             + std::to_string(term) + "; rel_residual = |A x - rq x| / (rq |x|); "
                                             "seconds without the measurement; cheb interval ["
                                             + (cheb ? std::to_string(lo) + ", " + std::to_string(hi) : std::string("-"))
                                             + "])");
                        Vec y(commBase), work(commBase);
                        for (int j = 0; j < ns; j++) {
                            const MDWFDeflSnapshotCallback<HaloDepth, Ls> record = [&](int n, const Vec &x, double sec) {
                                const size_t at = std::find(list.begin(), list.end(), n) - list.begin();
                                const size_t w = std::find(written.begin(), written.end(), n) - written.begin();
                                Vec &dst = n > 0 ? *snap[w][j] : y;
                                dst = x;
                                dst *= COMPLEX(double)(1.0 / std::sqrt(x.norm2()), 0.0);
                                op->apply(work, dst);
                                const double q = real<double>(dst.dot(work));
                                work.axpyThisB(-q, dst);
                                const double res = std::sqrt(work.norm2()) / q;
                                if (n > 0) {
                                    rq[w][j] = q;
                                }
                                sumRq[at] += q / ns;
                                sumRes[at] += res / ns;
                                sumSec[at] += sec;
                                table.row(method, n, n, sec, j, q, res, lambdaStale[j]);
                            };
                            if (cheb) {
                                mdwfDeflChebyshevSnapshots(*op, *stale[j], lo, hi, list, record, commBase);
                            } else {
                                mdwfDeflCGSnapshots(*op, *stale[j], list, record, commBase);
                            }
                        }
                    }
                    for (size_t i = 0; i < list.size(); i++) {
                        rootLogger.info("MDWF deflation refresh: ", method, " n = ", list[i], ": mean Rayleigh quotient ",
                                        sumRq[i], ", mean |A x - rq x| / rq ", sumRes[i], ", ", list[i] * ns,
                                        " applications, ", sumSec[i], " s for all ", ns, " vectors");
                    }
                    if (param.refresh_write() != 0) {
                        for (size_t w = 0; w < written.size(); w++) {
                            mdwfDeflWriteSet<HaloDepth, Ls>(commBase, snap[w], ns, rq[w],
                                                          refreshDir + "/vec_" + std::string(tag) + std::to_string(written[w]));
                        }
                    }
                    commBase.globalBarrier();
                    if (commBase.IamRoot() && std::rename((tsv + ".part").c_str(), tsv.c_str()) != 0) {
                        throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: cannot rename ", tsv, ".part"));
                    }
                    commBase.globalBarrier();
                    rootLogger.info("MDWF deflation refresh: wrote ", tsv);
                }
            }
        }
        return;
    }

    // ------------------------------------------------------------------------------------------------------ track
    if (mode != "track") {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0: mode must be eig, angles, track, calib or "
                                                 "refresh, got ", mode));
    }
    const int chainLength = static_cast<int>(evals.size()) - 1;
    auto contains = [](const std::vector<int> &list, int value) {
        return std::find(list.begin(), list.end(), value) != list.end();
    };
    for (const int term : terms) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "/track_t%d_c%03d", term, evals[0]);
        const std::string base = outDir + tag;
        MDWFStep0Table sigmaTable(commBase, base + "_sigma.tsv",
                                  "# term\tK\teval_0\teval_n\tlag\tvariant\tk\tcosine   (principal angles between the "
                                  "variant's subspace and the exact one at eval_n; T1 has the span of T0)");
        MDWFStep0Table ritzTable(commBase, base + "_ritz.tsv",
                                 "# term\tK\teval_0\teval_n\tlag\tvariant\tk\ttheta\tresidual\texact_lambda   (theta = "
                                 "Ritz value in A_n, T0: Rayleigh quotient; residual = |A v - theta v|, T0: |A v - "
                                 "lambda_stale v|)");
        MDWFStep0Table cgTable(commBase, base + "_cg.tsv",
                               "# term\tK\teval_0\teval_n\tlag\trhs\tvariant\tmode\titerations\tconverged\t"
                               "true_residual\tinitial_residual\tseconds   (none: K = 0)");
        std::unique_ptr<Op> opA = makeOperator(0, term, "a"), opB = makeOperator(1, term, "b");
        Op *ops[2] = {opA.get(), opB.get()};
        std::map<std::pair<int, int>, bool> noneDone;   // (lag, rhs kind): the undeflated solve is K independent

        for (const int K : kList) {
            int cur = 0;
            loadGauge(evals[0], cur);
            ops[cur]->refresh();
            Set stale, t2, t3;
            std::vector<double> lambdaStale;
            mdwfDeflReadSet<HaloDepth, Ls>(commBase, stale, K, lambdaStale, mdwfStep0Name(outDir, "vec", evals[0], term));
            mdwfDeflAllocate(t2, K, commBase);
            mdwfDeflAllocate(t3, K, commBase);
            for (int k = 0; k < K; k++) {
                *t2[k] = *stale[k];
                *t3[k] = *stale[k];
            }
            Vec work(commBase);

            for (int m = 1; m <= chainLength; m++) {
                const int prev = cur;
                cur = 1 - cur;
                loadGauge(evals[m], cur);
                ops[cur]->refresh();
                Op &A = *ops[cur], &Aprev = *ops[prev];
                const bool measure = contains(lags, m) || contains(cgLags, m);
                const auto stepStart = std::chrono::steady_clock::now();

                MDWFDeflRitz ritz1, ritz2, ritz3;
                std::vector<double> quotient0(K, 0.0), residual0(K, 0.0);
                {
                    Set x, z;
                    mdwfDeflAllocate(x, K, commBase);
                    mdwfDeflAllocate(z, K, commBase);
                    ritz2 = mdwfDeflTrackStep(A, &Aprev, t2, K, x, z, work, measure);                 // T2
                    ritz3 = mdwfDeflTrackStep(A, static_cast<Op *>(nullptr), t3, K, x, z, work, measure);   // T3
                    if (measure) {
                        // T1: Rayleigh-Ritz of the stale span; T0: the stale vectors as they are.
                        ritz1 = mdwfDeflRayleighRitz(A, mdwfDeflPointers(stale, K), K, z, work, true);
                        for (int k = 0; k < K; k++) {
                            A.apply(work, *stale[k]);
                            quotient0[k] = real<double>(stale[k]->dot(work));
                            work.axpyThisB(-lambdaStale[k], *stale[k]);
                            residual0[k] = std::sqrt(work.norm2());
                        }
                    }
                }
                rootLogger.info("MDWF deflation track: term ", term, " K ", K, " chain step ", m, " (evaluation ",
                                evals[m], "): tracking ",
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - stepStart).count(),
                                " s");
                if (!measure) {
                    continue;
                }

                Set exact;
                std::vector<double> lambdaExact;
                mdwfDeflReadSet<HaloDepth, Ls>(commBase, exact, K, lambdaExact,
                                              mdwfStep0Name(outDir, "vec", evals[m], term));
                const std::vector<const Vec *> pExact = mdwfDeflPointers(exact, K), pStale = mdwfDeflPointers(stale, K),
                                               p2 = mdwfDeflPointers(t2, K), p3 = mdwfDeflPointers(t3, K);
                if (contains(lags, m)) {
                    const char *names[3] = {"T0", "T2", "T3"};
                    const std::vector<const Vec *> *sets[3] = {&pStale, &p2, &p3};
                    for (int s = 0; s < 3; s++) {
                        const std::vector<double> sigma = mdwfDeflPrincipalCosines(pExact, *sets[s]);
                        double capture = 0.0;
                        for (int k = 0; k < K; k++) {
                            capture += sigma[k] * sigma[k] / K;
                            sigmaTable.row(term, K, evals[0], evals[m], m, names[s], k, sigma[k]);
                        }
                        rootLogger.info("MDWF deflation track: term ", term, " K ", K, " lag ", m, " ", names[s],
                                        ": mean cos^2 with the exact subspace = ", capture, ", smallest cosine ",
                                        sigma.back());
                    }
                    for (int k = 0; k < K; k++) {
                        ritzTable.row(term, K, evals[0], evals[m], m, "T0", k, quotient0[k], residual0[k], lambdaExact[k]);
                        ritzTable.row(term, K, evals[0], evals[m], m, "T1", k, ritz1.theta[k], ritz1.residual[k],
                                      lambdaExact[k]);
                        ritzTable.row(term, K, evals[0], evals[m], m, "T2", k, ritz2.theta[k], ritz2.residual[k],
                                      lambdaExact[k]);
                        ritzTable.row(term, K, evals[0], evals[m], m, "T3", k, ritz3.theta[k], ritz3.residual[k],
                                      lambdaExact[k]);
                    }
                }
                if (!contains(cgLags, m)) {
                    continue;
                }

                Vec b(commBase), x(commBase);
                Subspace sub;
                for (int kind = 0; kind < 2; kind++) {
                    const char *rhsName = kind == 0 ? "random" : "actual";
                    if (kind == 0) {
                        if (param.rhs_random() == 0) {
                            continue;
                        }
                        b.noise(seed + 1000003ULL * evals[m] + 7919ULL * term);
                    } else {
                        const std::string rhsBase = mdwfDeflationDumpTerm(dumpDir, evals[m], term, "rhs");
                        if (param.rhs_actual() == 0 || !mdwfStep0Exists(mdwfFieldDumpRankFile(rhsBase, 0))) {
                            continue;
                        }
                        readField(b, rhsBase);
                    }
                    auto solve = [&](const char *variant, const Subspace *subspace, MDWFDeflCGMode cgMode, int k) {
                        const MDWFDeflCGResult r = mdwfDeflCG(A, x, b, subspace, cgMode, precision, maxIter, commBase);
                        const char *modeName = cgMode == MDWFDeflNone ? "none" : (cgMode == MDWFDeflInit ? "init" : "project");
                        cgTable.row(term, k, evals[0], evals[m], m, rhsName, variant, modeName, r.iterations,
                                    r.converged ? 1 : 0, r.trueResidual, r.initialResidual, r.seconds);
                        rootLogger.info("MDWF deflation track: CG term ", term, " K ", k, " lag ", m, " rhs ", rhsName,
                                        " ", variant, " (", modeName, "): ", r.iterations, " iterations, true residual ",
                                        r.trueResidual, ", after the initial guess ", r.initialResidual, ", ",
                                        r.seconds, " s", r.converged ? "" : "  NOT CONVERGED");
                    };
                    if (!noneDone[{m, kind}]) {
                        solve("none", nullptr, MDWFDeflNone, 0);
                        noneDone[{m, kind}] = true;
                    }
                    sub.setupSpectral(pStale, lambdaStale);
                    solve("T0", &sub, MDWFDeflInit, K);
                    const char *names[4] = {"exact", "T1", "T2", "T3"};
                    const std::vector<const Vec *> *sets[4] = {&pExact, &pStale, &p2, &p3};
                    for (int s = 0; s < 4; s++) {
                        sub.setup(A, *sets[s], commBase);
                        solve(names[s], &sub, MDWFDeflProject, K);
                        if (param.cg_init() != 0) {
                            solve(names[s], &sub, MDWFDeflInit, K);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFDeflationStep0Parameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfDeflationStep0.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        if (param.ls() == 8) {
            runMDWFDeflationStep0<8>(commBase, param);
        } else if (param.ls() == 16) {
            runMDWFDeflationStep0<16>(commBase, param);
        } else if (param.ls() == 24) {
            runMDWFDeflationStep0<24>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0 supports Ls = 8, 16 or 24, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
