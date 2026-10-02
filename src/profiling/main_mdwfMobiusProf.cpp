/*
 * MDWF general-Mobius baseline benchmark.
 *
 * Times the current correctness-first MDWF scaffold on one GPU: the per-slice
 * Wilson kernel, the fifth-direction coupling, the clover field-strength
 * precompute, the clover slice, the full Mobius operator (c_sw = 0 and
 * clover), its adjoint, the normal operator, and optionally a coupled-CG
 * solve. With run_even_odd (default on) also the even/odd pieces the m_res and
 * RHMC solves use (MDWFMobiusEvenOdd.h): refresh (clover blocks and block
 * inverses), M_ee, M_oo^-1, the hopping block, the Schur complement, the
 * normal operator Mhat^+ Mhat, the per-iteration halo update and CG vector
 * operations on even-site fields, and (double, run_cg) an even/odd CG solve. It records a baseline for the performance roadmap and the SIMULATeQCD
 * side of the Grid comparison; see src/experimental/mdwf/BENCHMARK_PROTOCOL.md
 * for the measurement definitions and the comparison protocol.
 *
 * Every measurement prints one "MDWF_BENCH" line. Timing uses GPU events
 * after warm-up applications. "hop_gflops" uses the stated convention of 1320
 * flops per 5D site per Wilson hopping application and is only a
 * normalization; comparisons with Grid should use the time for an identical
 * operation, volume, Ls, precision, and preconditioning.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFMobiusEvenOdd.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

class MDWFMobiusProfParameters : public LatticeParameters {
public:
    Parameter<int> Ls;
    Parameter<double> M5;
    Parameter<double> mf;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<int> reps;
    Parameter<int> warmup;
    Parameter<int> seed;
    Parameter<bool> run_single;
    Parameter<bool> run_cg;
    Parameter<bool> run_even_odd;
    Parameter<bool> run_cg_unpreconditioned;
    Parameter<double> cg_tolerance;
    Parameter<int> cg_max_iter;

    MDWFMobiusProfParameters() {
        addDefault(Ls, "Ls", 16);
        addDefault(M5, "M5", 1.8);
        addDefault(mf, "mf", 0.05);
        addDefault(b5, "b5", 1.5);
        addDefault(csw, "c_sw", 0.5);
        addDefault(reps, "reps", 20);
        addDefault(warmup, "warmup", 3);
        addDefault(seed, "seed", 1337);
        addDefault(run_single, "run_single", true);
        addDefault(run_cg, "run_cg", true);
        addDefault(run_even_odd, "run_even_odd", true);
        addDefault(run_cg_unpreconditioned, "run_cg_unpreconditioned", true);
        addDefault(cg_tolerance, "cg_tolerance", 1e-8);
        addDefault(cg_max_iter, "cg_max_iter", 20000);
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusProfSource {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(1e-4) * static_cast<floatT>((site.isite * 7 + component) % 1000),
                static_cast<floatT>(0.01) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.002) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<class floatT, size_t HaloDepth>
void setupMDWFMobiusProfGauge(Gaugefield<floatT, true, HaloDepth, R18> &gauge,
                              MDWFMobiusProfParameters &param) {
    if (param.GaugefileName.isSet()) {
        const std::string format = param.format.isSet() ? param.format() : std::string("nersc");
        rootLogger.info("MDWF Mobius profile: reading gauge configuration ", param.GaugefileName(),
                        " (format ", format, ")");
        gauge.readconf(param.GaugefileName(), format);
    } else if (param.use_unit_conf()) {
        rootLogger.info("MDWF Mobius profile: unit gauge configuration");
        gauge.one();
    } else {
        rootLogger.info("MDWF Mobius profile: random gauge configuration, seed ", param.seed());
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(param.seed());
        d_rand = h_rand;
        gauge.random(d_rand.state);
    }
    gauge.updateAll();
}

template<class floatT, size_t Ls>
void runMDWFMobiusProfile(CommunicationBase &commBase, MDWFMobiusProfParameters &param, bool runCG) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<floatT, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepth, Ls>;
    using CloverField = Spinorfield<floatT, true, All, HaloDepth, 18, 1>;
    using PlainForward = MDWFMobiusLinearOperator<floatT, HaloDepth, HaloDepth, Ls>;
    using CloverForward = MDWFMobiusCloverLinearOperator<floatT, HaloDepth, HaloDepth, Ls>;
    using CloverAdjoint = MDWFMobiusCloverAdjointLinearOperator<floatT, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<CloverForward, CloverAdjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<floatT, HaloDepth, HaloDepth, Ls, Normal>;

    const std::string precision = sizeof(floatT) == sizeof(double) ? "double" : "single";
    const LatticeData lat = GInd::getLatData();
    const double sites4 = static_cast<double>(lat.globvol4);
    const double sites5 = sites4 * static_cast<double>(Ls);
    const std::string latticeLabel = std::to_string(lat.globLX) + "x" + std::to_string(lat.globLY) + "x"
                                     + std::to_string(lat.globLZ) + "x" + std::to_string(lat.globLT);

    const floatT M5 = static_cast<floatT>(param.M5());
    const floatT mf = static_cast<floatT>(param.mf());
    const floatT b5 = static_cast<floatT>(param.b5());
    const floatT csw = static_cast<floatT>(param.csw());
    const floatT mass = mdwfShamirKernelMass(M5);
    const int reps = param.reps();
    const int warmup = param.warmup();

    rootLogger.info("MDWF Mobius profile: precision = ", precision, ", Ls = ", Ls, ", lattice = ", latticeLabel,
                    ", M5 = ", param.M5(), ", mf = ", param.mf(), ", b5 = ", param.b5(),
                    ", c5 = ", param.b5() - 1.0, ", c_sw = ", param.csw(),
                    ", reps = ", reps, ", warmup = ", warmup);

    constexpr bool isDouble = std::is_same<floatT, double>::value;
    // Read (or generate) in double; the single-precision pass uses the same links converted.
    Gaugefield<double, true, HaloDepth, R18> gaugeD(commBase, "MDWF_mobius_prof_dlinks");
    setupMDWFMobiusProfGauge(gaugeD, param);
    Gauge gauge(commBase, "MDWF_mobius_prof_gauge");
    if constexpr (isDouble) {
        gauge = gaugeD;
    } else {
        gauge.convert_precision(gaugeD);
    }
    gauge.updateAll();

    Spinor in(commBase, "MDWF_mobius_prof_in");
    Spinor out(commBase, "MDWF_mobius_prof_out");
    Spinor tmp(commBase, "MDWF_mobius_prof_tmp");
    in.template iterateOverBulk<>(FillMDWFMobiusProfSource<floatT, All, HaloDepth, Ls>());
    in.updateAll();

    CloverField fmunuUpper(commBase, "MDWF_mobius_prof_fmunu_upper");
    CloverField fmunuLower(commBase, "MDWF_mobius_prof_fmunu_lower");
    CloverField fmunuInvUpper(commBase, "MDWF_mobius_prof_fmunu_inv_upper");
    CloverField fmunuInvLower(commBase, "MDWF_mobius_prof_fmunu_inv_lower");

    const MDWFFifthDimCoefficients<floatT> shiftCoeff = mdwfShamirFifthDimCoefficients(mf);

    StopWatch<true> timer;
    auto timeOperation = [&](const std::string &op, double hopApplications, const std::function<void()> &apply) {
        for (int i = 0; i < warmup; i++) {
            apply();
        }
        timer.reset();
        timer.start();
        for (int i = 0; i < reps; i++) {
            apply();
        }
        timer.stop();
        const double msPerApp = timer.milliseconds() / static_cast<double>(reps);
        const double nsPer5dSite = msPerApp * 1e6 / sites5;
        const double hopGflops = hopApplications > 0.0
                                 ? hopApplications * 1320.0 * sites5 / (msPerApp * 1e-3) * 1e-9 : 0.0;
        rootLogger.info("MDWF_BENCH precision=", precision, " Ls=", Ls, " lattice=", latticeLabel,
                        " op=", op, " ms_per_app=", msPerApp, " ns_per_5d_site=", nsPer5dSite,
                        " hop_gflops=", hopGflops);
    };

    // Unpreconditioned operator: double only (the shared Wilson/clover functors it uses compile only in double).
    if constexpr (isDouble) {
        timeOperation("wilson_slice", 1.0, [&]() {
            applyMDWFWilsonSlice<floatT, HaloDepth, HaloDepth, Ls>(out, gauge, tmp, in, mass, static_cast<floatT>(0.0));
        });

        timeOperation("fifth_dim_coupling", 0.0, [&]() {
            applyMDWFFifthDimCoupling<floatT, true, All, HaloDepth, Ls>(out, in, shiftCoeff);
        });

        // Mirrors the precompute that applyMDWFCloverWilsonSlice repeats on every call.
        timeOperation("clover_fmunu_precompute", 0.0, [&]() {
            CalcGSite<All, HaloDepth> calcGSite;
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                preCalcFmunu<floatT, HaloDepth>(gauge, fmunuUpper, fmunuLower, fmunuInvUpper, fmunuInvLower, mass, csw),
                calcGSite, GInd::getLatData().vol4);
            fmunuUpper.updateAll();
            fmunuLower.updateAll();
        });

        timeOperation("clover_slice", 1.0, [&]() {
            applyMDWFCloverWilsonSlice<floatT, HaloDepth, HaloDepth, Ls>(
                out, gauge, tmp, fmunuUpper, fmunuLower, fmunuInvUpper, fmunuInvLower, in, mass, csw);
        });

        PlainForward plainForward(gauge, M5, mf, b5, "MDWF_mobius_prof_plain_forward");
        timeOperation("mobius_M_csw0", 1.0, [&]() { plainForward.apply(out, in, false); });

        CloverForward cloverForward(gauge, M5, mf, b5, csw, "MDWF_mobius_prof_clover_forward");
        timeOperation("mobius_M_clover", 1.0, [&]() { cloverForward.apply(out, in, false); });

        CloverAdjoint cloverAdjoint(gauge, M5, mf, b5, csw, "MDWF_mobius_prof_clover_adjoint");
        timeOperation("mobius_Mdag_clover", 1.0, [&]() { cloverAdjoint.apply(out, in, false); });

        Normal normal(commBase, cloverForward, cloverAdjoint, "MDWF_mobius_prof_normal");
        timeOperation("mobius_MdagM_clover", 2.0, [&]() { normal.apply(out, in, false); });

        // The scaffold CG has no mixed-precision or reliable-update logic, so the
        // time-to-solution baseline is taken in double precision only.
        {
            if (runCG && param.run_cg_unpreconditioned()) {
            NormalAdapter adapter(normal);
            MDWFCoupledCG<floatT, NormalAdapter> cg;
            Spinor solution(commBase, "MDWF_mobius_prof_cg_solution");
            Spinor check(commBase, "MDWF_mobius_prof_cg_check");

            timer.reset();
            timer.start();
            const MDWFCoupledCGResult<floatT> result
                = cg.invert(adapter, solution, in, param.cg_max_iter(), param.cg_tolerance(), true);
            timer.stop();

            normal.apply(check, solution, true);
            check -= in;
            const double trueResidual = std::sqrt(adapter.norm2(check) / std::max(adapter.norm2(in), 1.0));
            const double seconds = timer.seconds();
            const double msPerIteration = result.iterations > 0 ? timer.milliseconds() / result.iterations : 0.0;

            rootLogger.info("MDWF_BENCH precision=", precision, " Ls=", Ls, " lattice=", latticeLabel,
                            " op=cg_MdagM_clover_unpreconditioned",
                            " iterations=", result.iterations, " converged=", result.converged,
                            " tolerance=", param.cg_tolerance(), " residue=", result.residue,
                            " true_residual=", trueResidual, " seconds=", seconds,
                            " ms_per_iteration=", msPerIteration);
            }
        }
    }

    if (param.run_even_odd()) {
        using EvenOdd = MDWFMobiusCloverEvenOdd<floatT, HaloDepth, HaloDepth, Ls>;
        using SpinorE = typename EvenOdd::SpinorE;
        using SpinorO = typename EvenOdd::SpinorO;
        using SchurNormal = MDWFMobiusSchurNormalOperator<EvenOdd>;
        using SchurAdapter = MDWFCoupledSolverAdapter<floatT, HaloDepth, HaloDepth, Ls, SchurNormal>;
        const double half = 0.5;   // even-site fields: half the 5D sites

        using EvenOddD = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
        EvenOddD refEo(gaugeD, param.M5(), param.mf(), param.b5(), param.csw(), "MDWF_mobius_prof_refeo");
        EvenOdd eo(gauge, M5, mf, b5, csw, "MDWF_mobius_prof_eo");
        // double: eo.refresh(); float: blocks converted from the refreshed double operator (as in the mixed solver).
        auto refreshEo = [&]() {
            if constexpr (isDouble) {
                eo.refresh();
            } else {
                refEo.refresh();
                eo.refreshFrom(refEo);
            }
        };
        SpinorE inE(commBase, "MDWF_mobius_prof_eo_ine");
        SpinorE outE(commBase, "MDWF_mobius_prof_eo_oute");
        SpinorO inO(commBase, "MDWF_mobius_prof_eo_ino");
        SpinorO outO(commBase, "MDWF_mobius_prof_eo_outo");
        refreshEo();
        EvenOdd::split(inE, inO, in);
        inE.updateAll();
        inO.updateAll();

        timeOperation(isDouble ? "eo_refresh" : "eo_refresh_from_double", 0.0, [&]() { refreshEo(); });
        timeOperation("eo_Mee", 0.0, [&]() { eo.Mee(outE, inE); });
        timeOperation("eo_MooInv", 0.0, [&]() { eo.MooInv(outO, inO); });
        timeOperation("eo_Meo_hop", half, [&]() { eo.Meo(outE, inO); });
        timeOperation("eo_schur", 2.0 * half, [&]() { eo.schur(outE, inE, false); });
        timeOperation("eo_schur_dagger", 2.0 * half, [&]() { eo.schur(outE, inE, true); });
        SchurNormal schurNormal(eo, commBase, "MDWF_mobius_prof_eo_normal");
        timeOperation("eo_MhatdagMhat", 4.0 * half, [&]() { schurNormal.apply(outE, inE, false); });
        // The rest of one CG iteration: halo update of the search vector, two reductions, three vector updates.
        SchurAdapter schurAdapter(schurNormal);
        timeOperation("eo_cg_updateAll", 0.0, [&]() { inE.updateAll(); });
        timeOperation("eo_cg_norm2", 0.0, [&]() { volatile double n = schurAdapter.norm2(inE); (void)n; });
        timeOperation("eo_cg_axpy", 0.0, [&]() {
            outE.template axpyThisB<64>(static_cast<floatT>(1e-3), inE);
        });

        if constexpr (std::is_same<floatT, double>::value) {
            if (runCG) {
                MDWFCoupledCG<floatT, SchurAdapter> cg;
                SpinorE solutionE(commBase, "MDWF_mobius_prof_eo_cg_sol");
                timer.reset();
                timer.start();
                const MDWFCoupledCGResult<floatT> result
                    = cg.invert(schurAdapter, solutionE, inE, param.cg_max_iter(), param.cg_tolerance(), true);
                timer.stop();
                rootLogger.info("MDWF_BENCH precision=", precision, " Ls=", Ls, " lattice=", latticeLabel,
                                " op=cg_MhatdagMhat_clover_even_odd iterations=", result.iterations,
                                " converged=", result.converged, " tolerance=", param.cg_tolerance(),
                                " residue=", result.residue, " seconds=", timer.seconds(), " ms_per_iteration=",
                                result.iterations > 0 ? timer.milliseconds() / result.iterations : 0.0);
            }
        }
    }

}

template<size_t Ls>
void runMDWFMobiusProfileForLs(CommunicationBase &commBase, MDWFMobiusProfParameters &param) {
    if (param.run_single()) {
        runMDWFMobiusProfile<float, Ls>(commBase, param, false);
    }
    runMDWFMobiusProfile<double, Ls>(commBase, param, param.run_cg());
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFMobiusProfParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/profiling/mdwfMobiusProf.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        switch (param.Ls()) {
        case 8:
            runMDWFMobiusProfileForLs<8>(commBase, param);
            break;
        case 12:
            runMDWFMobiusProfileForLs<12>(commBase, param);
            break;
        case 16:
            runMDWFMobiusProfileForLs<16>(commBase, param);
            break;
        default:
            throw std::runtime_error(stdLogger.fatal(
                "MDWF Mobius profile supports Ls = 8, 12, 16; got Ls = ", param.Ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
