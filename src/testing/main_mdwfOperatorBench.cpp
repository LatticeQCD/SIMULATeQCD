/*
 * MDWF operator benchmark (CPC paper F3): the even/odd Schur normal operator Mhat^+ Mhat (MDWFMobiusSchurNormalOperator,
 * the operator of every light and EOFA CG iteration) on one GPU, in double and single precision, fused (four kernels
 * per Mhat, MDWFMobiusCloverEvenOdd::fusedSchur) and unfused (block path M_ee - M_eo M_oo^-1 M_oe, mdwfFusedSchur() =
 * false), c_sw = 0, Ls = 16.
 *
 * Reported per variant: ms per Mhat^+ Mhat, effective bandwidth (model bytes / time), model GFLOP/s. Reference: the
 * achievable memory bandwidth of the same GPU, measured here with a field copy (a = b) and an axpy (a += x b) on 5D
 * fields of the same size (STREAM-like copy / triad).
 *
 * Models per Mhat (half-lattice field F = V/2 * Ls * 24 reals, gauge G = V * 4 * 18 reals; minimal (compulsory) DRAM
 * traffic of the fused kernels, every field read or written once per kernel, links once per hop):
 *   Mhat:   DinMxx 1F in, 2F out; hop 1F + G in, 1F out; dense 1F in, 1F out; hop 2F + G in, 1F out  = 10F + 2G
 *   Mhat^+: hop 1F + G in, 1F out; dense 1F, 1F; hop 1F + G, 1F; Mxx^+ - Din^+ g5 2F in, 1F out    =  9F + 2G
 *   Mhat^+ Mhat = 19F + 4G (the same model is used for the unfused path, so its effective bandwidth shows the
 *   cost of the extra passes). Measured DRAM bytes per kernel come from Nsight Compute (job_f3.sbatch).
 *   Flops per (even site, s) per Mhat: 2 Wilson hops 2 x 1320, dense Ls x Ls fifth-dimension block 48 Ls,
 *   Din / M_xx / subtraction 168; Mhat^+ Mhat = 2 x (2640 + 48 Ls + 168).
 *
 * Parameters: Lattice, Nodes (1 1 1 1), Gaugefile (optional; random links otherwise), M5, b5, mf, n_apply, n_warm,
 * force_halos (0). Output lines start with "F3 " (parsed by dsdr/tools/make_figdata.py).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMobiusEvenOdd.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

class MDWFOperatorBenchParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> mf;
    Parameter<int> n_apply;
    Parameter<int> n_warm;
    Parameter<int> force_halos;

    MDWFOperatorBenchParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 1.9);
        addDefault(mf, "mf", 0.00084525);
        addDefault(n_apply, "n_apply", 200);
        addDefault(n_warm, "n_warm", 10);
        addDefault(force_halos, "force_halos", 0);
    }
};

static double benchSeconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template<class F>
static double timePerCall(F &&f, int nWarm, int n) {
    for (int i = 0; i < nWarm; i++) {
        f();
    }
    gpuError_t err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        throw std::runtime_error(stdLogger.fatal("device synchronize failed before timing"));
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        f();
    }
    err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        throw std::runtime_error(stdLogger.fatal("device synchronize failed after timing"));
    }
    return benchSeconds(t0) / n;
}

template<class floatT, size_t HaloDepth, size_t Ls>
void benchPrecision(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gaugeD,
                    MDWFOperatorBenchParameters &param, uint4 *randState) {
    using EvenOdd = MDWFMobiusCloverEvenOdd<floatT, HaloDepth, HaloDepth, Ls>;
    using EvenOddD = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using SpinorE = typename EvenOdd::SpinorE;
    constexpr bool isDouble = std::is_same<floatT, double>::value;
    const char *prec = isDouble ? "double" : "single";

    EvenOddD eoD(gaugeD, param.M5(), param.mf(), param.b5(), 0.0, "bench_eoD");
    eoD.refresh();
    Gaugefield<floatT, true, HaloDepth, R18> gauge(commBase, std::string("bench_gauge_") + prec);
    gauge.convert_precision(gaugeD);
    mdwfUpdateHalo(gauge);
    EvenOdd eo(gauge, param.M5(), param.mf(), param.b5(), 0.0, std::string("bench_eo_") + prec);
    if constexpr (isDouble) {
        eo.refresh();
    } else {
        eo.refreshFrom(eoD);
    }
    NormalOp normal(eo, commBase, "bench_normal");

    SpinorE in(commBase, std::string("bench_in_") + prec), out(commBase, std::string("bench_out_") + prec),
        ref(commBase, std::string("bench_ref_") + prec);
    in.gauss(randState);
    mdwfUpdateHalo(in);

    const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
    const double vol4 = static_cast<double>(lat.globvol4);
    const double ranks = static_cast<double>(commBase.getNumberProcesses());
    const double w = sizeof(floatT);
    const double F = 0.5 * vol4 / ranks * Ls * 24.0 * w;
    const double G = vol4 / ranks * 4.0 * 18.0 * w;
    const double modelBytes = 19.0 * F + 4.0 * G;
    const double modelFlops = 0.5 * vol4 / ranks * Ls * 2.0 * (2640.0 + 48.0 * Ls + 168.0);

    // achievable bandwidth: copy (read F + write F) and axpy (read 2F + write F) on the same field size
    const double tCopy = timePerCall([&] { out = in; }, param.n_warm(), param.n_apply());
    const double tAxpy = timePerCall([&] { out.template axpyThisB<64>(static_cast<floatT>(0.5), in); },
                                     param.n_warm(), param.n_apply());
    rootLogger.info("F3 stream ", prec, " field_MB ", F / 1e6, " copy_GBs ", 2.0 * F / tCopy / 1e9,
                    " axpy_GBs ", 3.0 * F / tAxpy / 1e9);

    for (int fused = 1; fused >= 0; fused--) {
        mdwfFusedSchur() = (fused == 1);
        normal.apply(out, in);
        if (fused == 1) {
            ref = out;
        } else {
            // fused and block paths are the same operator: |out - ref| / |ref|
            out.template axpyThisB<64>(static_cast<floatT>(-1.0), ref);
            double num = 0.0, den = 0.0;
            for (const COMPLEX(double) &d : out.dotProductStacked(out)) num += real<double>(d);
            for (const COMPLEX(double) &d : ref.dotProductStacked(ref)) den += real<double>(d);
            rootLogger.info("F3 check ", prec, " fused_vs_block_reldiff ", std::sqrt(num / den));
        }
        const double t = timePerCall([&] { normal.apply(out, in); }, param.n_warm(), param.n_apply());
        rootLogger.info("F3 result ", prec, " fused ", fused, " ms_per_normal ", 1e3 * t,
                        " eff_GBs ", modelBytes / t / 1e9, " GFLOPs ", modelFlops / t / 1e9,
                        " model_MB ", modelBytes / 1e6, " model_GFLOP ", modelFlops / 1e9);
    }
    mdwfFusedSchur() = true;
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFOperatorBenchParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfOperatorBench.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261009);
        d_rand = h_rand;

        Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "bench_gauge");
        if (param.GaugefileName.isSet() && !param.GaugefileName().empty()) {
            gauge.readconf_nersc(param.GaugefileName());
        } else {
            gauge.random(d_rand.state);
        }
        gauge.updateAll();

        rootLogger.info("F3 setup lattice ", param.latDim()[0], "x", param.latDim()[1], "x", param.latDim()[2], "x",
                        param.latDim()[3], " Ls 16 ranks ", commBase.getNumberProcesses(), " n_apply ", param.n_apply());
        benchPrecision<double, HaloDepth, 16>(commBase, gauge, param, d_rand.state);
        benchPrecision<float, HaloDepth, 16>(commBase, gauge, param, d_rand.state);
        rootLogger.info("F3 done");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
