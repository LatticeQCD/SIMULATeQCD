/*
 * MDWF spectral-bounds tool (for choosing AlgRemez intervals of the MDWF RHMC).
 *
 * 1. Self-test of mdwfLanczosExtremes (MDWFSpectralBounds.h) on a diagonal
 *    mock operator with lambda(site, s) = 1 + 0.1 s + 0.001 (isite mod 97):
 *    776 distinct eigenvalues, exact lambda_min = 1 and lambda_max = 1.796
 *    (Ls = 8). The final Ritz extremes must match to 1e-6 (relative).
 * 2. Lanczos extremes of the Mobius clover M(m)^\dagger M(m) for each mass in
 *    `masses` on the gauge field (Gaugefile, NERSC; random if unset, which is
 *    not representative of a thermalized ensemble), printed at increasing
 *    step counts together with the suggested interval
 *    [lambda_min / margin, lambda_max * margin].
 *
 * Usage (from the build's testing directory):
 *   ./mdwfSpectralBounds <param file>
 * See parameter/tests/mdwfSpectralBounds.param. Single rank; Ls = 8.
 */

#include "../simulateqcd.h"
#include "../gauge/gaugeAction.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFSpectralParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<double, 3> masses;
    Parameter<int> lanczos_steps;
    Parameter<double> margin;
    Parameter<int> seed;

    MDWFSpectralParameters() {
        const double defaultMasses[3] = {0.02, 0.1, 1.0};
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 1.5);
        addDefault(csw, "c_sw", 0.5);
        addDefault<double, 3>(masses, "masses", defaultMasses);
        addDefault(lanczos_steps, "lanczos_steps", 1000);
        addDefault(margin, "margin", 2.0);
        addDefault(seed, "seed", 20260927);
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFSpectralMockDiagonal {
    Vect12ArrayAcc<floatT> _in;

    template<bool onDevice>
    explicit MDWFSpectralMockDiagonal(const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &in)
        : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT lambda = static_cast<floatT>(1.0) + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack)
                              + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite % 97);
        return lambda * _in.getElement(site);
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFSpectralMockOperator {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        out.template iterateOverBulk<>(MDWFSpectralMockDiagonal<double, All, HaloDepth, Ls>(in));
        if (update) {
            out.updateAll();
        }
    }
};

std::vector<int> mdwfSpectralCheckpoints(int maxSteps) {
    std::vector<int> steps;
    for (int k = 25; k < maxSteps; k *= 2) {
        steps.push_back(k);
    }
    steps.push_back(maxSteps);
    return steps;
}

template<size_t Ls>
void runMDWFSpectralBounds(CommunicationBase &commBase, MDWFSpectralParameters &param) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MockOperator = MDWFSpectralMockOperator<HaloDepth, Ls>;
    using MockAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, MockOperator>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;

    if (GInd::getLatData().vol4 != GInd::getLatData().globvol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF spectral-bounds tool is single-rank only"));
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    Spinor start(commBase, "MDWF_spectral_start");

    // --- Part 1: mock self-test with exactly known extremes. ---
    {
        MockOperator mock;
        MockAdapter mockAdapter(mock);
        start.gauss(d_rand.state);
        const int mockSteps = 400;
        const MDWFLanczosResult mockResult = mdwfLanczosExtremes(mockAdapter, start, mockSteps,
                                                                 mdwfSpectralCheckpoints(mockSteps), "MDWF_spec_mock");
        for (const MDWFLanczosCheckpoint &c : mockResult.checkpoints) {
            rootLogger.info("MDWF spectral bounds mock: steps = ", c.steps, ", lambda_min = ", c.lambda_min,
                            ", lambda_max = ", c.lambda_max, " (exact 1, 1.796)");
        }
        const MDWFLanczosCheckpoint &last = mockResult.checkpoints.back();
        const double minErr = std::abs(last.lambda_min - 1.0);
        const double maxErr = std::abs(last.lambda_max - 1.796) / 1.796;
        const bool mockPassed = minErr <= 1e-6 && maxErr <= 1e-6;
        rootLogger.info("MDWF spectral bounds mock self-test: relErr(lambda_min) = ", minErr,
                        ", relErr(lambda_max) = ", maxErr, ", breakdown = ", mockResult.breakdown,
                        ", passed = ", mockPassed);
        if (!mockPassed) {
            throw std::runtime_error(stdLogger.fatal("MDWF spectral bounds mock self-test failed"));
        }
    }

    // --- Part 2: Mobius clover M^\dagger M on the gauge field. ---
    Gauge gauge(commBase, "MDWF_spectral_gauge");
    if (param.GaugefileName.isSet()) {
        rootLogger.info("MDWF spectral bounds: reading gauge configuration ", param.GaugefileName());
        gauge.readconf_nersc(param.GaugefileName());
    } else {
        rootLogger.warn("MDWF spectral bounds: no Gaugefile set, using a random (hot) gauge field; "
                        "not representative of a thermalized ensemble");
        gauge.random(d_rand.state);
    }
    gauge.updateAll();
    {
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        rootLogger.info("MDWF spectral bounds: plaquette = ", action.plaquette(), ", M5 = ", param.M5(),
                        ", b5 = ", param.b5(), ", c5 = ", param.b5() - 1.0, ", c_sw = ", param.csw(), ", Ls = ", Ls,
                        ", Lanczos steps = ", param.lanczos_steps(), ", margin = ", param.margin());
    }

    const std::vector<int> checkpoints = mdwfSpectralCheckpoints(param.lanczos_steps());
    for (int i = 0; i < 3; i++) {
        const double mass = param.masses[i];
        const std::string name = "MDWF_spec_m" + std::to_string(i);
        Forward forward(gauge, param.M5(), mass, param.b5(), param.csw(), name + "_forward");
        Adjoint adjoint(gauge, param.M5(), mass, param.b5(), param.csw(), name + "_adjoint");
        Normal normal(commBase, forward, adjoint, name + "_normal");
        NormalAdapter adapter(normal);

        start.gauss(d_rand.state);
        const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, param.lanczos_steps(), checkpoints,
                                                             name);
        for (const MDWFLanczosCheckpoint &c : result.checkpoints) {
            rootLogger.info("MDWF spectral bounds mf = ", mass, ": steps = ", c.steps,
                            ", lambda_min = ", c.lambda_min, ", lambda_max = ", c.lambda_max,
                            ", condition = ", c.lambda_max / c.lambda_min);
        }
        const MDWFLanczosCheckpoint &last = result.checkpoints.back();
        const MDWFLanczosCheckpoint &previous = result.checkpoints.size() > 1
                                                ? result.checkpoints[result.checkpoints.size() - 2] : last;
        rootLogger.info("MDWF spectral bounds mf = ", mass, " summary: lambda_min = ", last.lambda_min,
                        " (change since ", previous.steps, " steps: ",
                        std::abs(last.lambda_min - previous.lambda_min) / last.lambda_min, "), lambda_max = ",
                        last.lambda_max, " (change: ", std::abs(last.lambda_max - previous.lambda_max) / last.lambda_max,
                        "), suggested interval [", last.lambda_min / param.margin(), ", ",
                        last.lambda_max * param.margin(), "]");
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFSpectralParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfSpectralBounds.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runMDWFSpectralBounds<8>(commBase, param);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
