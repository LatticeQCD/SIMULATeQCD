/*
 * Spectral gap of the Wilson(-clover) kernel of the Mobius operator against M5: the lowest and highest eigenvalue
 * of H_W^2 = (gamma5 D_W(-M5))^2 = D_W(-M5)^dagger D_W(-M5) on one gauge configuration, for thin or smeared kernel
 * links, at a list of M5 values.
 *
 * Domain-wall fermions work where H_W has a gap: the residual mass is controlled by the eigenvalues of H_W near zero
 * (m_res ~ exp(-lambda Ls) from the extended modes plus a term from localized near-zero modes). The M5 at which
 * lambda_min(H_W^2) closes marks the edge of the domain-wall window; the M5 with the largest lambda_min is the best
 * kernel. Mean field (links ~ u0 times the identity, u0 = plaquette^(1/4)) predicts
 *     lambda_min(M5) = min_n (4 - M5 - 4 u0 + 2 n u0)^2,  n = 0 .. 4 (doubler branches),
 * with the edges at M5 = 4 - 4 u0 and 4 - 2 u0 and the optimum at 4 - 3 u0; this program measures the real curve.
 *
 * Method: plain Lanczos (MDWFSpectralBounds.h) on H_W^2 from a Gaussian start vector; the extreme Ritz values are
 * reported at several step counts, so their convergence is visible (the lowest Ritz value approaches lambda_min
 * from above, i.e. every printed value is an upper bound on the true lambda_min). No eigenvectors.
 *
 * Kernel links as in mdwfResidualMass: thin, or fat7 smeared (smearing = fat7_su3 | fat7_u3, smear_steps), then the
 * fermion boundary phases (antiperiodic_t); D_W includes the clover term for c_sw != 0 (mass = 4 - M5).
 * Output: <measurements_dir>/<output_name>, lines "M5 steps lambda_min lambda_max sqrt(lambda_min)".
 *
 * Usage: ./mdwfKernelSpectrum <param file> [key=value ...]   (parameter/tests/mdwfKernelSpectrum.param)
 * Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFFermionBoundary.h"
#include "../experimental/mdwf/MDWFKernelSmearing.h"
#include "../experimental/mdwf/MDWFPhysicalMapping.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"
#include "../experimental/mdwf/MDWFWilsonSlice.h"
#include "../gauge/gaugeAction.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFKernelSpectrumParameters : public LatticeParameters {
public:
    DynamicParameter<double> M5;
    Parameter<double> csw;
    Parameter<int> antiperiodic_t;
    Parameter<std::string> smearing;
    Parameter<int> smear_steps;
    Parameter<int> lanczos_steps;
    Parameter<int> seed;
    Parameter<std::string> output_name;

    MDWFKernelSpectrumParameters() {
        addOptional(M5, "M5");
        addDefault(csw, "c_sw", 0.0);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(smearing, "smearing", std::string("none"));
        addDefault(smear_steps, "smear_steps", 2);
        addDefault(lanczos_steps, "lanczos_steps", 2000);
        addDefault(seed, "seed", 20261013);
        addDefault(output_name, "output_name", std::string("mdwfKernelSpectrum.dat"));
    }
};

// H_W^2 = gamma5 D_W gamma5 D_W on one slice (Ls = 1), D_W with clover, mass = 4 - M5.
template<size_t HaloDepth>
class MDWFHermitianWilsonSquared {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Clover = Spinorfield<double, true, All, HaloDepth, 18, 1>;

private:
    Gauge &_gauge;
    double _mass, _csw;
    Spinor _t1, _t2, _tmp;
    Clover _fu, _fl, _fiu, _fil;

public:
    MDWFHermitianWilsonSquared(Gauge &gauge, double M5, double csw, const std::string &name)
        : _gauge(gauge), _mass(mdwfShamirKernelMass<double>(M5)), _csw(csw),
          _t1(gauge.getComm(), name + "_t1"), _t2(gauge.getComm(), name + "_t2"), _tmp(gauge.getComm(), name + "_tmp"),
          _fu(gauge.getComm(), name + "_fu"), _fl(gauge.getComm(), name + "_fl"),
          _fiu(gauge.getComm(), name + "_fiu"), _fil(gauge.getComm(), name + "_fil") {}

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(_t1, _gauge, _tmp, _fu, _fl, _fiu, _fil, in, _mass,
                                                                    _csw, false);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(_t2, _t1, true);
        applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(_t1, _gauge, _tmp, _fu, _fl, _fiu, _fil, _t2,
                                                                    _mass, _csw, false);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _t1, update);
    }
};

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFKernelSpectrumParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfKernelSpectrum.param", argc, argv);
        commBase.init(param.nodeDim());
        const size_t HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        typedef GIndexer<All, HaloDepth> GInd;
        if (GInd::getLatData().vol4 != GInd::getLatData().globvol4) {
            throw std::runtime_error(stdLogger.fatal("mdwfKernelSpectrum is single-rank"));
        }
        if (!param.M5.isSet() || param.M5.numberValues() == 0) {
            throw std::runtime_error(stdLogger.fatal("mdwfKernelSpectrum needs a list M5 = m1 m2 ..."));
        }
        if (!param.GaugefileName.isSet()) {
            throw std::runtime_error(stdLogger.fatal("mdwfKernelSpectrum needs Gaugefile"));
        }
        using Gauge = Gaugefield<double, true, HaloDepth, R18>;
        using Op = MDWFHermitianWilsonSquared<HaloDepth>;
        using Spinor = typename Op::Spinor;

        Gauge gauge(commBase, "MDWF_ks_gauge");
        Gauge kernelGauge(commBase, "MDWF_ks_kgauge");
        Gauge fermionGauge(commBase, "MDWF_ks_fgauge");
        gauge.readconf_nersc(param.GaugefileName());
        gauge.updateAll();
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        const double plaquette = static_cast<double>(action.plaquette());
        const MDWFKernelSmearingType type = mdwfKernelSmearingFromString(param.smearing());
        mdwfSmearKernelLinks<HaloDepth>(kernelGauge, gauge, type, param.smear_steps(), "MDWF_ks_smear");
        double kernelPlaquette = plaquette;
        if (type != MDWFKernelSmearingType::None) {
            kernelPlaquette = mdwfKernelLinkReport<HaloDepth>(kernelGauge, "MDWF_ks_rep").plaquette;
        }
        mdwfFermionGaugeField<HaloDepth>(fermionGauge, kernelGauge, param.antiperiodic_t() != 0);
        const double u0 = std::pow(kernelPlaquette, 0.25);
        const std::string kernelName = type == MDWFKernelSmearingType::None
                                       ? std::string("thin")
                                       : param.smearing() + " x" + std::to_string(param.smear_steps());
        rootLogger.info("MDWF kernel spectrum: ", param.GaugefileName(), ", kernel links ", kernelName,
                        ", plaquette thin ", plaquette, ", kernel ", kernelPlaquette, " (u0 = ", u0,
                        ": mean-field window ", 4.0 - 4.0 * u0, " ... ", 4.0 - 2.0 * u0, ", optimum ", 4.0 - 3.0 * u0,
                        "), c_sw = ", param.csw(), ", Lanczos steps ", param.lanczos_steps());

        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(param.seed());
        d_rand = h_rand;
        Spinor start(commBase, "MDWF_ks_start");
        start.gauss(d_rand.state);
        start.updateAll();

        const int steps = param.lanczos_steps();
        const std::vector<int> checkpoints = {steps / 8, steps / 4, steps / 2, steps};
        std::ofstream out;
        const std::string path = param.measurements_dir() + "/" + param.output_name();
        if (commBase.IamRoot()) {
            out.open(path, std::ios::out | std::ios::trunc);
            if (!out) {
                throw std::runtime_error(stdLogger.fatal("cannot open ", path));
            }
            out << "# mdwfKernelSpectrum: " << param.GaugefileName() << ", kernel links " << kernelName << ", c_sw "
                << param.csw() << ", plaquette thin " << std::setprecision(10) << plaquette << ", kernel "
                << kernelPlaquette << ", u0 " << u0 << "\n";
            out << "# lambda of H_W^2 = (gamma5 D_W(-M5))^2 from plain Lanczos (lowest value: upper bound on lambda_min)\n";
            out << "# columns: M5  steps  lambda_min  lambda_max  sqrt(lambda_min)\n";
        }
        for (const double m5 : param.M5.get()) {
            Op op(fermionGauge, m5, param.csw(), "MDWF_ks_op");
            MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, 1, Op> adapter(op);
            const MDWFLanczosResult r = mdwfLanczosExtremes(adapter, start, steps, checkpoints, "MDWF_ks");
            for (const MDWFLanczosCheckpoint &c : r.checkpoints) {
                rootLogger.info("MDWF kernel spectrum M5 = ", m5, ": ", c.steps, " Lanczos steps, lambda_min(H_W^2) = ",
                                c.lambda_min, ", lambda_max = ", c.lambda_max, ", |lambda|_min(H_W) = ",
                                std::sqrt(std::max(c.lambda_min, 0.0)), r.breakdown ? " (breakdown)" : "");
                if (commBase.IamRoot()) {
                    out << std::setprecision(10) << m5 << " " << c.steps << " " << c.lambda_min << " " << c.lambda_max
                        << " " << std::sqrt(std::max(c.lambda_min, 0.0)) << "\n";
                }
            }
            if (commBase.IamRoot()) {
                out.flush();
            }
        }
        rootLogger.info("MDWF kernel spectrum written to ", path);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
