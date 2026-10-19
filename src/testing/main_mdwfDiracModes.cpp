/*
 * Low eigenmodes of the MDWF kernel or of the 4D effective MDWF Dirac operator on one gauge configuration, with
 * per-mode localization, chirality and Ginsparg-Wilson (sign-function) defect.
 *
 *   spectrum = kernel   A = H_W^2 = (gamma5 D_W(-M5))^2 (Wilson-clover kernel, no solves)
 *   spectrum = dirac    A = D(m)^dagger D(m), D(m) the 4D effective operator of MDWFEffectiveOperator.h (two
 *                       Pauli-Villars solves per application)
 *
 * The nev lowest eigenpairs of A come from thick-restart Lanczos (MDWFThickRestartLanczos.h), optionally on the
 * Chebyshev-filtered operator -T_d((hi + lo - 2 A) / (hi - lo)) (cheb_degree > 0; lo above the wanted modes, hi above
 * lambda_max: estimated with plain Lanczos for the kernel, 1.02 for the Dirac operator whose spectrum lies in [0, 1]).
 * For every eigenvector y (normalized) the program writes
 *   lambda      <y|A|y>
 *   residual    |A y - lambda y|
 *   chirality   <y|gamma5|y>
 *   ipr         inverse participation ratio V sum_x rho(x)^2, rho(x) = sum over spin and colour |y(x)|^2:
 *               1 for a mode spread evenly over the lattice, V for a mode on one site
 *   gw_defect   <y|1 - eps^2|y> = 1 - |eps y|^2, eps = eps_Ls(H_M) the Mobius sign function (one Pauli-Villars solve;
 *               written when gw_defect = 1): the Ginsparg-Wilson violation seen by this mode
 *   sqrt        sqrt(lambda) (kernel: |eigenvalue of H_W|); dirac: sqrt((lambda - m^2) / (1 - m^2)), the modulus
 *               of the eigenvalue of D(0) if the Ginsparg-Wilson relation held exactly
 *
 * Kernel links as in mdwfResidualMass: thin or fat7 smeared (smearing = fat7_su3 | fat7_u3, smear_steps), then the
 * fermion boundary phases (antiperiodic_t). Ls = 8 or 16; mixed_precision = 1 uses the mixed-precision Pauli-Villars
 * solver. Output: <measurements_dir>/<output_name>.
 *
 * Usage: ./mdwfDiracModes <param file> [key=value ...]   (parameter/tests/mdwfDiracModes.param). Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFEffectiveOperator.h"
#include "../experimental/mdwf/MDWFFermionBoundary.h"
#include "../experimental/mdwf/MDWFKernelSmearing.h"
#include "../experimental/mdwf/MDWFMixedPrecisionSolver.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"
#include "../experimental/mdwf/MDWFThickRestartLanczos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class MDWFDiracModesParameters : public LatticeParameters {
public:
    Parameter<std::string> spectrum;
    Parameter<double> M5;
    Parameter<double> mass;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<int> ls;
    Parameter<int> antiperiodic_t;
    Parameter<std::string> smearing;
    Parameter<int> smear_steps;
    Parameter<int> nev;
    Parameter<int> nbasis;
    Parameter<int> nkeep;
    Parameter<double> tol;
    Parameter<int> max_restarts;
    Parameter<int> cheb_degree;
    Parameter<double> cheb_lo;
    Parameter<double> cheb_hi;
    Parameter<int> hi_lanczos_steps;
    Parameter<int> gw_defect;
    Parameter<int> mixed_precision;
    Parameter<double> pv_precision;
    Parameter<int> pv_max_iter;
    Parameter<int> seed;
    Parameter<std::string> output_name;

    MDWFDiracModesParameters() {
        addDefault(spectrum, "spectrum", std::string("kernel"));
        addDefault(M5, "M5", 1.8);
        addDefault(mass, "mass", 0.0);
        addDefault(b5, "b5", 2.0);
        addDefault(csw, "c_sw", 0.0);
        addDefault(ls, "ls", 16);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(smearing, "smearing", std::string("none"));
        addDefault(smear_steps, "smear_steps", 2);
        addDefault(nev, "nev", 20);
        addDefault(nbasis, "nbasis", 60);
        addDefault(nkeep, "nkeep", 40);
        addDefault(tol, "tol", 1e-8);
        addDefault(max_restarts, "max_restarts", 500);
        addDefault(cheb_degree, "cheb_degree", 0);
        addDefault(cheb_lo, "cheb_lo", 0.0);
        addDefault(cheb_hi, "cheb_hi", 0.0);
        addDefault(hi_lanczos_steps, "hi_lanczos_steps", 60);
        addDefault(gw_defect, "gw_defect", 1);
        addDefault(mixed_precision, "mixed_precision", 0);
        addDefault(pv_precision, "pv_precision", 1e-11);
        addDefault(pv_max_iter, "pv_max_iter", 20000);
        addDefault(seed, "seed", 20261014);
        addDefault(output_name, "output_name", std::string("mdwfDiracModes.dat"));
    }
};

// (sum over spin and colour |y(x)|^2)^2 per site.
template<size_t HaloDepth>
struct MDWFModeDensitySquared {
    Vect12ArrayAcc<double> _y;

    explicit MDWFModeDensitySquared(const MDWFSpinor<double, true, All, HaloDepth, 1> &y) : _y(y.getAccessor()) {}

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const Vect12<double> a = _y.getElement(GInd::getSiteStack(site, 0));
        const double rho = real(a * a);
        return rho * rho;
    }
};

struct MDWFModeRecord {
    double lambda, residual, chirality, ipr, gwDefect, root;
};

template<size_t Ls, bool Mixed>
void runMDWFDiracModes(CommunicationBase &commBase, MDWFDiracModesParameters &param) {
    constexpr size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using Solver = std::conditional_t<Mixed, MDWFMobiusEvenOddMixedSolver<HaloDepth, HaloDepth, Ls>,
                                      MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>>;
    using Effective = MDWFEffectiveDiracOperator<HaloDepth, Ls, Solver>;
    using Normal = MDWFEffectiveNormalOperator<Effective>;
    using Kernel = MDWFHermitianWilsonSquaredOperator<HaloDepth>;

    const bool dirac = param.spectrum() == "dirac";
    if (!dirac && param.spectrum() != "kernel") {
        throw std::runtime_error(stdLogger.fatal("spectrum must be kernel or dirac, got ", param.spectrum()));
    }
    if (!param.GaugefileName.isSet()) {
        throw std::runtime_error(stdLogger.fatal("mdwfDiracModes needs Gaugefile"));
    }
    if (param.cheb_degree() > 0 && !(param.cheb_lo() > 0.0)) {
        throw std::runtime_error(stdLogger.fatal("cheb_degree > 0 needs cheb_lo > 0 (above the wanted eigenvalues)"));
    }

    Gauge gauge(commBase, "MDWF_dm_gauge"), kernelGauge(commBase, "MDWF_dm_kgauge"),
            fermionGauge(commBase, "MDWF_dm_fgauge");
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();
    const MDWFKernelSmearingType type = mdwfKernelSmearingFromString(param.smearing());
    mdwfSmearKernelLinks<HaloDepth>(kernelGauge, gauge, type, param.smear_steps(), "MDWF_dm_smear");
    mdwfFermionGaugeField<HaloDepth>(fermionGauge, kernelGauge, param.antiperiodic_t() != 0);
    const std::string kernelName = type == MDWFKernelSmearingType::None
                                   ? std::string("thin")
                                   : param.smearing() + " x" + std::to_string(param.smear_steps());

    Kernel kernel(fermionGauge, param.M5(), param.csw(), "MDWF_dm_kernel");
    // The 5D Pauli-Villars solver (several Ls = 16 fields) and the Lanczos basis do not fit on a 40 GB A100 together
    // for nbasis ~ 100: for the kernel spectrum it is built only after the eigensolver, once the basis is trimmed to
    // the nev eigenvectors.
    std::unique_ptr<Effective> effective;
    std::unique_ptr<Normal> normal;
    auto makeEffective = [&]() {
        effective.reset(new Effective(fermionGauge, param.M5(), param.mass(), param.b5(), param.csw(),
                                      param.pv_max_iter(), param.pv_precision(), "MDWF_dm_eff"));
        normal.reset(new Normal(*effective));
    };
    if (dirac) {
        makeEffective();
    }
    rootLogger.info("MDWF modes: ", param.GaugefileName(), ", spectrum ", param.spectrum(), ", kernel links ",
                    kernelName, ", M5 = ", param.M5(), ", c_sw = ", param.csw(), ", b5 = ", param.b5(), ", c5 = ",
                    param.b5() - 1.0, ", Ls = ", Ls, ", m = ", param.mass(), ", nev = ", param.nev(), ", nbasis = ",
                    param.nbasis(), ", nkeep = ", param.nkeep(), ", Chebyshev degree ", param.cheb_degree(),
                    Mixed ? ", mixed-precision PV solver" : "");

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;
    Spinor start(commBase, "MDWF_dm_start"), ay(commBase, "MDWF_dm_ay"), gy(commBase, "MDWF_dm_gy");
    start.gauss(d_rand.state);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<Spinor>> vecs;
    MDWFThickRestartResult result;
    double hi = param.cheb_hi();
    const MDWFThickRestartParameters tp{param.nev(), param.nbasis(), param.nkeep(), param.tol(), param.max_restarts()};

    // A-specific part: the filter interval and the eigensolver.
    auto solveFor = [&](auto &a) {
        using OpA = std::remove_reference_t<decltype(a)>;
        if (param.cheb_degree() > 0 && !(hi > 0.0)) {
            if (dirac) {
                hi = 1.02;
            } else {
                MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, 1, OpA> adapter(a);
                const MDWFLanczosResult lz = mdwfLanczosExtremes(adapter, start, param.hi_lanczos_steps(),
                                                                 {param.hi_lanczos_steps()}, "MDWF_dm_hi");
                hi = 1.05 * lz.checkpoints.back().lambda_max;
            }
            rootLogger.info("MDWF modes: Chebyshev interval [", param.cheb_lo(), ", ", hi, "]");
        }
        MDWFChebyshevOperator<Spinor, OpA> op(a, param.cheb_lo(), hi, param.cheb_degree(), commBase, "MDWF_dm_cheb");
        result = mdwfThickRestartLanczos(op, start, tp, vecs, "MDWF_dm_trl");
    };
    if (dirac) {
        solveFor(*normal);
    } else {
        solveFor(kernel);
    }
    const double eigenSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const long eigenSolves = effective ? effective->solves() : 0;
    vecs.resize(param.nev());   // frees the rest of the Lanczos basis
    if (!effective && param.gw_defect() != 0) {
        makeEffective();
    }
    rootLogger.info("MDWF modes: eigensolver converged = ", result.converged, " after ", result.restarts,
                    " restarts, ", result.applications, " filtered-operator calls, ", eigenSolves, " PV solves, ",
                    eigenSeconds, " s");

    // Per-mode observables.
    const LatticeData lat = GInd::getLatData();
    LatticeContainer<true, double> reduction(commBase);
    reduction.adjustSize(lat.vol4);
    std::vector<MDWFModeRecord> modes;
    for (int l = 0; l < param.nev(); l++) {
        Spinor &y = *vecs[l];
        y.updateAll();
        if (dirac) {
            normal->apply(ay, y);
        } else {
            kernel.apply(ay, y);
        }
        MDWFModeRecord rec{};
        rec.lambda = real<double>(mdwfDot4(y, ay));
        ay.template axpyThisB<64>(-rec.lambda, y);
        rec.residual = std::sqrt(real<double>(mdwfDot4(ay, ay)));
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(gy, y, false);
        rec.chirality = real<double>(mdwfDot4(y, gy));
        reduction.template iterateOverBulk<All, HaloDepth>(MDWFModeDensitySquared<HaloDepth>(y));
        double sumRho2 = 0.0;
        reduction.reduce(sumRho2, lat.vol4);
        rec.ipr = static_cast<double>(lat.globvol4) * sumRho2;
        rec.gwDefect = -1.0;
        if (effective && param.gw_defect() != 0) {
            effective->applyEpsilon(gy, y);
            rec.gwDefect = 1.0 - real<double>(mdwfDot4(gy, gy));
        }
        const double m = param.mass();
        rec.root = dirac ? std::sqrt(std::max(rec.lambda - m * m, 0.0) / (1.0 - m * m))
                         : std::sqrt(std::max(rec.lambda, 0.0));
        rootLogger.info("MDWF mode ", l, ": lambda = ", rec.lambda, ", residual = ", rec.residual, ", chirality = ",
                        rec.chirality, ", IPR = ", rec.ipr, ", gw_defect = ", rec.gwDefect, ", sqrt = ", rec.root);
        modes.push_back(rec);
    }
    const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const std::string path = param.measurements_dir() + "/" + param.output_name();
    if (commBase.IamRoot()) {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("cannot open ", path));
        }
        out << "# mdwfDiracModes: " << param.GaugefileName() << "\n";
        out << "# spectrum " << param.spectrum() << " (" << (dirac ? "A = D(m)^dagger D(m), 4D effective MDWF operator"
                                                                   : "A = H_W^2 = (gamma5 D_W(-M5))^2")
            << "), kernel links " << kernelName << ", M5 " << param.M5() << ", c_sw " << param.csw() << ", b5 "
            << param.b5() << ", c5 " << param.b5() - 1.0 << ", Ls " << Ls << ", m " << param.mass()
            << ", antiperiodic_t " << param.antiperiodic_t() << "\n";
        out << "# eigensolver: thick-restart Lanczos, nev " << param.nev() << ", nbasis " << param.nbasis()
            << ", nkeep " << param.nkeep() << ", tol " << param.tol() << ", Chebyshev degree " << param.cheb_degree()
            << " on [" << param.cheb_lo() << ", " << hi << "], converged " << result.converged << ", restarts "
            << result.restarts << ", operator calls " << result.applications << "\n";
        out << "# cost: " << eigenSolves << " PV solves in the eigensolver, "
            << (effective ? effective->solves() : 0) << " in total, "
            << (effective ? effective->iterations() : 0) << " CG iterations, PV precision " << param.pv_precision()
            << ", " << eigenSeconds << " s eigensolver, " << totalSeconds << " s total\n";
        out << "# ipr = V sum_x rho(x)^2 (1: spread evenly, V = " << lat.globvol4 << ": one site); gw_defect = <y|1 - "
            << "eps^2|y> (-1: not computed); sqrt: " << (dirac ? "sqrt((lambda - m^2) / (1 - m^2))" : "sqrt(lambda)")
            << "\n";
        out << "# columns: index  lambda  residual  chirality  ipr  gw_defect  sqrt\n";
        out << std::setprecision(12);
        for (size_t l = 0; l < modes.size(); l++) {
            const MDWFModeRecord &r = modes[l];
            out << l << " " << r.lambda << " " << r.residual << " " << r.chirality << " " << r.ipr << " "
                << r.gwDefect << " " << r.root << "\n";
        }
    }
    rootLogger.info("MDWF modes written to ", path);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFDiracModesParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfDiracModes.param", argc, argv);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        if (GIndexer<All, HaloDepth>::getLatData().vol4 != GIndexer<All, HaloDepth>::getLatData().globvol4) {
            throw std::runtime_error(stdLogger.fatal("mdwfDiracModes is single-rank"));
        }
        const bool mixed = param.mixed_precision() != 0;
        if (param.ls() == 8) {
            mixed ? runMDWFDiracModes<8, true>(commBase, param) : runMDWFDiracModes<8, false>(commBase, param);
        } else if (param.ls() == 16) {
            mixed ? runMDWFDiracModes<16, true>(commBase, param) : runMDWFDiracModes<16, false>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfDiracModes supports ls = 8 or 16, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
