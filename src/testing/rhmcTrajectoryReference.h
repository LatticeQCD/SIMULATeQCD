#pragma once

#include "../modules/dslash/condensate.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Used by the existing rhmcTest in BOTH trees. No force/action implementation
// is copied. Each invocation reloads U and initializes a fresh trajectory RNG.
template<class floatT>
int rhmc_trajectory_reference(CommunicationBase &comm, RhmcParameters &param,
                              RationalCoeff &rat, const std::string &prefix,
                              bool save, double fieldTolerance,
                              double observableTolerance, const char *precision,
                              double condensateTolerance = -1.0, bool compareToDouble = false) {
    if (condensateTolerance < 0.0) condensateTolerance = observableTolerance;
    constexpr size_t halo = 2;
    initIndexer(4, param, comm);
    Gaugefield<floatT, true, halo, R18> gauge(comm);
    gauge.readconf_nersc(param.gauge_file());
    gauge.updateAll();
    gauge.su3latunitarize();
    grnd_state<true> trajectoryRng;
    initialize_rng(param.seed(), trajectoryRng);

    StopWatch<true> timer;
    int accepted;
    {
        rhmc<floatT, true, halo> HMC(param, rat, gauge, trajectoryRng.state);
        HMC.init_ratapprox();
        timer.start();
        accepted = HMC.update(false); // retain evolved U, not Metropolis rollback
        timer.stop();
    }
    rootLogger.info("RHMC reference trajectory time: ", sformat("%.6fs", timer.seconds()));
    rootLogger.info("RHMC reference update status: ", accepted,
                    " (Metropolis disabled for evolved-field comparison)");

    // Independent fixed measurement RNG: integration RNG consumption cannot
    // change the stochastic sources used for CHI_UD. Measurements are untimed.
    grnd_state<true> measurementRng;
    initialize_rng(424242, measurementRng);
    GaugeAction<floatT, true, halo, R18> action(gauge);
    PolyakovLoop<floatT, true, halo, R18> polyakov(gauge);
    const auto loop = polyakov.getPolyakovLoop();
    auto chi = measure_condensate<floatT, true, halo, 4, 10>(
        comm, param, static_cast<floatT>(param.m_ud()), gauge, measurementRng);
    std::vector<double> values{action.plaquette(), real(loop), imag(loop)};
    for (size_t i = 0; i < 10; ++i) values.push_back(chi[i]);
    rootLogger.info("Reference plaquette: ", sformatScientific(values[0]));
    rootLogger.info("Reference Polyakov loop: ", sformatScientific(loop));
    for (size_t i = 0; i < 10; ++i)
        rootLogger.info("Reference CHI_UD[", i, "]: ", sformatScientific(chi[i]));

    // Fail on a wrong reference case/settings, rather than silently comparing
    // unrelated trajectories. A global 8^3x4 lattice is enforced by the caller.
    std::ostringstream identity;
    identity << std::setprecision(17) << "RHMC_REFERENCE_V1 " << precision << ' '
             << param.mu_f() << ' ' << param.no_md() << ' ' << param.seed() << ' '
             << param.m_ud() << ' ' << param.m_s() << ' ' << param.beta() << ' '
             << param.step_size() << ' ' << param.no_step_sf() << ' '
             << param.no_sw() << ' ' << param.integrator() << ' '
             << param.residue() << ' ' << param.residue_force() << ' '
             << param.cgMax() << ' ' << param.residue_meas() << ' '
             << param.cgMax_meas() << ' ' << (compareToDouble ? false : param.cgMixedPrec()) << ' '
             << param.cgMixedPrec_delta() << ' ' << param.no_pf() << ' '
             << param.gauge_file() << ' ' << param.rat_file();
    bool pass = true;
    for (double value : values) pass = pass && std::isfinite(value);
    if (!pass) {
        rootLogger.error("RHMC trajectory reference: FAIL (nonfinite observable)");
        return 1;
    }
    if (save) {
        // Explicit save mode refuses to overwrite a frozen reference.
        std::ifstream existing(prefix + ".nersc"), existingMetadata(prefix + ".txt");
        if (existing || existingMetadata)
            throw std::runtime_error("Reference already exists; choose a new prefix");
        gauge.writeconf_nersc(prefix + ".nersc", 3, 2);
        std::ofstream output(prefix + ".txt");
        output << identity.str() << '\n' << std::setprecision(17);
        for (double value : values) output << value << '\n';
        output << timer.seconds() << '\n'; // informational, never compared
        output.close();
        if (!output) throw std::runtime_error("Cannot write reference metadata");
        rootLogger.info("RHMC trajectory reference: SAVED (not a correctness verdict)");
        return 0;
    }

    std::ifstream input(prefix + ".txt");
    std::string savedIdentity;
    if (!std::getline(input, savedIdentity) || savedIdentity != identity.str())
        throw std::runtime_error("Missing/incompatible reference metadata");
    bool gaugeObservablesPass = true, condensatePass = true;
    for (size_t i = 0; i < values.size(); ++i) {
        double expected;
        if (!(input >> expected) || !std::isfinite(expected))
            throw std::runtime_error("Invalid reference observable");
        const double difference = std::abs(values[i] - expected);
        rootLogger.info("RHMC observable[", i, "] reference=", sformat("%.17g", expected),
                        " computed=", sformat("%.17g", values[i]),
                        " absolute difference=", sformat("%.17g", difference));
        if (i < 3) gaugeObservablesPass = gaugeObservablesPass && difference <= observableTolerance;
        else condensatePass = condensatePass && difference <= condensateTolerance;
    }
    Gaugefield<floatT, true, halo, R18> reference(comm);
    reference.readconf_nersc(prefix + ".nersc");
    Gaugefield<floatT, false, halo, R18> computedHost(comm), referenceHost(comm);
    computedHost = gauge;
    referenceHost = reference;
    using GInd = GIndexer<All, halo>;
    const auto local = GInd::getLatData();
    double maxDifference = 0.0;
    for (size_t t = 0; t < local.lt; ++t)
    for (size_t z = 0; z < local.lz; ++z)
    for (size_t y = 0; y < local.ly; ++y)
    for (size_t x = 0; x < local.lx; ++x)
    for (size_t mu = 0; mu < 4; ++mu) {
        const auto siteMu = GInd::getSiteMu(GInd::getSite(x, y, z, t), mu);
        const auto a = computedHost.getAccessor().template getLink<double>(siteMu);
        const auto b = referenceHost.getAccessor().template getLink<double>(siteMu);
        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double re = std::abs(real(a(i, j)) - real(b(i, j)));
            const double im = std::abs(imag(a(i, j)) - imag(b(i, j)));
            pass = pass && std::isfinite(re) && std::isfinite(im);
            if (re > maxDifference) maxDifference = re;
            if (im > maxDifference) maxDifference = im;
        }
    }
    maxDifference = comm.globalMaximum(maxDifference);
    pass = comm.globalMinimum(pass ? 1.0 : 0.0) > 0.5;
    rootLogger.info("RHMC full-field max absolute component difference: ",
                    sformatScientific(maxDifference));
    rootLogger.info("RHMC field tolerance: ", fieldTolerance,
                    "; gauge observable absolute tolerance: ", observableTolerance,
                    "; CHI_UD absolute tolerance: ", condensateTolerance);
    // Three-row 64-bit output preserves every float value exactly and avoids
    // reconstructing the third row. Checking never writes into frozen inputs.
    const bool fieldPass = compare_fields<floatT, halo, true, R18>(gauge, reference, fieldTolerance) && pass;
    rootLogger.info("RHMC full gauge field: ", fieldPass ? "PASS" : "FAIL");
    rootLogger.info("RHMC plaquette and Polyakov loop: ", gaugeObservablesPass ? "PASS" : "FAIL");
    rootLogger.info("RHMC fixed-source CHI_UD: ", condensatePass ? "PASS" : "FAIL");
    pass = fieldPass && gaugeObservablesPass && condensatePass;
    rootLogger.info("RHMC trajectory reference ", precision, " mu0=", param.mu_f(),
                    " no_md=", param.no_md(), ": ", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Collaborator-facing policy. These are practical numerical-regression
// envelopes, informed by jobs 69031/69032/69154, NOT exact-equality thresholds
// or a physics proof. Strict author mode remains available below.
inline int rhmc_trajectory_regression_suite(CommunicationBase &comm, int argc, char **argv) {
    try {
        if (argc < 4)
            throw std::runtime_error("Usage: rhmcTest --trajectory-regression-suite PARAM REFERENCE_DIRECTORY "
                                     "[--double-only] [Key=value ...]");
        // Key=value arguments override the parameter file, e.g. 'Nodes=1 1 1 2'.
        bool doubleOnly = false;
        std::vector<char *> readArgs{argv[0], argv[2]};
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--double-only") doubleOnly = true;
            else readArgs.push_back(argv[i]);
        }
        RhmcParameters param;
        param.readfile(comm, argv[2], static_cast<int>(readArgs.size()), readArgs.data());
        comm.init(param.nodeDim());
        initIndexer(4, param, comm);
        const auto lat = GIndexer<All, 2>::getLatData();
        if (lat.globLX != 8 || lat.globLY != 8 || lat.globLZ != 8 || lat.globLT != 4 || param.cgMixedPrec())
            throw std::runtime_error("Regression fixtures require a global 8x8x8x4 lattice and cgMixedPrec=0");
        RationalCoeff rat;
        rat.readfile(comm, param.rat_file());
        rat.check_rat(param);
        rootLogger.info("RHMC regression: frozen legacy 1732861, numerical tolerances; timing is informational");
        bool pass = true;
        const std::vector<std::string> precisions = doubleOnly
            ? std::vector<std::string>{"double"} : std::vector<std::string>{"float", "double"};
        for (const std::string precision : precisions)
        for (const std::string mu : {std::string("0"), std::string("0.4")})
        for (const int steps : {1, 10}) {
            param.mu_f.set(std::stod(mu));
            param.no_md.set(steps);
            const std::string prefix = std::string(argv[3]) + "/" + precision + "_mu" + mu + "_md" + std::to_string(steps);
            int status;
            if (precision == "float")
                status = rhmc_trajectory_reference<float>(comm, param, rat, prefix, false,
                    steps == 1 ? 1e-6 : 1e-5, 1e-7, "float", 1e-5);
            else
                status = rhmc_trajectory_reference<double>(comm, param, rat, prefix, false,
                    steps == 1 ? 1e-10 : 1e-9, 1e-11, "double", 1e-5);
            pass = status == 0 && pass;
        }
        rootLogger.info("RHMC legacy-reference regression suite: ", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception &error) {
        rootLogger.error("RHMC legacy-reference regression suite: FAIL: ", error.what());
        return 1;
    }
}

inline int rhmc_trajectory_reference_main(CommunicationBase &comm, int argc, char **argv) {
    try {
        const bool save = std::string(argv[1]) == "--save-trajectory-reference";
        // Custom options precede the usual parameter file/override arguments.
        const int firstParameter = save ? 4 : 6;
        if (argc <= firstParameter)
            throw std::runtime_error("Usage: rhmcTest --save-trajectory-reference PREFIX {float|double} PARAM [overrides], or --check-trajectory-reference PREFIX {float|double} FIELD_TOL OBS_ABS_TOL PARAM [overrides] [--chi-tolerance=CHI_ABS_TOL]");
        const std::string precision(argv[3]);
        if (precision != "float" && precision != "double")
            throw std::runtime_error("Precision must be float or double");
        const double fieldTolerance = save ? 0 : std::stod(argv[4]);
        const double observableTolerance = save ? 0 : std::stod(argv[5]);
        if (!save && (!(fieldTolerance > 0) || !std::isfinite(fieldTolerance) ||
                      !(observableTolerance > 0) || !std::isfinite(observableTolerance)))
            throw std::runtime_error("Supply positive finite calibrated tolerances");
        // CHI_UD comes from CG solves converged to residue_meas, so it can need a looser
        // tolerance than the plaquette; by default it uses OBS_ABS_TOL.
        const std::string chiFlag = "--chi-tolerance=";
        double condensateTolerance = -1.0;
        // --compare-to-double: check a cgMixedPrec=1 run against a reference saved in double.
        bool compareToDouble = false;
        std::vector<char *> parameterArgs{argv[0]};
        for (int i = firstParameter; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (!save && i > firstParameter && arg == "--compare-to-double") {
                compareToDouble = true;
            } else if (!save && i > firstParameter && arg.rfind(chiFlag, 0) == 0) {
                condensateTolerance = std::stod(arg.substr(chiFlag.size()));
                if (!(condensateTolerance > 0) || !std::isfinite(condensateTolerance))
                    throw std::runtime_error("Supply a positive finite --chi-tolerance");
            } else {
                parameterArgs.push_back(argv[i]);
            }
        }
        RhmcParameters param;
        param.readfile(comm, argv[firstParameter], parameterArgs.size(), parameterArgs.data());
        comm.init(param.nodeDim());
        initIndexer(4, param, comm);
        // Any lattice and Nodes layout: a reference saved on one rank can check a
        // multi-rank run of the same lattice (the RNG is seeded over the global volume).
        if (param.cgMixedPrec() && !compareToDouble)
            throw std::runtime_error("Reference cases require cgMixedPrec=0 (or --compare-to-double)");
        RationalCoeff rat;
        rat.readfile(comm, param.rat_file());
        rat.check_rat(param);
        if (precision == "float")
            return rhmc_trajectory_reference<float>(comm, param, rat, argv[2], save,
                fieldTolerance, observableTolerance, "float", condensateTolerance, compareToDouble);
        return rhmc_trajectory_reference<double>(comm, param, rat, argv[2], save,
            fieldTolerance, observableTolerance, "double", condensateTolerance, compareToDouble);
    } catch (const std::exception &error) {
        rootLogger.error("RHMC trajectory reference: FAIL: ", error.what());
        return 1;
    }
}
