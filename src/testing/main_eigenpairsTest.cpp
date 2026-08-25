#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

struct LanczosTestConfiguration {
    int numEigenvectors = 5;
    int krylovDimension = 256;
    int thickRestartDimension = 80;
    int maximumRestarts = 10;
    double residualTolerance = 1.0e-6;
    TRLanConvergenceCriterion convergenceCriterion =
            TRLanConvergenceCriterion::MaximumScaledPerMode;
};

int readEnvironmentInteger(const char *name, const int defaultValue) {
    const char *rawValue = std::getenv(name);
    if (rawValue == nullptr) {
        return defaultValue;
    }

    const std::string value(rawValue);
    int parsed = 0;
    const std::from_chars_result result =
            std::from_chars(
                    value.data(),
                    value.data() + value.size(),
                    parsed);
    if (value.empty()
        || result.ec != std::errc()
        || result.ptr != value.data() + value.size()) {
        throw std::runtime_error(
                std::string(name)
                + " must be a base-10 integer representable as int");
    }
    return parsed;
}

double readEnvironmentDouble(
        const char *name,
        const double defaultValue) {
    const char *rawValue = std::getenv(name);
    if (rawValue == nullptr) {
        return defaultValue;
    }

    const std::string value(rawValue);
    if (value.empty()) {
        throw std::runtime_error(
                std::string(name)
                + " must be a finite floating-point value");
    }
    for (const unsigned char character : value) {
        if (std::isspace(character)) {
            throw std::runtime_error(
                    std::string(name)
                    + " must not contain whitespace");
        }
    }

    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno == ERANGE
        || end == value.c_str()
        || end != value.c_str() + value.size()
        || !std::isfinite(parsed)) {
        throw std::runtime_error(
                std::string(name)
                + " must be a finite floating-point value");
    }
    return parsed;
}

TRLanConvergenceCriterion readEnvironmentConvergenceCriterion() {
    const char *rawValue =
            std::getenv(
                    "SIMQCD_LANCZOS_CONVERGENCE_CRITERION");
    if (rawValue == nullptr) {
        return TRLanConvergenceCriterion::
                MaximumScaledPerMode;
    }

    const std::string value(rawValue);
    if (value == "maximum_scaled_per_mode") {
        return TRLanConvergenceCriterion::
                MaximumScaledPerMode;
    }
    if (value == "projected_physical_aggregate") {
        return TRLanConvergenceCriterion::
                ProjectedPhysicalAggregate;
    }
    if (value == "direct_physical_aggregate"
        || value == "densecode_aggregate_physical") {
        return TRLanConvergenceCriterion::
                DirectPhysicalAggregate;
    }
    throw std::runtime_error(
            "SIMQCD_LANCZOS_CONVERGENCE_CRITERION must be "
            "'maximum_scaled_per_mode', "
            "'projected_physical_aggregate', or "
            "'direct_physical_aggregate' "
            "('densecode_aggregate_physical' is accepted as an alias "
            "for 'direct_physical_aggregate')");
}

LanczosTestConfiguration readLanczosTestConfiguration() {
    LanczosTestConfiguration configuration;
    configuration.numEigenvectors =
            readEnvironmentInteger(
                    "SIMQCD_NUM_EIGENVECTORS",
                    configuration.numEigenvectors);
    configuration.krylovDimension =
            readEnvironmentInteger(
                    "SIMQCD_KRYLOV_DIM",
                    configuration.krylovDimension);
    configuration.thickRestartDimension =
            readEnvironmentInteger(
                    "SIMQCD_THICK_RESTART_DIM",
                    configuration.thickRestartDimension);
    configuration.maximumRestarts =
            readEnvironmentInteger(
                    "SIMQCD_MAX_RESTARTS",
                    configuration.maximumRestarts);
    configuration.residualTolerance =
            readEnvironmentDouble(
                    "SIMQCD_LANCZOS_RESIDUAL_TOL",
                    configuration.residualTolerance);
    configuration.convergenceCriterion =
            readEnvironmentConvergenceCriterion();
    return configuration;
}

void validateLanczosTestConfiguration(
        const LanczosTestConfiguration &configuration) {
    if (configuration.numEigenvectors <= 0) {
        throw std::runtime_error(
                "SIMQCD_NUM_EIGENVECTORS must be positive");
    }
    if (configuration.krylovDimension <= 0) {
        throw std::runtime_error(
                "SIMQCD_KRYLOV_DIM must be positive");
    }
    if (configuration.thickRestartDimension <= 0) {
        throw std::runtime_error(
                "SIMQCD_THICK_RESTART_DIM must be positive");
    }
    if (configuration.numEigenvectors
        > configuration.thickRestartDimension) {
        throw std::runtime_error(
                "SIMQCD_NUM_EIGENVECTORS must not exceed "
                "SIMQCD_THICK_RESTART_DIM");
    }
    if (configuration.thickRestartDimension
        >= configuration.krylovDimension) {
        throw std::runtime_error(
                "SIMQCD_THICK_RESTART_DIM must be smaller than "
                "SIMQCD_KRYLOV_DIM");
    }
    if (configuration.krylovDimension
                - configuration.thickRestartDimension
        < 2) {
        throw std::runtime_error(
                "SIMQCD_KRYLOV_DIM must leave at least two "
                "non-retained Lanczos slots");
    }
    if (configuration.maximumRestarts < 0) {
        throw std::runtime_error(
                "SIMQCD_MAX_RESTARTS must be non-negative");
    }
    if (!std::isfinite(configuration.residualTolerance)
        || configuration.residualTolerance <= 0.0) {
        throw std::runtime_error(
                "SIMQCD_LANCZOS_RESIDUAL_TOL must be positive "
                "and finite");
    }
}

void printLanczosTestConfiguration(
        CommunicationBase &commBase,
        const LanczosTestConfiguration &configuration) {
    if (!commBase.IamRoot()) {
        return;
    }
    std::cout << std::setprecision(17);
    std::cout << "SIMQCD_LANCZOS_CONFIG num_eigenvectors = "
              << configuration.numEigenvectors << "\n";
    std::cout << "SIMQCD_LANCZOS_CONFIG krylov_dimension = "
              << configuration.krylovDimension << "\n";
    std::cout << "SIMQCD_LANCZOS_CONFIG thick_restart_dimension = "
              << configuration.thickRestartDimension << "\n";
    std::cout << "SIMQCD_LANCZOS_CONFIG maximum_restarts = "
              << configuration.maximumRestarts << "\n";
    std::cout << "SIMQCD_LANCZOS_CONFIG residual_tolerance = "
              << configuration.residualTolerance << "\n";
    std::cout << "SIMQCD_LANCZOS_CONFIG convergence_criterion = "
              << trlanConvergenceCriterionName(
                         configuration.convergenceCriterion)
              << "\n";
    std::cout.flush();
}

struct BenchmarkTiming {
    double localSeconds;
    double maximumSeconds;
};

bool deflationBenchmarkEnabled() {
    const char *value = std::getenv("SIMQCD_RUN_DEFLATION_BENCHMARK");
    return value != nullptr && std::string(value) == "1";
}

void synchronizeBenchmarkBackend() {
    const gpuError_t gpuError = gpuDeviceSynchronize();
    if (gpuError != gpuSuccess) {
        GpuError("Deflation benchmark backend synchronization failed", gpuError);
    }
}

template<class Callable>
BenchmarkTiming timeBenchmarkRegion(CommunicationBase &commBase, Callable &&callable) {
    synchronizeBenchmarkBackend();
    commBase.globalBarrier();

    StopWatch<false> timer;
    timer.start();
    callable();
    synchronizeBenchmarkBackend();
    const double localSeconds = timer.stop() / 1000.0;

    return {localSeconds, commBase.globalMaximum(localSeconds)};
}

double safeRatio(const double numerator, const double denominator) {
    if (denominator > 0.0) {
        return numerator / denominator;
    }
    return numerator == 0.0 ? 1.0 : std::numeric_limits<double>::infinity();
}

void requireBenchmarkCondition(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error("Deflation benchmark validation failed: " + message);
    }
}

template<class Operator, class Spinor>
double calculateRelativeResidual(Operator &dslash, const Spinor &solution, Spinor &rhs) {
    Spinor applied(rhs.getComm());
    Spinor residual(rhs.getComm());

    dslash.applyMdaggM(applied, solution, true);
    residual = applied - rhs;

    const double rhsNormSquared = rhs.realdotProduct(rhs);
    const double residualNormSquared = residual.realdotProduct(residual);
    requireBenchmarkCondition(std::isfinite(rhsNormSquared) && rhsNormSquared > 0.0,
                              "right-hand-side norm is not finite and positive");
    requireBenchmarkCondition(std::isfinite(residualNormSquared) && residualNormSquared >= 0.0,
                              "residual norm is not finite and non-negative");
    return std::sqrt(residualNormSquared / rhsNormSquared);
}

template<class Spinor>
double calculateRelativeDifference(Spinor &reference, const Spinor &candidate) {
    Spinor difference(reference.getComm());
    difference = reference - candidate;

    const double referenceNormSquared = reference.realdotProduct(reference);
    const double differenceNormSquared = difference.realdotProduct(difference);
    requireBenchmarkCondition(std::isfinite(referenceNormSquared) && referenceNormSquared > 0.0,
                              "ordinary solution norm is not finite and positive");
    requireBenchmarkCondition(std::isfinite(differenceNormSquared) && differenceNormSquared >= 0.0,
                              "solution-difference norm is not finite and non-negative");
    return std::sqrt(differenceNormSquared / referenceNormSquared);
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void runDeflationBenchmark(
        CommunicationBase &commBase,
        Gaugefield<floatT, true, HaloDepthGauge, R18> &gaugeSmeared,
        Gaugefield<floatT, true, HaloDepthGauge, U3R14> &gaugeNaik,
        const Eigenpairs<floatT, true, Even, HaloDepthGauge, HaloDepthSpin, NStacks> &eigenpairs,
        const int expectedEigenpairs, const double mass, const floatT naikEpsilon,
        const int maximumIterations, const double solveTolerance,
        const double eigenpairResidualTolerance, const unsigned int rhsSeed) {
    using Spinor = Spinorfield<floatT, true, Even, HaloDepthSpin, NStacks>;
    using Dslash = HisqDSlash<floatT, true, Even, HaloDepthGauge, HaloDepthSpin, NStacks>;

    requireBenchmarkCondition(NStacks == 1, "benchmark requires exactly one right-hand side");
    requireBenchmarkCondition(eigenpairs.SpinorCount() == expectedEigenpairs,
                              "required eigenpair count is not present");
    requireBenchmarkCondition(std::isfinite(mass) && mass > 0.0, "mass is not finite and positive");
    requireBenchmarkCondition(std::isfinite(static_cast<double>(naikEpsilon)),
                              "Naik epsilon is not finite");
    requireBenchmarkCondition(maximumIterations > 0, "maximum iteration count is not positive");
    requireBenchmarkCondition(std::isfinite(solveTolerance) && solveTolerance > 0.0,
                              "solve tolerance is not finite and positive");
    requireBenchmarkCondition(std::isfinite(eigenpairResidualTolerance)
                                      && eigenpairResidualTolerance > 0.0,
                              "eigenpair validation tolerance is not finite and positive");

    for (int index = 0; index < eigenpairs.SpinorCount(); ++index) {
        const double lambda = eigenpairs.getEigenValue(index);
        requireBenchmarkCondition(std::isfinite(lambda) && lambda >= 0.0,
                                  "eigenvalue is not finite and non-negative");
        requireBenchmarkCondition(std::isfinite(mass * mass + lambda) && mass * mass + lambda > 0.0,
                                  "deflation denominator is not finite and positive");
    }

    Dslash massiveDslash(gaugeSmeared, gaugeNaik, mass, naikEpsilon);
    ConjugateGradient<floatT, NStacks> cg;

    grnd_state<false> hostRandom;
    grnd_state<true> deviceRandom;
    hostRandom.make_rng_state(rhsSeed);
    deviceRandom = hostRandom;

    Spinor rhs(commBase);
    Spinor ordinaryRhs(commBase);
    Spinor deflatedRhs(commBase);
    rhs.gauss(deviceRandom.state);
    rhs.updateAll();
    ordinaryRhs = rhs;
    deflatedRhs = rhs;
    ordinaryRhs.updateAll();
    deflatedRhs.updateAll();

    Spinor ordinarySolution(commBase);
    Spinor deflatedSolution(commBase);
    CGSolveResult ordinaryResult;
    CGSolveResult deflatedResult;

    const BenchmarkTiming ordinaryTiming = timeBenchmarkRegion(commBase, [&] {
        // invert_new compares ||r||^2/||b||^2, so square the requested norm tolerance.
        cg.invert_new(massiveDslash, ordinarySolution, ordinaryRhs, maximumIterations,
                      solveTolerance * solveTolerance, &ordinaryResult);
    });

    double eigenpairMaximumResidual = 0.0;
    const BenchmarkTiming eigenpairValidationTiming = timeBenchmarkRegion(commBase, [&] {
        cg.template checkEigenValueEquation<true, Even, HaloDepthGauge, HaloDepthSpin>(
                mass, massiveDslash, eigenpairs, &eigenpairMaximumResidual);
    });

    const BenchmarkTiming startVectorTiming = timeBenchmarkRegion(commBase, [&] {
        cg.template startVector<true, Even, HaloDepthGauge, HaloDepthSpin>(
                mass, deflatedSolution, deflatedRhs, eigenpairs);
    });

    const BenchmarkTiming startVectorTesterTiming = timeBenchmarkRegion(commBase, [&] {
        cg.template startVectorTester<true, Even, HaloDepthGauge, HaloDepthSpin>(
                mass, massiveDslash, deflatedSolution, deflatedRhs, eigenpairs);
    });

    const BenchmarkTiming deflatedCgTiming = timeBenchmarkRegion(commBase, [&] {
        cg.invert_deflation(massiveDslash, deflatedSolution, deflatedRhs, maximumIterations,
                            solveTolerance, &deflatedResult, true);
    });

    const double localDeflatedTotal =
            eigenpairValidationTiming.localSeconds
            + startVectorTiming.localSeconds
            + startVectorTesterTiming.localSeconds
            + deflatedCgTiming.localSeconds;
    const double deflatedTotalSeconds = commBase.globalMaximum(localDeflatedTotal);

    const double ordinaryFinalResidual =
            calculateRelativeResidual(massiveDslash, ordinarySolution, ordinaryRhs);
    const double deflatedFinalResidual =
            calculateRelativeResidual(massiveDslash, deflatedSolution, deflatedRhs);
    const double solutionRelativeDifference =
            calculateRelativeDifference(ordinarySolution, deflatedSolution);

    const double explicitResidualCertificationFactor = 1.01;
    const double explicitResidualCertificationTolerance =
            explicitResidualCertificationFactor * solveTolerance;
    const bool ordinaryExplicitResidualStrictPass =
            std::isfinite(ordinaryFinalResidual)
            && ordinaryFinalResidual <= solveTolerance;
    const bool deflatedExplicitResidualStrictPass =
            std::isfinite(deflatedFinalResidual)
            && deflatedFinalResidual <= solveTolerance;
    const bool ordinaryExplicitResidualCertified =
            std::isfinite(ordinaryFinalResidual)
            && ordinaryFinalResidual <= explicitResidualCertificationTolerance;
    const bool deflatedExplicitResidualCertified =
            std::isfinite(deflatedFinalResidual)
            && deflatedFinalResidual <= explicitResidualCertificationTolerance;

    const int iterationReduction = ordinaryResult.iterations - deflatedResult.iterations;
    const double iterationSpeedup =
            safeRatio(static_cast<double>(ordinaryResult.iterations),
                      static_cast<double>(deflatedResult.iterations));
    const double cgTimeSpeedup =
            safeRatio(ordinaryTiming.maximumSeconds, deflatedCgTiming.maximumSeconds);
    const double totalTimeSpeedup =
            safeRatio(ordinaryTiming.maximumSeconds, deflatedTotalSeconds);

    if (commBase.IamRoot()) {
        std::cout << std::setprecision(17);
        std::cout << "SIMQCD_DEFLATION_BENCHMARK eigenpair_count = " << eigenpairs.SpinorCount() << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK rhs_seed = " << rhsSeed << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK mass = " << mass << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK naik_epsilon = " << naikEpsilon << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK requested_tolerance = " << solveTolerance << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK configured_squared_residue = "
                  << solveTolerance * solveTolerance << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK explicit_residual_certification_factor = "
                  << explicitResidualCertificationFactor << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK explicit_residual_certification_tolerance = "
                  << explicitResidualCertificationTolerance << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK maximum_iterations = " << maximumIterations << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_iterations = " << ordinaryResult.iterations << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_iterations = " << deflatedResult.iterations << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_recursive_residual = "
                  << ordinaryResult.recursiveRelativeResidual << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_recursive_residual = "
                  << deflatedResult.recursiveRelativeResidual << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_solve_seconds = "
                  << ordinaryTiming.maximumSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK eigenpair_validation_seconds = "
                  << eigenpairValidationTiming.maximumSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK start_vector_seconds = "
                  << startVectorTiming.maximumSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK start_vector_tester_seconds = "
                  << startVectorTesterTiming.maximumSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_cg_seconds = "
                  << deflatedCgTiming.maximumSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_total_seconds = "
                  << deflatedTotalSeconds << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK eigenpair_maximum_residual = "
                  << eigenpairMaximumResidual << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_final_residual = "
                  << ordinaryFinalResidual << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_final_residual = "
                  << deflatedFinalResidual << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK solution_relative_difference = "
                  << solutionRelativeDifference << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_explicit_residual_strict_pass = "
                  << ordinaryExplicitResidualStrictPass << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_explicit_residual_strict_pass = "
                  << deflatedExplicitResidualStrictPass << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK ordinary_explicit_residual_certified = "
                  << ordinaryExplicitResidualCertified << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK deflated_explicit_residual_certified = "
                  << deflatedExplicitResidualCertified << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK iteration_reduction = "
                  << iterationReduction << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK iteration_speedup = "
                  << iterationSpeedup << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK cg_time_speedup = "
                  << cgTimeSpeedup << "\n";
        std::cout << "SIMQCD_DEFLATION_BENCHMARK total_time_speedup = "
                  << totalTimeSpeedup << "\n";
        std::cout.flush();
    }

    requireBenchmarkCondition(ordinaryResult.converged, "ordinary CG did not reach its recurrence tolerance");
    requireBenchmarkCondition(deflatedResult.converged, "deflated CG did not reach its recurrence tolerance");
    requireBenchmarkCondition(std::isfinite(eigenpairMaximumResidual)
                                      && eigenpairMaximumResidual <= eigenpairResidualTolerance,
                              "eigenpair equation residual exceeds its validation tolerance");
    requireBenchmarkCondition(ordinaryExplicitResidualCertified,
                              "ordinary explicit final residual exceeds the 1 percent certification margin");
    requireBenchmarkCondition(deflatedExplicitResidualCertified,
                              "deflated explicit final residual exceeds the 1 percent certification margin");
    const double solutionAgreementTolerance = std::max(100.0 * solveTolerance, 1.0e-4);
    requireBenchmarkCondition(std::isfinite(solutionRelativeDifference)
                                      && solutionRelativeDifference <= solutionAgreementTolerance,
                              "ordinary and deflated solutions do not agree");
}

} // namespace

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);
    
    commBase.init(param.nodeDim());

    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1;
    typedef float floatT; // Define the precision here
    constexpr floatT naikEpsilon = 0.0;

    const LanczosTestConfiguration lanczosConfiguration =
            readLanczosTestConfiguration();
    validateLanczosTestConfiguration(lanczosConfiguration);
    printLanczosTestConfiguration(
            commBase, lanczosConfiguration);
    const int numVec =
            lanczosConfiguration.numEigenvectors;

    initIndexer(HaloDepthGauge, param, commBase);

    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge(commBase);      /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge_smeared(commBase);
    Gaugefield<floatT,true,HaloDepthGauge,U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, true, HaloDepthGauge, R18, R18, R18, U3R14> smearing(
            gauge, gauge_smeared, gauge_Naik, naikEpsilon);
    smearing.SmearAll();

    HisqDSlash<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> dslash(
            gauge_smeared, gauge_Naik, 0.0, naikEpsilon);
    
    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsWrite(commBase);
    TRLanRestartParams lanczosParams;
    lanczosParams.krylovDim =
            lanczosConfiguration.krylovDimension;
    lanczosParams.thickRestartDim =
            lanczosConfiguration.thickRestartDimension;
    lanczosParams.maxRestarts =
            lanczosConfiguration.maximumRestarts;
    lanczosParams.residualTol =
            lanczosConfiguration.residualTolerance;
    lanczosParams.convergenceCriterion =
            lanczosConfiguration.convergenceCriterion;
    lanczosParams.breakdownTol = 1e-12;
    lanczosParams.seed = 1234;
    lanczosParams.reorthogonalizationPasses = 2;
    lanczosParams.physicalCheckInterval = 5;
    lanczosParams.failOnNoConvergence = false;
    lanczosParams.convergenceDiagnostics = true;
    lanczosParams.mpiConsistencyDiagnostics = true;

    lanczosParams.chebyshev.enabled = false;
    lanczosParams.exponential.enabled = true;
    lanczosParams.exponential.order = 26;
    lanczosParams.exponential.alpha = 9.0;
    lanczosParams.exponential.beta = 1.0;
    lanczosParams.exponential.operatorShift = 0.0;
    lanczosParams.exponential.operatorScale = 1.0;

    eigenpairsWrite.lanczos(dslash, numVec, lanczosParams);

    if (commBase.IamRoot()) {
        std::ofstream evout("simqcd_eigenvalues.txt");
        evout << std::setprecision(17);

        for (int idx = 0; idx < eigenpairsWrite.SpinorCount(); idx++) {
            const double lambda = eigenpairsWrite.getEigenValue(idx);
            rootLogger.info("SIMQCD_EIGENVALUE ", idx, " = ", lambda);
            evout << idx << " " << lambda << "\n";
        }
    }
    eigenpairsWrite.writeEigenpairsToFile("testEigenpairsFile", 0, ENDIAN_AUTO);

    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsRead(commBase);
    eigenpairsRead.readEigenpairsFromFile("testEigenpairsFile");
    eigenpairsRead.updateAll();
    
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorWrite(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorRead(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorDiff(commBase);

    for (int idx = 0; idx < eigenpairsRead.SpinorCount(); idx++) {
        eigenpairsWrite.getEigenSpinor(spinorWrite, idx);
        eigenpairsRead.getEigenSpinor(spinorRead, idx);

        spinorDiff = spinorWrite - spinorRead;
        const double spinorDiffNorm =
                spinorDiff.realdotProduct(spinorDiff);
        if (spinorDiffNorm > 1e-10) {
            rootLogger.warn("Eigenpair with index ", idx, " differs between written and read version! Norm of difference: ", spinorDiffNorm);
        } else {
            rootLogger.info("Eigenpair with index ", idx, " matches between written and read version. Norm of difference: ", spinorDiffNorm);
        }
    }

    for (int idx = 0; idx < eigenpairsRead.SpinorCount(); idx++) {
        double lambdaWrite = eigenpairsWrite.getEigenValue(idx);
        double lambdaRead = eigenpairsRead.getEigenValue(idx);

        double lambdaDiff = lambdaWrite - lambdaRead;
        if (std::abs(lambdaDiff) > 1e-10) {
            rootLogger.warn("Eigenvalue with index ", idx, " differs between written and read version! Difference: ", lambdaDiff);
        } else {
            rootLogger.info("Eigenvalue with index ", idx, " matches between written and read version. Difference: ", lambdaDiff);
        }
    }

    if (deflationBenchmarkEnabled()) {
        if (param.valence_masses.numberValues() == 0) {
            throw std::runtime_error("Deflation benchmark requires at least one valence mass");
        }
        const double eigenpairResidualTolerance =
                std::max(10.0 * lanczosParams.residualTol, 1.0e-5);
        if (!std::isfinite(param.residue()) || param.residue() <= 0.0) {
            throw std::runtime_error("Deflation benchmark requires a finite positive residue");
        }
        runDeflationBenchmark(
                commBase, gauge_smeared, gauge_Naik, eigenpairsRead, numVec,
                param.valence_masses[0], naikEpsilon, param.cgMax(), std::sqrt(param.residue()),
                eigenpairResidualTolerance, 5678);
    }
}
