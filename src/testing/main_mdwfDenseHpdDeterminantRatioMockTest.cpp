/*
 * Tiny complex noncommuting matrices only; no physical MDWF mass mapping.
 * C = (Ma Mb^-1)^dagger (Ma Mb^-1), det(C) = det(Na)/det(Nb).
 * Exact 2x2 spectral powers test phi = C^(p/2) xi and S = phi^dagger C^-p phi.
 * These private helpers are not production inverse, eigensolver, or RHMC APIs.
 */
#include "../simulateqcd.h"
#ifdef MDWF_DENSE_RATIONAL_MOCK_TEST
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using Complex = std::complex<double>;
using Matrix = std::array<std::array<Complex, 2>, 2>;
using Vector = std::array<Complex, 2>;

void require(bool valid, const std::string &label) {
    if (!valid) throw std::runtime_error("MDWF dense HPD ratio mock: " + label);
}

Matrix identity() {
    Matrix result{};
    result[0][0] = result[1][1] = 1.0;
    return result;
}

Matrix dagger(const Matrix &a) {
    Matrix result{};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) result[i][j] = std::conj(a[j][i]);
    return result;
}

Matrix multiply(const Matrix &a, const Matrix &b) {
    Matrix result{};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = 0; k < 2; ++k) result[i][j] += a[i][k] * b[k][j];
    return result;
}

Vector applyMatrix(const Matrix &a, const Vector &v) {
    Vector result{};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) result[i] += a[i][j] * v[j];
    return result;
}

Matrix subtract(const Matrix &a, const Matrix &b) {
    Matrix result{};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) result[i][j] = a[i][j] - b[i][j];
    return result;
}

double norm(const Matrix &a) {
    double sum = 0.0;
    for (const auto &row : a) for (const auto &entry : row) sum += std::norm(entry);
    return std::sqrt(sum);
}

double normSquared(const Vector &v) {
    return std::norm(v[0]) + std::norm(v[1]);
}

Complex inner(const Vector &a, const Vector &b) {
    return std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
}

Complex determinant(const Matrix &a) {
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}

double checkClose(double actual, double expected, const std::string &label) {
    require(std::isfinite(actual) && std::isfinite(expected), label + " nonfinite");
    const double difference = std::abs(actual - expected);
    require(difference <= 2e-12 * std::max(1.0, std::abs(expected)), label + " differs");
    return difference;
}

double checkMatrix(const Matrix &actual, const Matrix &expected, const std::string &label) {
    const double difference = norm(subtract(actual, expected));
    require(std::isfinite(difference) && std::isfinite(norm(expected)), label + " nonfinite");
    require(difference <= 2e-12 * std::max(1.0, norm(expected)), label + " differs");
    return difference;
}

Matrix inverse(const Matrix &a) {
    const Complex d = determinant(a);
    if (!std::isfinite(std::abs(d)) || std::abs(d) <= 1e-12)
        throw std::invalid_argument("singular/nonfinite tiny mock matrix");
    Matrix result{};
    result[0][0] = a[1][1] / d;
    result[0][1] = -a[0][1] / d;
    result[1][0] = -a[1][0] / d;
    result[1][1] = a[0][0] / d;
    return result;
}

std::array<double, 2> positiveSpectrum(const Matrix &a) {
    if (!std::isfinite(norm(a)) || norm(subtract(a, dagger(a))) > 1e-12 * std::max(1.0, norm(a)))
        throw std::invalid_argument("spectral powers require a finite Hermitian mock matrix");
    const double midpoint = (a[0][0].real() + a[1][1].real()) / 2.0;
    const double radius = std::hypot((a[0][0].real() - a[1][1].real()) / 2.0,
                                     std::abs(a[0][1]));
    const std::array<double, 2> eigenvalues{{midpoint - radius, midpoint + radius}};
    if (!(eigenvalues[0] > 1e-12) || !std::isfinite(eigenvalues[1]))
        throw std::invalid_argument("spectral powers require positive mock eigenvalues");
    return eigenvalues;
}

Matrix positivePower(const Matrix &a, double power) {
    if (!std::isfinite(power)) throw std::invalid_argument("nonfinite mock exponent");
    const auto spectrum = positiveSpectrum(a);
    const double gap = spectrum[1] - spectrum[0];
    Matrix result{};
    if (gap <= 1e-13 * std::max(1.0, spectrum[1])) {
        // Degenerate scalar case, including equal-operator C = I.
        const double value = std::pow((spectrum[0] + spectrum[1]) / 2.0, power);
        result[0][0] = result[1][1] = value;
    } else {
        const double low = std::pow(spectrum[0], power);
        const double high = std::pow(spectrum[1], power);
        // Spectral projectors P_high = (A - lambda_low I)/(lambda_high-lambda_low).
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                const double delta = i == j ? 1.0 : 0.0;
                result[i][j] = low * delta + (high - low)
                    * (a[i][j] - spectrum[0] * delta) / gap;
            }
        }
    }
    require(std::isfinite(norm(result)), "spectral power result nonfinite");
    return result;
}

Matrix mockOperator(double mass) {
    Matrix result{};
    result[0][0] = Complex(1.7 + 0.8 * mass, 0.2);
    result[0][1] = Complex(0.45, -0.35);
    result[1][0] = Complex(-0.25, 0.55);
    result[1][1] = Complex(2.2 + 0.3 * mass, -0.15 + 0.2 * mass);
    return result;
}

Matrix hpdRatio(const Matrix &numerator, const Matrix &denominator) {
    const Matrix q = multiply(numerator, inverse(denominator));
    return multiply(dagger(q), q);
}

Matrix cholesky(const Matrix &c) {
    positiveSpectrum(c);
    Matrix lower{};
    lower[0][0] = std::sqrt(c[0][0].real());
    lower[1][0] = c[1][0] / lower[0][0];
    const double pivot = c[1][1].real() - std::norm(lower[1][0]);
    require(pivot > 0.0, "positive Cholesky pivot");
    lower[1][1] = std::sqrt(pivot);
    return lower;
}

template<class Function>
void checkRejected(Function function, size_t &count) {
    try { function(); }
    catch (const std::invalid_argument &) { ++count; return; }
    throw std::runtime_error("invalid dense mock input was accepted");
}

struct Diagnostics {
    double maxDeterminantDifference = 0.0;
    double maxMatrixDifference = 0.0;
    double maxActionDifference = 0.0;
    double minEigenvalue = std::numeric_limits<double>::infinity();
    double minCommutatorNorm = std::numeric_limits<double>::infinity();
    double minNaiveHermiticityViolation = std::numeric_limits<double>::infinity();
    size_t factors = 0;
    size_t powers = 0;
};

void checkFactor(const Matrix &ma, const Matrix &mb, bool equalOperators, Diagnostics &d) {
    auto compareMatrix = [&](const Matrix &actual, const Matrix &expected, const std::string &label) {
        d.maxMatrixDifference = std::max(d.maxMatrixDifference, checkMatrix(actual, expected, label));
    };
    const Matrix na = multiply(dagger(ma), ma);
    const Matrix nb = multiply(dagger(mb), mb);
    const Matrix c = hpdRatio(ma, mb);
    const Matrix mbInverse = inverse(mb);
    const Matrix congruence = multiply(multiply(dagger(mbInverse), na), mbInverse);
    compareMatrix(c, congruence, "HPD congruence");
    compareMatrix(c, dagger(c), "ratio Hermiticity");
    const auto spectrum = positiveSpectrum(c);
    d.minEigenvalue = std::min(d.minEigenvalue, spectrum[0]);
    const double expectedDeterminant = std::norm(determinant(ma)) / std::norm(determinant(mb));
    checkClose(determinant(c).imag(), 0.0, "real ratio determinant");
    d.maxDeterminantDifference = std::max(d.maxDeterminantDifference,
        checkClose(determinant(c).real(), expectedDeterminant, "operator determinant ratio"));
    checkClose(determinant(c).real(), determinant(na).real() / determinant(nb).real(),
               "normal determinant ratio");
    const Matrix naive = multiply(na, inverse(nb));
    if (!equalOperators) {
        const double commutator = norm(subtract(multiply(na, nb), multiply(nb, na)));
        const double violation = norm(subtract(naive, dagger(naive)));
        require(commutator > 1e-3 && violation > 1e-3, "genuinely noncommuting inputs");
        require(std::abs(expectedDeterminant - 1.0) > 1e-3, "nontrivial ratio determinant");
        d.minCommutatorNorm = std::min(d.minCommutatorNorm, commutator);
        d.minNaiveHermiticityViolation = std::min(d.minNaiveHermiticityViolation, violation);
    } else {
        compareMatrix(c, identity(), "equal operators give identity");
    }
    const std::array<Vector, 3> noises{{
        Vector{{Complex(1.0, 0.0), Complex(0.0, 0.0)}},
        Vector{{Complex(0.0, 0.0), Complex(1.0, 0.0)}},
        Vector{{Complex(0.3, -0.7), Complex(1.2, 0.4)}}
    }};
    for (double p : {0.5, 1.0, 1.5}) {
        const Matrix heatbath = positivePower(c, p / 2.0);
        const Matrix action = positivePower(c, -p);
        const Matrix covariance = positivePower(c, p);
        compareMatrix(multiply(heatbath, dagger(heatbath)), covariance, "heatbath covariance");
        compareMatrix(multiply(multiply(dagger(heatbath), action), heatbath), identity(),
                      "full heatbath/action identity");
        checkClose(determinant(action).imag(), 0.0, "real action determinant");
        checkClose(determinant(action).real(), std::pow(expectedDeterminant, -p),
                   "Gaussian action-kernel determinant");
        for (const auto &xi : noises) {
            const auto phi = applyMatrix(heatbath, xi);
            const Complex value = inner(phi, applyMatrix(action, phi));
            checkClose(value.imag(), 0.0, "real pseudofermion action");
            require(value.real() > 0.0, "positive nonzero action");
            d.maxActionDifference = std::max(d.maxActionDifference,
                checkClose(value.real(), normSquared(xi), "heatbath/action consistency"));
            if (equalOperators) checkClose(normSquared(Vector{{phi[0] - xi[0], phi[1] - xi[1]}}),
                                           0.0, "identity heatbath unchanged");
        }
        if (!equalOperators) {
            // Negative control: reversing heatbath sign must NOT satisfy this identity.
            const Matrix wrongHeatbath = positivePower(c, -p / 2.0);
            require(norm(subtract(multiply(multiply(dagger(wrongHeatbath), action), wrongHeatbath),
                                  identity())) > 1e-3, "wrong heatbath sign detected");
        }
        if (p == 1.0) {
            compareMatrix(action, inverse(c), "independent inverse action");
            const Matrix lower = cholesky(c);
            compareMatrix(multiply(lower, dagger(lower)), c, "Cholesky covariance");
            compareMatrix(multiply(multiply(dagger(lower), inverse(c)), lower), identity(),
                          "independent Cholesky heatbath/action");
            for (const auto &xi : noises) {
                const auto phi = applyMatrix(lower, xi);
                const auto value = inner(phi, applyMatrix(inverse(c), phi));
                checkClose(value.imag(), 0.0, "real Cholesky action");
                d.maxActionDifference = std::max(d.maxActionDifference,
                    checkClose(value.real(), normSquared(xi), "Cholesky action consistency"));
            }
        }
        ++d.powers;
    }
    ++d.factors;
}

void runTest() {
    // Independent known-spectrum reference for the private fractional-power helper.
    Matrix unitary{};
    const double r = 1.0 / std::sqrt(2.0);
    unitary[0][0] = unitary[1][1] = r;
    unitary[0][1] = unitary[1][0] = Complex(0.0, r);
    checkMatrix(multiply(dagger(unitary), unitary), identity(), "reference unitary");
    Matrix diagonal{};
    diagonal[0][0] = 0.6;
    diagonal[1][1] = 2.4;
    const Matrix reference = multiply(multiply(unitary, diagonal), dagger(unitary));
    for (double p : {-1.5, -1.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.5}) {
        Matrix expectedDiagonal{};
        expectedDiagonal[0][0] = std::pow(0.6, p);
        expectedDiagonal[1][1] = std::pow(2.4, p);
        checkMatrix(positivePower(reference, p),
                    multiply(multiply(unitary, expectedDiagonal), dagger(unitary)), "known spectrum power");
    }

    Diagnostics d;
    const std::array<double, 4> masses{{0.05, 0.3, 0.7, 1.0}};
    double factorDeterminantProduct = 1.0;
    for (size_t i = 0; i + 1 < masses.size(); ++i) {
        const Matrix ma = mockOperator(masses[i]);
        const Matrix mb = mockOperator(masses[i + 1]);
        checkFactor(ma, mb, false, d);
        factorDeterminantProduct *= determinant(hpdRatio(ma, mb)).real();
    }
    checkFactor(mockOperator(masses.front()), mockOperator(masses.back()), false, d);
    const double endpointRatio = std::norm(determinant(mockOperator(masses.front())))
        / std::norm(determinant(mockOperator(masses.back())));
    checkClose(factorDeterminantProduct, endpointRatio, "dense ladder determinant telescoping");
    for (double p : {0.5, 1.0, 1.5}) {
        double product = 1.0;
        for (size_t i = 0; i + 1 < masses.size(); ++i)
            product *= std::pow(determinant(hpdRatio(mockOperator(masses[i]),
                                                    mockOperator(masses[i + 1]))).real(), p);
        checkClose(product, std::pow(endpointRatio, p), "weighted dense ladder telescoping");
    }
    checkFactor(mockOperator(0.4), mockOperator(0.4), true, d);
    size_t rejected = 0;
    checkRejected([] { inverse(Matrix{}); }, rejected);
    Matrix indefinite = identity();
    indefinite[0][0] = -1.0;
    checkRejected([&] { positivePower(indefinite, 0.5); }, rejected);
    Matrix singular = identity();
    singular[1][1] = 0.0;
    checkRejected([&] { positivePower(singular, -0.5); }, rejected);
    Matrix nonHermitian = identity();
    nonHermitian[0][1] = Complex(0.0, 0.3);
    checkRejected([&] { positivePower(nonHermitian, 0.5); }, rejected);
    checkRejected([] { positivePower(identity(), std::numeric_limits<double>::quiet_NaN()); }, rejected);
    Matrix nonfinite = identity();
    nonfinite[0][0] = std::numeric_limits<double>::infinity();
    checkRejected([&] { positivePower(nonfinite, 0.5); }, rejected);
    require(d.factors == 5 && d.powers == 15 && rejected == 6, "expected coverage");
    std::cout << std::setprecision(17)
              << "MDWF dense HPD determinant ratio mock test passed with factors = " << d.factors
              << ", powerCases = " << d.powers << ", rejectedInputs = " << rejected
              << ", minEigenvalue = " << d.minEigenvalue
              << ", minNormalCommutatorNorm = " << d.minCommutatorNorm
              << ", minNaiveHermiticityViolation = " << d.minNaiveHermiticityViolation
              << ", maxRatioDeterminantAbsDiff = " << d.maxDeterminantDifference
              << ", maxFactorMatrixAbsDiff = " << d.maxMatrixDifference
              << ", maxHeatbathActionAbsDiff = " << d.maxActionDifference << std::endl;
}
} // namespace

#ifdef MDWF_DENSE_RATIONAL_MOCK_TEST
namespace {
// Mock-only logarithmic trapezoid quadrature of
// x^-beta = sin(pi beta)/pi * integral exp((1-beta)s)/(x+exp(s)) ds.
// See https://dlmf.nist.gov/5.12.E3 and https://dlmf.nist.gov/5.5.E3.
// Not minimax, not production coefficients, and not tailored to eigenvalues.
MDWFRationalCoefficients<double> inverseFractionCoefficients(
    double beta, int halfSteps, MDWFRationalCoefficientRole role) {
    if (!(beta > 0.0 && beta < 1.0) || halfSteps <= 0 || halfSteps > 448)
        throw std::invalid_argument("invalid tiny mock quadrature request");
    MDWFExplicitRationalInput<double> input{"dense-mock-intermediate-inverse-fraction", role,
                                           0.0, {}, {}};
    const double step = 0.25;
    const double prefactor = std::sin(std::acos(-1.0) * beta) / std::acos(-1.0);
    for (int k = -halfSteps; k <= halfSteps; ++k) {
        const double s = step * k;
        const double endpointWeight = (k == -halfSteps || k == halfSteps) ? 0.5 : 1.0;
        input.numerator.push_back(endpointWeight * step * prefactor * std::exp((1.0 - beta) * s));
        input.denominator.push_back(std::exp(s));
    }
    return makeMDWFRationalCoefficients(input);
}

void checkMockInterval(const Matrix &c) {
    const auto spectrum = positiveSpectrum(c);
    if (spectrum[0] < 0.25 || spectrum[1] > 2.0)
        throw std::invalid_argument("dense mock spectrum outside [0.25, 2]");
}

Matrix scalarSpectralReference(const Matrix &c, const MDWFRationalCoefficients<double> &coefficients) {
    const auto spectrum = positiveSpectrum(c);
    const double low = evaluateMDWFRationalScalar(spectrum[0], coefficients);
    const double high = evaluateMDWFRationalScalar(spectrum[1], coefficients);
    const double gap = spectrum[1] - spectrum[0];
    Matrix result{};
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            const double delta = i == j ? 1.0 : 0.0;
            result[i][j] = gap <= 1e-13 ? low * delta
                : low * delta + (high - low) * (c[i][j] - spectrum[0] * delta) / gap;
        }
    }
    return result;
}

Matrix densePartialFraction(const Matrix &c, const MDWFRationalCoefficients<double> &coefficients,
                            double &maxResidual) {
    checkMockInterval(c);
    coefficients.validate();
    Matrix result{};
    result[0][0] = result[1][1] = coefficients.constant;
    for (size_t term = 0; term < coefficients.shift.size(); ++term) {
        Matrix shifted = c;
        shifted[0][0] += coefficients.shift[term];
        shifted[1][1] += coefficients.shift[term];
        const Matrix solution = inverse(shifted);
        // Both basis columns are solved directly, not through CG/multishift.
        maxResidual = std::max(maxResidual,
            checkMatrix(multiply(shifted, solution), identity(), "dense shifted solve residual"));
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                result[i][j] += coefficients.numerator[term] * solution[i][j];
    }
    checkMatrix(result, scalarSpectralReference(c, coefficients), "dense/scalar partial fraction");
    return result;
}

Matrix rationalPower(const Matrix &c, double power, double &maxResidual) {
    checkMockInterval(c);
    if (power > 0.0 && power < 1.0) {
        // x^alpha = x * x^-(1-alpha): avoid enormous cancelling constants.
        const auto coefficients = inverseFractionCoefficients(
            1.0 - power, 448, MDWFRationalCoefficientRole::Heatbath);
        return multiply(c, densePartialFraction(c, coefficients, maxResidual));
    }
    if (power == -1.0) {
        MDWFRationalCoefficients<double> coefficients{0.0, {1.0}, {0.0}};
        return densePartialFraction(c, coefficients, maxResidual);
    }
    if (power < 0.0 && power > -2.0) {
        const double beta = power > -1.0 ? -power : -power - 1.0;
        const auto coefficients = inverseFractionCoefficients(beta, 448, MDWFRationalCoefficientRole::Action);
        const Matrix fraction = densePartialFraction(c, coefficients, maxResidual);
        // For p = 1.5, x^-p = x^-1 * r_(p-1)(x); retain the composition explicitly.
        return power > -1.0 ? fraction : multiply(inverse(c), fraction);
    }
    throw std::invalid_argument("unsupported dense mock rational exponent");
}

double checkApproxMatrix(const Matrix &actual, const Matrix &expected, double budget,
                         const std::string &label) {
    const double relative = norm(subtract(actual, expected)) / norm(expected);
    require(std::isfinite(relative) && relative <= budget, label + " exceeds mock approximation budget");
    return relative;
}

void runRationalTest() {
    double maxScalarRelative = 0.0;
    double minCoarseRelative = std::numeric_limits<double>::infinity();
    double maxResidual = 0.0;
    double maxHeatbathRelative = 0.0;
    double maxActionKernelRelative = 0.0;
    double maxConsistencyRelative = 0.0;
    double maxGeneratedActionRelative = 0.0;
    double maxFixedPhiActionRelative = 0.0;
    for (double beta : {0.25, 0.5, 0.75}) {
        const auto coarse = inverseFractionCoefficients(beta, 96, MDWFRationalCoefficientRole::Action);
        const auto refined = inverseFractionCoefficients(beta, 448, MDWFRationalCoefficientRole::Action);
        double coarseError = 0.0;
        double fineError = 0.0;
        for (size_t point = 0; point <= 256; ++point) {
            const double x = 0.25 + 1.75 * point / 256.0;
            const double exact = std::pow(x, -beta);
            const double coarseValue = evaluateMDWFRationalScalar(x, coarse);
            const double fineValue = evaluateMDWFRationalScalar(x, refined);
            require(std::isfinite(coarseValue) && std::isfinite(fineValue), "finite scalar approximation");
            coarseError = std::max(coarseError, std::abs(coarseValue / exact - 1.0));
            fineError = std::max(fineError, std::abs(fineValue / exact - 1.0));
        }
        require(std::isfinite(fineError) && fineError < 5e-12, "refined scalar interval accuracy");
        require(coarseError > 1e-7 && fineError < coarseError * 1e-4, "quadrature refinement control");
        maxScalarRelative = std::max(maxScalarRelative, fineError);
        minCoarseRelative = std::min(minCoarseRelative, coarseError);
    }

    std::array<Matrix, 6> matrices{};
    const std::array<double, 4> masses{{0.05, 0.3, 0.7, 1.0}};
    for (size_t i = 0; i < 3; ++i)
        matrices[i] = hpdRatio(mockOperator(masses[i]), mockOperator(masses[i + 1]));
    matrices[3] = hpdRatio(mockOperator(masses.front()), mockOperator(masses.back()));
    matrices[4] = hpdRatio(mockOperator(0.4), mockOperator(0.4));
    // Rotated interval endpoints, deliberately not tied to the mock mass ladder.
    matrices[5][0][0] = matrices[5][1][1] = 1.125;
    matrices[5][0][1] = Complex(0.0, 0.875);
    matrices[5][1][0] = Complex(0.0, -0.875);
    const std::array<Vector, 3> noises{{
        Vector{{Complex(1.0, 0.0), Complex(0.0, 0.0)}},
        Vector{{Complex(0.0, 0.0), Complex(1.0, 0.0)}},
        Vector{{Complex(0.3, -0.7), Complex(1.2, 0.4)}}
    }};
    size_t cases = 0;
    for (const auto &c : matrices) {
        // Check constant handling and signed residues independently of power coefficients.
        const MDWFRationalCoefficients<double> generic{0.25, {0.3, -0.1}, {0.0, 0.2}};
        densePartialFraction(c, generic, maxResidual);
        for (double p : {0.5, 1.0, 1.5}) {
            const Matrix heatbath = rationalPower(c, p / 2.0, maxResidual);
            const Matrix action = rationalPower(c, -p, maxResidual);
            const Matrix exactAction = positivePower(c, -p);
            positiveSpectrum(action);
            maxHeatbathRelative = std::max(maxHeatbathRelative,
                checkApproxMatrix(heatbath, positivePower(c, p / 2.0), 2e-11, "rational heatbath"));
            maxActionKernelRelative = std::max(maxActionKernelRelative,
                checkApproxMatrix(action, exactAction, 2e-11, "rational action kernel"));
            maxConsistencyRelative = std::max(maxConsistencyRelative,
                checkApproxMatrix(multiply(multiply(dagger(heatbath), action), heatbath),
                                  identity(), 1e-10, "rational full heatbath/action identity"));
            checkApproxMatrix(multiply(heatbath, dagger(heatbath)), positivePower(c, p),
                              1e-10, "rational heatbath covariance");
            for (const auto &xi : noises) {
                const auto phi = applyMatrix(heatbath, xi);
                const auto value = inner(phi, applyMatrix(action, phi));
                checkClose(value.imag(), 0.0, "real rational action");
                require(value.real() > 0.0, "positive rational action");
                const auto exactValue = inner(phi, applyMatrix(exactAction, phi));
                checkClose(exactValue.imag(), 0.0, "real exact fixed-phi action");
                require(exactValue.real() > 0.0, "positive exact fixed-phi action");
                const double fixedPhiRelative = std::abs(value.real() / exactValue.real() - 1.0);
                require(std::isfinite(fixedPhiRelative) && fixedPhiRelative < 1e-10,
                        "rational versus exact fixed-phi action");
                maxFixedPhiActionRelative = std::max(maxFixedPhiActionRelative, fixedPhiRelative);
                const double relative = std::abs(value.real() / normSquared(xi) - 1.0);
                require(std::isfinite(relative) && relative < 1e-10, "rational generated action consistency");
                maxGeneratedActionRelative = std::max(maxGeneratedActionRelative, relative);
            }
            ++cases;
        }
    }
    size_t rejected = 0;
    checkRejected([] { inverseFractionCoefficients(0.0, 448, MDWFRationalCoefficientRole::Action); }, rejected);
    checkRejected([] { inverseFractionCoefficients(1.0, 448, MDWFRationalCoefficientRole::Action); }, rejected);
    checkRejected([] { inverseFractionCoefficients(0.5, 0, MDWFRationalCoefficientRole::Action); }, rejected);
    Matrix outside = identity();
    outside[0][0] = 0.1;
    checkRejected([&] { rationalPower(outside, 0.25, maxResidual); }, rejected);
    checkRejected([&] { rationalPower(identity(), 2.0, maxResidual); }, rejected);
    require(cases == 18 && rejected == 5, "rational expected coverage");
    std::cout << std::setprecision(17)
              << "MDWF dense HPD rational application mock test passed with matrices = 6, powerCases = " << cases
              << ", interval = [0.25, 2], samplesPerInversePower = 257, fractionalTerms = 897"
              << ", rejectedInputs = " << rejected
              << ", minCoarseScalarRelError = " << minCoarseRelative
              << ", maxRefinedScalarRelError = " << maxScalarRelative
              << ", maxDenseShiftResidual = " << maxResidual
              << ", maxHeatbathMatrixRelError = " << maxHeatbathRelative
              << ", maxActionKernelMatrixRelError = " << maxActionKernelRelative
              << ", maxFullConsistencyRelError = " << maxConsistencyRelative
              << ", maxFixedPhiActionRelError = " << maxFixedPhiActionRelative
              << ", maxGeneratedActionRelError = " << maxGeneratedActionRelative << std::endl;
}
} // namespace
#endif

int main() {
    try {
        stdLogger.setVerbosity(INFO);
        runTest();
#ifdef MDWF_DENSE_RATIONAL_MOCK_TEST
        runRationalTest();
#endif
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return -1;
    }
}
