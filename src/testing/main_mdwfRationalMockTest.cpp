/*
 * MDWF rational-operator mock SPD smoke test.
 *
 * This validates the isolated rational-operator scaffold on controlled
 * positive-definite mock operators before using it with the MDWF normal
 * operator.  It checks
 *
 *     R(A) b = c0 b + sum_i numerator_i (A + shift_i)^(-1) b
 *
 * against exact diagonal answers.  It does not apply MDWF, does not call
 * RHMC/HMC, and does not touch force code, HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRationalOperator.h"

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFRationalMockDiagonalLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

private:
    floatT _diagonal;

public:
    explicit MDWFRationalMockDiagonalLinearOperator(floatT diagonal)
        : _diagonal(diagonal) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out = _diagonal * spinor_in;
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFRationalMockSliceDiagonalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFRationalMockSliceDiagonalApply(const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0) + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFRationalMockSliceDiagonalLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFRationalMockSliceDiagonalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFRationalMockPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(1000 * site.stack + site.isite + component + 1), 0.0);
        }
        return out;
    }
};

template<class floatT>
floatT rationalMockFactor(floatT diagonal, const MDWFRationalCoefficients<floatT> &coefficients) {
    floatT factor = coefficients.constant;
    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        factor += coefficients.numerator[term] / (diagonal + coefficients.shift[term]);
    }
    return factor;
}

template<class Spinor, size_t Ls>
double compareRationalMockResult(Spinor &output,
                                 Spinor &source,
                                 const MDWFRationalCoefficients<double> &coefficients,
                                 bool slice_diagonal) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> outputHost(
        output.getComm(), "MDWF_rational_mock_output_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(
        source.getComm(), "MDWF_rational_mock_source_host");

    outputHost = output;
    sourceHost = source;

    Vect12ArrayAcc<double> outputAcc = outputHost.getAccessor();
    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
            const double diagonal = slice_diagonal ? 1.0 + 0.1 * static_cast<double>(stack) : 2.0;
            const double factor = rationalMockFactor(diagonal, coefficients);
            Vect12<double> outputValue = outputAcc.getElement(site);
            Vect12<double> sourceValue = sourceAcc.getElement(site);

            for (size_t component = 0; component < 12; component++) {
                const COMPLEX(double) expected = factor * sourceValue.data[component];
                const double diff = std::abs(real(outputValue.data[component] - expected))
                                    + std::abs(imag(outputValue.data[component] - expected));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
    return maxDiff;
}

template<size_t Ls>
void runMDWFRationalDiagonalMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using DiagonalOperator = MDWFRationalMockDiagonalLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, DiagonalOperator>;
    using Spinor = typename DiagonalOperator::Spinor;

    MDWFRationalCoefficients<double> coefficients{0.25, {0.5, -0.125, 0.75}, {0.0, 0.1, 0.3}};
    Spinor source(commBase, "MDWF_rational_diagonal_source");
    Spinor output(commBase, "MDWF_rational_diagonal_output");

    source.template iterateOverBulk<>(FillMDWFRationalMockPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    DiagonalOperator diagonalOperator(2.0);
    Adapter adapter(diagonalOperator);
    MDWFRationalOperator<double, Adapter> rationalOperator(
        coefficients, 8, 1e-12, "MDWF_rational_diagonal_mock");
    MDWFCoupledMultiShiftCGResults<double> result = rationalOperator.apply(adapter, output, source, true);

    double maxDiff = compareRationalMockResult<Spinor, Ls>(output, source, coefficients, false);

    for (size_t term = 0; term < result.shifts.size(); term++) {
        if (!result.shifts[term].converged
            || result.shifts[term].iterations != 1
            || result.shifts[term].residue > 1e-12) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational diagonal mock test failed for term ", term,
                ": converged = ", result.shifts[term].converged,
                ", iterations = ", result.shifts[term].iterations,
                ", residue = ", result.shifts[term].residue));
        }
    }

    if (!result.converged() || maxDiff > 1e-10) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational diagonal mock test failed: converged = ", result.converged(),
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF rational diagonal mock test passed with Ls = ", Ls,
                    ", terms = ", coefficients.shift.size(),
                    ", maxDiff = ", maxDiff);
}

template<size_t Ls>
void runMDWFRationalSliceDiagonalMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using SliceDiagonalOperator = MDWFRationalMockSliceDiagonalLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, SliceDiagonalOperator>;
    using Spinor = typename SliceDiagonalOperator::Spinor;

    MDWFRationalCoefficients<double> coefficients{0.25, {0.5, -0.125, 0.75}, {0.0, 0.1, 0.3}};
    Spinor source(commBase, "MDWF_rational_slice_diagonal_source");
    Spinor output(commBase, "MDWF_rational_slice_diagonal_output");

    source.template iterateOverBulk<>(FillMDWFRationalMockPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    SliceDiagonalOperator sliceDiagonalOperator;
    Adapter adapter(sliceDiagonalOperator);
    MDWFRationalOperator<double, Adapter> rationalOperator(
        coefficients, 32, 1e-12, "MDWF_rational_slice_diagonal_mock");
    MDWFCoupledMultiShiftCGResults<double> result = rationalOperator.apply(adapter, output, source, true);

    double maxDiff = compareRationalMockResult<Spinor, Ls>(output, source, coefficients, true);
    bool sawMultipleIterations = false;

    for (size_t term = 0; term < result.shifts.size(); term++) {
        sawMultipleIterations = sawMultipleIterations || result.shifts[term].iterations > 1;
        if (!result.shifts[term].converged || result.shifts[term].residue > 1e-12) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational slice-diagonal mock test failed for term ", term,
                ": converged = ", result.shifts[term].converged,
                ", iterations = ", result.shifts[term].iterations,
                ", residue = ", result.shifts[term].residue));
        }
    }

    if (!result.converged() || !sawMultipleIterations || maxDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational slice-diagonal mock test failed: converged = ", result.converged(),
            ", sawMultipleIterations = ", sawMultipleIterations,
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF rational slice-diagonal mock test passed with Ls = ", Ls,
                    ", terms = ", coefficients.shift.size(),
                    ", maxDiff = ", maxDiff);
}

template<size_t Ls>
void runMDWFRationalMockTest(CommunicationBase &commBase) {
    runMDWFRationalDiagonalMockTest<Ls>(commBase);
    runMDWFRationalSliceDiagonalMockTest<Ls>(commBase);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runMDWFRationalMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
