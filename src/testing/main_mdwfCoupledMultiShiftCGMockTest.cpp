/*
 * MDWF coupled-5D multishift-CG mock SPD smoke test.
 *
 * This validates the isolated multishift scaffold on controlled positive
 * definite mock operators before using it with the MDWF operator.  It checks
 * both A = 2 I and a fifth-slice diagonal operator A_s = 1 + 0.1 s, with exact
 * shifted solutions x_s = b_s / (A_s + sigma).  It does not apply MDWF, does
 * not call the existing inverter/RHMC code, and does not touch force code,
 * HISQ, or smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledMultiShiftCG.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFMultiShiftDiagonalLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

private:
    floatT _diagonal;

public:
    explicit MDWFMultiShiftDiagonalLinearOperator(floatT diagonal)
        : _diagonal(diagonal) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out = _diagonal * spinor_in;
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFMultiShiftSliceDiagonalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFMultiShiftSliceDiagonalApply(const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0) + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFMultiShiftSliceDiagonalLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFMultiShiftSliceDiagonalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCoupledMultiShiftPattern {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(1000 * site.stack + site.isite + component + 1), 0.0);
        }
        return out;
    }
};

template<class Spinor>
std::vector<Spinor *> makeSolutionPointers(std::vector<std::unique_ptr<Spinor>> &solutions) {
    std::vector<Spinor *> pointers;
    pointers.reserve(solutions.size());
    for (auto &solution : solutions) {
        pointers.push_back(solution.get());
    }
    return pointers;
}

template<size_t Ls>
void runMDWFCoupledMultiShiftDiagonalMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using DiagonalOperator = MDWFMultiShiftDiagonalLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, DiagonalOperator>;
    using Spinor = typename DiagonalOperator::Spinor;

    const std::vector<double> sigma = {0.0, 0.1, 0.5, 1.0};
    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_multishift_diagonal_source");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(commBase, "MDWF_multishift_diagonal_source_host");
    std::vector<std::unique_ptr<Spinor>> solutions;
    std::vector<std::unique_ptr<MDWFSpinor<double, false, All, HaloDepth, Ls>>> solutionHosts;

    for (size_t shift = 0; shift < sigma.size(); shift++) {
        solutions.emplace_back(new Spinor(commBase, "MDWF_multishift_diagonal_solution_" + std::to_string(shift)));
        solutionHosts.emplace_back(new MDWFSpinor<double, false, All, HaloDepth, Ls>(
            commBase, "MDWF_multishift_diagonal_solution_host_" + std::to_string(shift)));
    }

    source.template iterateOverBulk<>(FillMDWFCoupledMultiShiftPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    DiagonalOperator diagonalOperator(2.0);
    Adapter adapter(diagonalOperator);
    MDWFCoupledMultiShiftCG<double, Adapter> multishiftCg;
    std::vector<Spinor *> solutionPointers = makeSolutionPointers(solutions);

    MDWFCoupledMultiShiftCGResults<double> result
        = multishiftCg.invert(adapter, solutionPointers, source, sigma, 8, 1e-12, true);

    sourceHost = source;
    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t shift = 0; shift < sigma.size(); shift++) {
        *solutionHosts[shift] = *solutions[shift];
        Vect12ArrayAcc<double> solutionAcc = solutionHosts[shift]->getAccessor();

        for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
            for (size_t stack = 0; stack < Ls; stack++) {
                const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
                Vect12<double> sourceValue = sourceAcc.getElement(site);
                Vect12<double> solutionValue = solutionAcc.getElement(site);
                for (size_t component = 0; component < 12; component++) {
                    const COMPLEX(double) expected = sourceValue.data[component] / (2.0 + sigma[shift]);
                    const double diff = std::abs(real(solutionValue.data[component] - expected))
                                        + std::abs(imag(solutionValue.data[component] - expected));
                    if (diff > maxDiff) {
                        maxDiff = diff;
                    }
                }
            }

        if (!result.shifts[shift].converged
            || result.shifts[shift].iterations != 1
            || result.shifts[shift].residue > 1e-12) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF multishift diagonal mock test failed for shift ", shift,
                ": sigma = ", sigma[shift],
                ", converged = ", result.shifts[shift].converged,
                ", iterations = ", result.shifts[shift].iterations,
                ", residue = ", result.shifts[shift].residue));
        }
    }

    if (!result.converged() || maxDiff > 1e-10) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF multishift diagonal mock test failed: converged = ", result.converged(),
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF multishift diagonal mock test passed with Ls = ", Ls,
                    ", shifts = ", sigma.size(),
                    ", maxDiff = ", maxDiff);
}

template<size_t Ls>
void runMDWFCoupledMultiShiftSliceDiagonalMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using SliceDiagonalOperator = MDWFMultiShiftSliceDiagonalLinearOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, SliceDiagonalOperator>;
    using Spinor = typename SliceDiagonalOperator::Spinor;

    const std::vector<double> sigma = {0.0, 0.1, 0.3, 0.7};
    MDWFSpinor<double, true, All, HaloDepth, Ls> source(commBase, "MDWF_multishift_slice_diagonal_source");
    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(commBase, "MDWF_multishift_slice_diagonal_source_host");
    std::vector<std::unique_ptr<Spinor>> solutions;
    std::vector<std::unique_ptr<MDWFSpinor<double, false, All, HaloDepth, Ls>>> solutionHosts;

    for (size_t shift = 0; shift < sigma.size(); shift++) {
        solutions.emplace_back(new Spinor(commBase, "MDWF_multishift_slice_diagonal_solution_" + std::to_string(shift)));
        solutionHosts.emplace_back(new MDWFSpinor<double, false, All, HaloDepth, Ls>(
            commBase, "MDWF_multishift_slice_diagonal_solution_host_" + std::to_string(shift)));
    }

    source.template iterateOverBulk<>(FillMDWFCoupledMultiShiftPattern<double, All, HaloDepth, Ls>());
    source.updateAll();

    SliceDiagonalOperator sliceDiagonalOperator;
    Adapter adapter(sliceDiagonalOperator);
    MDWFCoupledMultiShiftCG<double, Adapter> multishiftCg;
    std::vector<Spinor *> solutionPointers = makeSolutionPointers(solutions);

    MDWFCoupledMultiShiftCGResults<double> result
        = multishiftCg.invert(adapter, solutionPointers, source, sigma, 32, 1e-12, true);

    sourceHost = source;
    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t shift = 0; shift < sigma.size(); shift++) {
        *solutionHosts[shift] = *solutions[shift];
        Vect12ArrayAcc<double> solutionAcc = solutionHosts[shift]->getAccessor();

        for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
            for (size_t stack = 0; stack < Ls; stack++) {
                const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
                const double diagonal = 1.0 + 0.1 * static_cast<double>(stack);
                Vect12<double> sourceValue = sourceAcc.getElement(site);
                Vect12<double> solutionValue = solutionAcc.getElement(site);
                for (size_t component = 0; component < 12; component++) {
                    const COMPLEX(double) expected = sourceValue.data[component] / (diagonal + sigma[shift]);
                    const double diff = std::abs(real(solutionValue.data[component] - expected))
                                        + std::abs(imag(solutionValue.data[component] - expected));
                    if (diff > maxDiff) {
                        maxDiff = diff;
                    }
                }
            }

        if (!result.shifts[shift].converged
            || result.shifts[shift].iterations <= 1
            || result.shifts[shift].residue > 1e-12) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF multishift slice-diagonal mock test failed for shift ", shift,
                ": sigma = ", sigma[shift],
                ", converged = ", result.shifts[shift].converged,
                ", iterations = ", result.shifts[shift].iterations,
                ", residue = ", result.shifts[shift].residue));
        }
    }

    if (!result.converged() || maxDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF multishift slice-diagonal mock test failed: converged = ", result.converged(),
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF multishift slice-diagonal mock test passed with Ls = ", Ls,
                    ", shifts = ", sigma.size(),
                    ", maxDiff = ", maxDiff);
}

template<size_t Ls>
void runMDWFCoupledMultiShiftCGMockTest(CommunicationBase &commBase) {
    runMDWFCoupledMultiShiftDiagonalMockTest<Ls>(commBase);
    runMDWFCoupledMultiShiftSliceDiagonalMockTest<Ls>(commBase);
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

        runMDWFCoupledMultiShiftCGMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
