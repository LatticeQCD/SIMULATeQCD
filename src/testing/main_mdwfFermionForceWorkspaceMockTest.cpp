/*
 * MDWF fermion-force workspace mock test.
 *
 * This validates only the pre-force workspace:
 *
 *     chi_i = (A + sigma_i)^(-1) phi
 *     eta_i = M chi_i
 *
 * with a controlled diagonal mock where A_s = 1 + 0.1 s and M_s = sqrt(A_s).
 * It does not accumulate gauge force, update momenta, call RHMC/HMC, touch
 * HISQ, or use smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFermionForceMockNormalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFFermionForceMockNormalApply(
        const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0)
                                + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFermionForceMockForwardApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFFermionForceMockForwardApply(
        const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0)
                                + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return sqrt(diagonal) * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFFermionForceMockNormalOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFFermionForceMockNormalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFFermionForceMockForwardOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFFermionForceMockForwardApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFFermionForceWorkspaceMockSource {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.01) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.002) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<class Workspace, class Spinor, size_t Ls>
double compareMDWFFermionForceWorkspaceMock(
    const Workspace &workspace,
    Spinor &source,
    const MDWFRationalCoefficients<double> &coefficients) {

    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> sourceHost(
        source.getComm(), "MDWF_force_workspace_mock_source_host");
    sourceHost = source;
    Vect12ArrayAcc<double> sourceAcc = sourceHost.getAccessor();

    double maxDiff = 0.0;
    for (size_t term = 0; term < workspace.size(); term++) {
        MDWFSpinor<double, false, All, HaloDepth, Ls> chiHost(
            source.getComm(), "MDWF_force_workspace_mock_chi_host_" + std::to_string(term));
        MDWFSpinor<double, false, All, HaloDepth, Ls> etaHost(
            source.getComm(), "MDWF_force_workspace_mock_eta_host_" + std::to_string(term));
        chiHost = workspace.chi(term);
        etaHost = workspace.eta(term);

        Vect12ArrayAcc<double> chiAcc = chiHost.getAccessor();
        Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();

        for (size_t isite = 0; isite < GInd::getLatData().vol4; isite++)
            for (size_t stack = 0; stack < Ls; stack++) {
                const gSiteStack site = GInd::getSiteStack(GInd::getSite(isite), stack);
                const double diagonal = 1.0 + 0.1 * static_cast<double>(stack);
                const double forwardDiagonal = std::sqrt(diagonal);
                const Vect12<double> sourceValue = sourceAcc.getElement(site);
                const Vect12<double> chiValue = chiAcc.getElement(site);
                const Vect12<double> etaValue = etaAcc.getElement(site);

                for (size_t component = 0; component < 12; component++) {
                    const COMPLEX(double) expectedChi
                        = sourceValue.data[component] / (diagonal + coefficients.shift[term]);
                    const COMPLEX(double) expectedEta = forwardDiagonal * expectedChi;
                    const double chiDiff = std::abs(real(chiValue.data[component] - expectedChi))
                                           + std::abs(imag(chiValue.data[component] - expectedChi));
                    const double etaDiff = std::abs(real(etaValue.data[component] - expectedEta))
                                           + std::abs(imag(etaValue.data[component] - expectedEta));
                    maxDiff = std::max(maxDiff, std::max(chiDiff, etaDiff));
                }
            }
    }
    return maxDiff;
}

template<size_t Ls>
void runMDWFFermionForceWorkspaceMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using NormalOperator = MDWFFermionForceMockNormalOperator<double, HaloDepth, Ls>;
    using ForwardOperator = MDWFFermionForceMockForwardOperator<double, HaloDepth, Ls>;
    using Workspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> forceInput{
        "mock_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.125,
        {0.5, -0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(forceInput);

    Spinor source(commBase, "MDWF_force_workspace_mock_source");
    source.template iterateOverBulk<>(FillMDWFFermionForceWorkspaceMockSource<double, All, HaloDepth, Ls>());
    source.updateAll();

    NormalOperator normalOperator;
    ForwardOperator forwardOperator;
    Workspace workspace;
    workspace.prepare(normalOperator, forwardOperator, source, coefficients, 32, 1e-12,
                      "MDWF_force_workspace_mock");

    double maxResidue = 0.0;
    bool sawMultipleIterations = false;
    const auto &shiftInfo = workspace.shiftInfo();
    for (size_t term = 0; term < shiftInfo.size(); term++) {
        maxResidue = std::max(maxResidue, shiftInfo[term].residue);
        sawMultipleIterations = sawMultipleIterations || shiftInfo[term].iterations > 1;
        if (shiftInfo[term].numerator != coefficients.numerator[term]
            || shiftInfo[term].shift != coefficients.shift[term]) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF force workspace mock test failed coefficient bookkeeping for term ", term));
        }
    }

    const double maxDiff = compareMDWFFermionForceWorkspaceMock<Workspace, Spinor, Ls>(
        workspace, source, coefficients);

    if (mdwfRationalCoefficientRoleName(forceInput.role) != "force"
        || workspace.size() != coefficients.shift.size()
        || !workspace.converged()
        || !sawMultipleIterations
        || maxResidue > 1e-12
        || maxDiff > 1e-8) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF force workspace mock test failed: size = ", workspace.size(),
            ", expectedSize = ", coefficients.shift.size(),
            ", converged = ", workspace.converged(),
            ", sawMultipleIterations = ", sawMultipleIterations,
            ", maxResidue = ", maxResidue,
            ", maxDiff = ", maxDiff));
    }

    rootLogger.info("MDWF force workspace mock test passed with Ls = ", Ls,
                    ", terms = ", workspace.size(),
                    ", maxResidue = ", maxResidue,
                    ", maxDiff = ", maxDiff);
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

        runMDWFFermionForceWorkspaceMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
