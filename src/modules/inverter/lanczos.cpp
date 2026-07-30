#include "lanczos.h"

#include "../../base/math/random.h"
#include "../../define.h"
#include "../../spinor/eigenpairs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

namespace {

template<class floatT>
struct TRLanLinearCombination {
    Vect3arrayAcc<floatT> first;
    Vect3arrayAcc<floatT> second;
    floatT firstCoefficient;
    floatT secondCoefficient;

    template<bool onDevice, Layout LatticeLayout,
             size_t HaloDepthSpin>
    TRLanLinearCombination(
            const Spinorfield<
                    floatT, onDevice, LatticeLayout,
                    HaloDepthSpin, 1> &firstVector,
            const floatT firstScale,
            const Spinorfield<
                    floatT, onDevice, LatticeLayout,
                    HaloDepthSpin, 1> &secondVector,
            const floatT secondScale)
        : first(firstVector.getAccessor()),
          second(secondVector.getAccessor()),
          firstCoefficient(firstScale),
          secondCoefficient(secondScale) {}

    __host__ __device__ Vect3<floatT>
    operator()(gSiteStack &site) const {
        return firstCoefficient * first.getElement(site)
                + secondCoefficient * second.getElement(site);
    }
};

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin>
void assignLinearCombination(
        Spinorfield<
                floatT, onDevice, LatticeLayout,
                HaloDepthSpin, 1> &out,
        const double firstCoefficient,
        const Spinorfield<
                floatT, onDevice, LatticeLayout,
                HaloDepthSpin, 1> &first,
        const double secondCoefficient,
        const Spinorfield<
                floatT, onDevice, LatticeLayout,
                HaloDepthSpin, 1> &second) {
    out.iterateOverFull(
            TRLanLinearCombination<floatT>(
                    first,
                    static_cast<floatT>(firstCoefficient),
                    second,
                    static_cast<floatT>(secondCoefficient)));
}

struct TRLanMPIConsistencySummary {
    double projectedMatrixMaxDifference = 0.0;
    double ritzEigenvalueMaxDifference = 0.0;
    double rotationMatrixMaxDifference = 0.0;
    int cyclesChecked = 0;
};

double maximumRankDifference(
        CommunicationBase &comm,
        const std::vector<double> &values) {
    const double localSize = static_cast<double>(values.size());
    const double minimumSize = comm.globalMinimum(localSize);
    const double maximumSize = comm.globalMaximum(localSize);
    if (minimumSize != maximumSize) {
        return std::numeric_limits<double>::infinity();
    }
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> rootValues(values.size(), 0.0);
    if (comm.IamRoot()) {
        rootValues = values;
    }
    comm.root2all(rootValues);

    double localMaximum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])
            || !std::isfinite(rootValues[i])) {
            if (values[i] != rootValues[i]) {
                localMaximum =
                        std::numeric_limits<double>::infinity();
            }
            continue;
        }
        localMaximum =
                std::max(
                        localMaximum,
                        std::fabs(values[i] - rootValues[i]));
    }
    return comm.globalMaximum(localMaximum);
}

std::vector<double> flattenProjectedMatrix(
        const std::vector<std::vector<double>> &projected,
        const int dimension) {
    std::vector<double> values;
    values.reserve(
            static_cast<size_t>(dimension)
            * static_cast<size_t>(dimension));
    for (int row = 0; row < dimension; ++row) {
        values.insert(
                values.end(),
                projected[row].begin(),
                projected[row].begin() + dimension);
    }
    return values;
}

std::vector<double> flattenRotationMatrix(
        const std::vector<std::vector<double>> &eigenvectors,
        const std::vector<int> &retainedColumns,
        const int sourceCount) {
    std::vector<double> values;
    values.reserve(
            static_cast<size_t>(sourceCount)
            * retainedColumns.size());
    for (int source = 0; source < sourceCount; ++source) {
        for (const int column : retainedColumns) {
            values.push_back(eigenvectors[source][column]);
        }
    }
    return values;
}

} // namespace

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::compute(
        CommunicationBase &comm,
        LinearOperator<Spinor_external> &op,
        const int requestedEigenpairs,
        std::vector<Spinor_internal> &eigenvectors,
        std::vector<double> &eigenvalues,
        const int krylovDim,
        const double breakdownTol,
        const unsigned int seed) {
    TRLanRestartParams params;
    params.krylovDim = krylovDim;
    params.breakdownTol = breakdownTol;
    params.seed = seed;
    params.maxRestarts = 0;
    params.failOnNoConvergence = false;
    compute(
            comm,
            op,
            requestedEigenpairs,
            eigenvectors,
            eigenvalues,
            params);
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::compute(
        CommunicationBase &comm,
        LinearOperator<Spinor_external> &op,
        const int requestedEigenpairs,
        std::vector<Spinor_internal> &eigenvectors,
        std::vector<double> &eigenvalues,
        const TRLanRestartParams &params) {
    if constexpr (LatticeLayout == Layout::All) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLanSpinorSolver supports only Even or Odd layout"));
    }

    eigenvectors.clear();
    eigenvalues.clear();
    if (requestedEigenpairs <= 0) {
        return;
    }

    const int krylovDimension =
            params.krylovDim > 0
                    ? params.krylovDim
                    : std::max(
                            2 * requestedEigenpairs + 8,
                            requestedEigenpairs + 2);
    const int guardVectors =
            std::max(8, requestedEigenpairs / 2);
    const int retainedDimension =
            params.thickRestartDim > 0
                    ? params.thickRestartDim
                    : std::min(
                            krylovDimension - 2,
                            requestedEigenpairs + guardVectors);
    validateParams(
            params,
            requestedEigenpairs,
            krylovDimension,
            retainedDimension);

    double residualTolerance = params.residualTol;
    double breakdownTolerance = params.breakdownTol;
    if constexpr (std::is_same_v<floatT, float>) {
        residualTolerance =
                std::max(residualTolerance, 1e-6);
        breakdownTolerance =
                std::max(breakdownTolerance, 1e-7);
    }

    rootLogger.info(
            "TRLan: requested=", requestedEigenpairs,
            " m=", krylovDimension,
            " k=", retainedDimension,
            " filterOrder=",
            params.exponential.enabled
                    ? params.exponential.order
                    : (params.chebyshev.enabled
                            ? params.chebyshev.order
                            : 0),
            " operatorShift=",
            params.exponential.enabled
                    ? params.exponential.operatorShift
                    : 0.0,
            " operatorScale=",
            params.exponential.enabled
                    ? params.exponential.operatorScale
                    : 1.0);

    const size_t basisStorageBytes =
            Basis::requiredStorageBytes(
                    static_cast<size_t>(krylovDimension),
                    static_cast<size_t>(retainedDimension));
    if constexpr (onDevice) {
        size_t availableDeviceBytes = 0;
        size_t totalDeviceBytes = 0;
        const gpuError_t memoryInfoError =
                gpuMemGetInfo(
                        &availableDeviceBytes,
                        &totalDeviceBytes);
        if (memoryInfoError != gpuSuccess) {
            GpuError(
                    "TRLan: gpuMemGetInfo failed",
                    memoryInfoError);
        }
        rootLogger.info(
                "TRLan: contiguous basis storage=",
                static_cast<double>(basisStorageBytes)
                        / (1024.0 * 1024.0 * 1024.0),
                " GiB, currently available device memory=",
                static_cast<double>(availableDeviceBytes)
                        / (1024.0 * 1024.0 * 1024.0),
                " GiB, total device memory=",
                static_cast<double>(totalDeviceBytes)
                        / (1024.0 * 1024.0 * 1024.0),
                " GiB");
        if (basisStorageBytes > availableDeviceBytes) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis and rotation scratch exceed available device memory"));
        }
    } else {
        rootLogger.info(
                "TRLan: contiguous basis storage=",
                static_cast<double>(basisStorageBytes)
                        / (1024.0 * 1024.0 * 1024.0),
                " GiB");
    }

    Basis basis(
            comm,
            static_cast<size_t>(krylovDimension),
            static_cast<size_t>(retainedDimension));
    OperatorWorkspace workspace(comm);
    Spinor_internal current(comm, "TRLan_current");
    Spinor_internal terminalVector(comm, "TRLan_terminal");

    grnd_state<false> randomState;
    randomState.make_rng_state(params.seed);
    current.gauss(randomState.state);
    normalizeOrThrow(
            current,
            breakdownTolerance,
            "TRLan could not normalize its initial vector");
    basis.store(0, current);

    std::vector<std::vector<double>> projected(
            krylovDimension,
            std::vector<double>(krylovDimension, 0.0));
    std::vector<double> filteredEigenvalues;
    std::vector<std::vector<double>> projectedEigenvectors;

    std::vector<int> finalSlots;
    std::vector<double> finalEigenvalues;
    std::vector<double> finalResiduals;

    bool converged = false;
    bool stoppedByBreakdown = false;
    int firstColumn = 0;
    TRLanMPIConsistencySummary mpiConsistency;

    for (int restart = 0;
         restart <= params.maxRestarts;
         ++restart) {
        const uint64_t applicationsAtCycleStart =
                workspace.operatorApplications;
        const uint64_t reductionsAtCycleStart =
                basis.globalReductions();
        double terminalBeta = 0.0;
        const int factorizationDimension =
                extendLanczosFactorization(
                        op,
                        basis,
                        firstColumn,
                        krylovDimension,
                        breakdownTolerance,
                        params.reorthogonalizationPasses,
                        params,
                        workspace,
                        terminalVector,
                        terminalBeta,
                        projected);

        if (factorizationDimension <= 0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan generated an empty Krylov factorization"));
        }

        diagonalizeProjected(
                projected,
                factorizationDimension,
                filteredEigenvalues,
                projectedEigenvectors);
        const std::vector<int> order =
                targetOrder(filteredEigenvalues, params);
        const int keep =
                std::min(
                        retainedDimension,
                        factorizationDimension);

        std::vector<int> retainedColumns(
                order.begin(), order.begin() + keep);

        if (params.mpiConsistencyDiagnostics) {
            mpiConsistency.projectedMatrixMaxDifference =
                    std::max(
                            mpiConsistency
                                    .projectedMatrixMaxDifference,
                            maximumRankDifference(
                                    comm,
                                    flattenProjectedMatrix(
                                            projected,
                                            factorizationDimension)));
            mpiConsistency.ritzEigenvalueMaxDifference =
                    std::max(
                            mpiConsistency
                                    .ritzEigenvalueMaxDifference,
                            maximumRankDifference(
                                    comm,
                                    filteredEigenvalues));
            mpiConsistency.rotationMatrixMaxDifference =
                    std::max(
                            mpiConsistency
                                    .rotationMatrixMaxDifference,
                            maximumRankDifference(
                                    comm,
                                    flattenRotationMatrix(
                                            projectedEigenvectors,
                                            retainedColumns,
                                            factorizationDimension)));
            ++mpiConsistency.cyclesChecked;
        }

        std::vector<double> retainedFilteredValues(
                keep, 0.0);
        std::vector<double> retainedFilteredResiduals(
                keep, 0.0);
        std::vector<double> restartCouplings(
                keep, 0.0);
        for (int i = 0; i < keep; ++i) {
            const int column = retainedColumns[i];
            retainedFilteredValues[i] =
                    filteredEigenvalues[column];
            restartCouplings[i] =
                    terminalBeta
                    * projectedEigenvectors[
                            factorizationDimension - 1][column];
            retainedFilteredResiduals[i] =
                    std::fabs(restartCouplings[i]);
        }

        basis.rotate(
                projectedEigenvectors,
                retainedColumns,
                factorizationDimension);

        double largestEstimatedResidual = 0.0;
        for (int i = 0;
             i < std::min(requestedEigenpairs, keep);
             ++i) {
            largestEstimatedResidual =
                    std::max(
                            largestEstimatedResidual,
                            retainedFilteredResiduals[i]);
        }
        rootLogger.info(
                "TRLan restart ", restart,
                ": dimension=", factorizationDimension,
                " beta=", terminalBeta,
                " estimatedResidual=",
                largestEstimatedResidual,
                " rawOperatorApplications=",
                workspace.operatorApplications
                        - applicationsAtCycleStart,
                " basisGlobalReductions=",
                basis.globalReductions()
                        - reductionsAtCycleStart);

        stoppedByBreakdown =
                terminalBeta <= breakdownTolerance
                || factorizationDimension < krylovDimension;
        const bool finalRestart =
                restart == params.maxRestarts;
        const bool periodicPhysicalCheck =
                params.physicalCheckInterval > 0
                && ((restart + 1)
                    % params.physicalCheckInterval == 0);
        const bool estimatedConverged =
                inexpensiveConvergenceGate(
                        retainedFilteredValues,
                        retainedFilteredResiduals,
                        requestedEigenpairs,
                        residualTolerance,
                        params);
        const bool performPhysicalCheck =
                estimatedConverged
                || periodicPhysicalCheck
                || finalRestart
                || stoppedByBreakdown;

        if (performPhysicalCheck) {
            std::vector<double> physicalValues(
                    keep, 0.0);
            std::vector<double> physicalResiduals(
                    keep,
                    std::numeric_limits<double>::infinity());
            std::vector<double> physicalVectorNorms(
                    keep,
                    std::numeric_limits<double>::quiet_NaN());
            for (int i = 0; i < keep; ++i) {
                basis.load(i, current);
                const double suppliedEigenvalue =
                        params.exponential.enabled
                                ? correctedExponentialEigenvalue(
                                        retainedFilteredValues[i],
                                        params.exponential)
                                : (params.chebyshev.enabled
                                        ? std::numeric_limits<
                                                double>::quiet_NaN()
                                        : retainedFilteredValues[i]);
                double rayleigh = 0.0;
                physicalResiduals[i] =
                        physicalResidual(
                                op,
                                current,
                                suppliedEigenvalue,
                                params.exponential.enabled,
                                params.exponential.enabled
                                        ? params.exponential
                                                .operatorShift
                                        : 0.0,
                                params.exponential.enabled
                                        ? params.exponential
                                                .operatorScale
                                        : 1.0,
                                rayleigh,
                                &physicalVectorNorms[i],
                                workspace);
                physicalValues[i] =
                        params.exponential.enabled
                                ? suppliedEigenvalue
                                : rayleigh;
            }

            std::vector<int> physicalOrder(keep);
            std::iota(
                    physicalOrder.begin(),
                    physicalOrder.end(),
                    0);
            std::stable_sort(
                    physicalOrder.begin(),
                    physicalOrder.end(),
                    [&](const int left, const int right) {
                        if (physicalValues[left]
                            == physicalValues[right]) {
                            return left < right;
                        }
                        return physicalValues[left]
                                < physicalValues[right];
                    });

            const int available =
                    std::min(requestedEigenpairs, keep);
            double largestScaledPhysicalResidual = 0.0;
            double densecodeCombinedResidual = 0.0;
            double maximumAbsoluteResidual = 0.0;
            double maximumRelativeResidual = 0.0;
            std::vector<double> absoluteResiduals(
                    available, 0.0);
            std::vector<double> relativeResiduals(
                    available, 0.0);
            // physicalOrder starts with the requested low modes. Retained
            // thick-restart guard vectors are deliberately excluded here.
            for (int i = 0; i < available; ++i) {
                const int index = physicalOrder[i];
                const double allowed =
                        residualTolerance
                        * std::max(
                                1.0,
                                std::fabs(
                                        physicalValues[index]));
                largestScaledPhysicalResidual =
                        std::max(
                                largestScaledPhysicalResidual,
                                physicalResiduals[index]
                                / std::max(
                                        allowed,
                                        std::numeric_limits<
                                                double>::min()));
                const double vectorNorm =
                        physicalVectorNorms[index];
                const double absoluteResidual =
                        physicalResiduals[index] * vectorNorm;
                absoluteResiduals[i] = absoluteResidual;
                densecodeCombinedResidual =
                        std::hypot(
                                densecodeCombinedResidual,
                                absoluteResidual);
                maximumAbsoluteResidual =
                        std::max(
                                maximumAbsoluteResidual,
                                absoluteResidual);

                const double operatorEigenvalue =
                        params.exponential.enabled
                                ? params.exponential.operatorShift
                                        - physicalValues[index]
                                                / params.exponential
                                                          .operatorScale
                                : physicalValues[index];
                const double relativeScale =
                        std::fabs(operatorEigenvalue)
                        * vectorNorm;
                const double relativeResidual =
                        relativeScale
                                        > std::numeric_limits<
                                                double>::min()
                                ? absoluteResidual / relativeScale
                                : (absoluteResidual == 0.0
                                        ? 0.0
                                        : std::numeric_limits<
                                                  double>::infinity());
                relativeResiduals[i] = relativeResidual;
                maximumRelativeResidual =
                        std::max(
                                maximumRelativeResidual,
                                relativeResidual);
            }
            const bool maximumScaledCriterionSatisfied =
                    available == requestedEigenpairs
                    && largestScaledPhysicalResidual <= 1.0;
            const bool densecodeCriterionSatisfied =
                    available == requestedEigenpairs
                    && densecodeCombinedResidual
                            <= params.residualTol;
            switch (params.convergenceCriterion) {
                case TRLanConvergenceCriterion::
                        MaximumScaledPerMode:
                    converged =
                            maximumScaledCriterionSatisfied;
                    break;
                case TRLanConvergenceCriterion::
                        DensecodeAggregatePhysical:
                    converged =
                            densecodeCriterionSatisfied;
                    break;
            }
            rootLogger.info(
                    "TRLan restart ", restart,
                    ": activeCriterion=",
                    trlanConvergenceCriterionName(
                            params.convergenceCriterion),
                    " maximumScaledPerMode=",
                    largestScaledPhysicalResidual,
                    " densecodeAggregatePhysical=",
                    densecodeCombinedResidual,
                    " converged=", converged);

            if (params.convergenceDiagnostics) {
                const double rmsResidual =
                        available > 0
                                ? densecodeCombinedResidual
                                        / std::sqrt(
                                                static_cast<double>(
                                                        available))
                                : std::numeric_limits<
                                          double>::infinity();

                rootLogger.info(
                        "=== TRLan convergence diagnostics ===");
                rootLogger.info("restart = ", restart);
                rootLogger.info(
                        "active criterion = ",
                        trlanConvergenceCriterionName(
                                params.convergenceCriterion));
                rootLogger.info(
                        "MaximumScaledPerMode criterion = max_i "
                        "((||r_i||/||q_i||) / "
                        "(residualTol_eff*max(1,|lambda_i|))) "
                        "<= 1");
                rootLogger.info(
                        "MaximumScaledPerMode criterion value = ",
                        largestScaledPhysicalResidual,
                        " satisfied = ",
                        maximumScaledCriterionSatisfied);
                rootLogger.info(
                        "Densecode combined residual = ",
                        densecodeCombinedResidual,
                        " threshold = ", params.residualTol,
                        " satisfied = ",
                        densecodeCriterionSatisfied);
                rootLogger.info(
                        "DensecodeAggregatePhysical criterion = "
                        "sqrt(sum_i ||A q_i - lambda_i q_i||^2) "
                        "<= residualTol");
                rootLogger.info(
                        "maximum individual absolute residual = ",
                        maximumAbsoluteResidual);
                rootLogger.info(
                        "maximum individual relative residual "
                        "(||r_i||/(|lambda_A_i|*||q_i||)) = ",
                        maximumRelativeResidual);
                rootLogger.info(
                        "RMS residual = ", rmsResidual);
                rootLogger.info(
                        "number of requested modes included = ",
                        available,
                        " of ", requestedEigenpairs);
                for (int i = 0; i < available; ++i) {
                    const int index = physicalOrder[i];
                    rootLogger.info(
                            "mode ", i,
                            ": physicalEigenvalue=",
                            physicalValues[index],
                            " absoluteResidual=",
                            absoluteResiduals[i],
                            " relativeResidual=",
                            relativeResiduals[i],
                            " vectorNorm=",
                            physicalVectorNorms[index]);
                }
                rootLogger.info(
                        "=== end TRLan convergence diagnostics ===");
            }

            finalSlots.assign(
                    physicalOrder.begin(),
                    physicalOrder.begin() + available);
            finalEigenvalues.resize(available);
            finalResiduals.resize(available);
            for (int i = 0; i < available; ++i) {
                const int index = finalSlots[i];
                finalEigenvalues[i] =
                        physicalValues[index];
                finalResiduals[i] =
                        physicalResiduals[index];
            }
        }

        if (converged || finalRestart || stoppedByBreakdown) {
            break;
        }
        if (keep + 1 >= krylovDimension) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan retained space leaves no room for restart"));
        }

        for (std::vector<double> &row : projected) {
            std::fill(row.begin(), row.end(), 0.0);
        }
        for (int i = 0; i < keep; ++i) {
            projected[i][i] =
                    retainedFilteredValues[i];
            projected[i][keep] =
                    restartCouplings[i];
            projected[keep][i] =
                    restartCouplings[i];
        }
        basis.commitRotation();
        basis.store(keep, terminalVector);
        firstColumn = keep;
    }

    if (!converged && params.failOnNoConvergence) {
        rootLogger.error(
                "TRLan stopped after ",
                workspace.operatorApplications,
                " raw operator applications and ",
                basis.globalReductions(),
                " batched basis reductions");
        throw std::runtime_error(stdLogger.fatal(
                "TRLan did not reach the requested physical residual"));
    }
    if (finalSlots.empty()) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan stopped without a physical Ritz-pair check"));
    }
    basis.releaseMainStorage();

    const int outputCount =
            std::min(
                    requestedEigenpairs,
                    static_cast<int>(finalSlots.size()));
    eigenvectors.reserve(outputCount);
    eigenvalues.reserve(outputCount);
    for (int i = 0; i < outputCount; ++i) {
        eigenvectors.emplace_back(
                comm,
                "TRLan_eigenvector_"
                        + std::to_string(i));
        basis.load(finalSlots[i], eigenvectors.back());
        normalizeOrThrow(
                eigenvectors.back(),
                breakdownTolerance,
                "TRLan output Ritz vector has invalid norm");
        eigenvalues.push_back(finalEigenvalues[i]);
    }
    const double largestOutputResidual =
            finalResiduals.empty()
                    ? std::numeric_limits<double>::infinity()
                    : *std::max_element(
                            finalResiduals.begin(),
                            finalResiduals.end());
    if (params.mpiConsistencyDiagnostics) {
        rootLogger.info(
                "=== TRLan MPI consistency diagnostics ===");
        rootLogger.info(
                "restart cycles checked = ",
                mpiConsistency.cyclesChecked);
        rootLogger.info(
                "projected matrix max rank difference = ",
                mpiConsistency.projectedMatrixMaxDifference);
        rootLogger.info(
                "Ritz eigenvalue max rank difference = ",
                mpiConsistency.ritzEigenvalueMaxDifference);
        rootLogger.info(
                "rotation matrix max rank difference = ",
                mpiConsistency.rotationMatrixMaxDifference);
        rootLogger.info(
                "=== end TRLan MPI consistency diagnostics ===");
    }
    rootLogger.info(
            "TRLan complete: eigenpairs=", outputCount,
            " rawOperatorApplications=",
            workspace.operatorApplications,
            " basisGlobalReductions=",
            basis.globalReductions(),
            " basisRotations=", basis.rotations(),
            " largestPhysicalResidual=",
            largestOutputResidual,
            " activeConvergenceCriterion=",
            trlanConvergenceCriterionName(
                    params.convergenceCriterion));
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::validateParams(
        const TRLanRestartParams &params,
        const int requestedEigenpairs,
        const int krylovDimension,
        const int retainedDimension) {
    if (krylovDimension < requestedEigenpairs + 2) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan krylovDim must be at least requestedEigenpairs + 2"));
    }
    if (retainedDimension < requestedEigenpairs) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan thickRestartDim must retain all requested pairs"));
    }
    if (retainedDimension > krylovDimension - 2) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan thickRestartDim must be at most krylovDim - 2"));
    }
    if (params.maxRestarts < 0) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan maxRestarts cannot be negative"));
    }
    if (!std::isfinite(params.residualTol)
        || params.residualTol <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan residualTol must be positive and finite"));
    }
    switch (params.convergenceCriterion) {
        case TRLanConvergenceCriterion::
                MaximumScaledPerMode:
        case TRLanConvergenceCriterion::
                DensecodeAggregatePhysical:
            break;
        default:
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan convergence criterion is invalid"));
    }
    if (!std::isfinite(params.breakdownTol)
        || params.breakdownTol <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan breakdownTol must be positive and finite"));
    }
    if (params.reorthogonalizationPasses < 1
        || params.reorthogonalizationPasses > 4) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan reorthogonalizationPasses must be between 1 and 4"));
    }
    if (params.physicalCheckInterval < 0) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan physicalCheckInterval cannot be negative"));
    }
    if (params.chebyshev.enabled
        && params.exponential.enabled) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan Chebyshev and exponential filters are mutually exclusive"));
    }
    if (params.chebyshev.enabled) {
        if (params.chebyshev.order <= 0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan Chebyshev order must be positive"));
        }
        if (!std::isfinite(params.chebyshev.lowerBound)
            || !std::isfinite(params.chebyshev.upperBound)
            || params.chebyshev.upperBound
                    <= params.chebyshev.lowerBound) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan Chebyshev bounds must be finite and increasing"));
        }
    }
    if (params.exponential.enabled) {
        if (params.exponential.order <= 0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential order must be positive"));
        }
        if (!std::isfinite(params.exponential.alpha)
            || params.exponential.alpha <= 0.0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential alpha must be positive and finite"));
        }
        if (!std::isfinite(params.exponential.beta)
            || params.exponential.beta <= 0.0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential beta must be positive and finite"));
        }
        if (!std::isfinite(
                params.exponential.operatorShift)) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential operatorShift must be finite"));
        }
        if (!std::isfinite(
                    params.exponential.operatorScale)
            || (params.exponential.operatorScale != 1.0
                && params.exponential.operatorScale != -1.0)) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential operatorScale must be +1 or -1"));
        }
        const double stageCoefficient =
                params.exponential.alpha
                * params.exponential.beta
                * params.exponential.operatorScale
                / static_cast<double>(params.exponential.order);
        const double identityCoefficient =
                params.exponential.beta
                - stageCoefficient
                        * params.exponential.operatorShift;
        if (!std::isfinite(stageCoefficient)
            || !std::isfinite(identityCoefficient)) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential stage coefficients are not finite"));
        }
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::applyMdaggMSingle(
        LinearOperator<Spinor_external> &op,
        Spinor_internal &out,
        const Spinor_internal &in,
        OperatorWorkspace &workspace) {
    ++workspace.operatorApplications;
    if constexpr (NStacks == 1) {
        op.applyMdaggM(out, in, true);
    } else {
        for (size_t stack = 0; stack < NStacks; ++stack) {
            workspace.stackedInput.copyFromStackToStack(
                    in, stack, 0);
        }
        op.applyMdaggM(
                workspace.stackedOutput,
                workspace.stackedInput,
                true);
        out.copyFromStackToStack(
                workspace.stackedOutput, 0, 0);
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::applyFilteredOperator(
        LinearOperator<Spinor_external> &op,
        Spinor_internal &out,
        const Spinor_internal &in,
        const TRLanRestartParams &params,
        OperatorWorkspace &workspace) {
    if (params.exponential.enabled) {
        applyExponentialFilter(
                op,
                out,
                in,
                params.exponential,
                workspace);
    } else if (params.chebyshev.enabled) {
        applyChebyshevFilter(
                op,
                out,
                in,
                params.chebyshev,
                workspace);
    } else {
        applyMdaggMSingle(op, out, in, workspace);
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::applyChebyshevFilter(
        LinearOperator<Spinor_external> &op,
        Spinor_internal &out,
        const Spinor_internal &in,
        const TRLanChebyshevFilterParams &filter,
        OperatorWorkspace &workspace) {
    const double halfWidth =
            0.5 * (filter.upperBound - filter.lowerBound);
    const double center =
            0.5 * (filter.upperBound + filter.lowerBound);
    const double inverseHalfWidth = 1.0 / halfWidth;

    Spinor_internal *previous = &workspace.stage0;
    Spinor_internal *current = &workspace.stage1;
    Spinor_internal *next = &workspace.stage2;
    *previous = in;

    applyMdaggMSingle(
            op, *current, *previous, workspace);
    assignLinearCombination(
            *current,
            inverseHalfWidth,
            *current,
            -center * inverseHalfWidth,
            *previous);
    if (filter.order == 1) {
        out = *current;
        return;
    }

    for (int degree = 2;
         degree <= filter.order;
         ++degree) {
        applyMdaggMSingle(
                op, *next, *current, workspace);
        assignLinearCombination(
                *next,
                2.0 * inverseHalfWidth,
                *next,
                -2.0 * center * inverseHalfWidth,
                *current);
        next->axpyThis(
                static_cast<floatT>(-1.0),
                *previous);
        Spinor_internal *oldPrevious = previous;
        previous = current;
        current = next;
        next = oldPrevious;
    }
    out = *current;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::applyExponentialFilter(
        LinearOperator<Spinor_external> &op,
        Spinor_internal &out,
        const Spinor_internal &in,
        const TRLanExponentialFilterParams &filter,
        OperatorWorkspace &workspace) {
    const double operatorCoefficient =
            filter.alpha * filter.beta
            * filter.operatorScale
            / static_cast<double>(filter.order);
    const double identityCoefficient =
            filter.beta
            - operatorCoefficient * filter.operatorShift;

    Spinor_internal *current = &workspace.stage0;
    Spinor_internal *applied = &workspace.stage1;
    Spinor_internal *next = &workspace.stage2;
    *current = in;
    for (int stage = 0; stage < filter.order; ++stage) {
        applyMdaggMSingle(
                op, *applied, *current, workspace);
        assignLinearCombination(
                *next,
                identityCoefficient,
                *current,
                operatorCoefficient,
                *applied);
        std::swap(current, next);
    }
    out = *current;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
int TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::extendLanczosFactorization(
        LinearOperator<Spinor_external> &op,
        Basis &basis,
        const int firstColumn,
        const int targetDimension,
        const double breakdownTol,
        const int reorthogonalizationPasses,
        const TRLanRestartParams &params,
        OperatorWorkspace &workspace,
        Spinor_internal &terminalVector,
        double &terminalBeta,
        std::vector<std::vector<double>> &projected) {
    Spinor_internal &basisVector = workspace.basisVector;
    Spinor_internal &applied = workspace.filteredOutput;

    terminalBeta = 0.0;
    for (int column = firstColumn;
         column < targetDimension;
         ++column) {
        basis.load(column, basisVector);
        applyFilteredOperator(
                op,
                applied,
                basisVector,
                params,
                workspace);

        std::vector<COMPLEX(double)> projections =
                basis.dot(applied, column + 1);
        for (const COMPLEX(double) projection : projections) {
            if (!std::isfinite(projection.cREAL)
                || !std::isfinite(projection.cIMAG)) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan projection contains a non-finite value"));
            }
        }

        projected[column][column] =
                projections[column].cREAL;
        if (column == firstColumn && firstColumn > 0) {
            for (int row = 0; row < column; ++row) {
                projected[row][column] =
                        projections[row].cREAL;
                projected[column][row] =
                        projections[row].cREAL;
            }
        }

        basis.subtractCombination(applied, projections);
        basis.orthogonalize(
                applied,
                column + 1,
                reorthogonalizationPasses - 1);

        const double normSquared =
                applied.realdotProduct(applied);
        if (!std::isfinite(normSquared)
            || normSquared < 0.0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan generated a non-finite residual norm"));
        }
        terminalBeta = std::sqrt(normSquared);
        if (terminalBeta <= breakdownTol) {
            terminalBeta = 0.0;
            return column + 1;
        }

        applied *= COMPLEX(floatT)(
                static_cast<floatT>(1.0 / terminalBeta),
                static_cast<floatT>(0.0));
        if (column + 1 < targetDimension) {
            projected[column][column + 1] =
                    terminalBeta;
            projected[column + 1][column] =
                    terminalBeta;
            basis.store(column + 1, applied);
        } else {
            terminalVector = applied;
        }
    }
    return targetDimension;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::normalizeOrThrow(
        Spinor_internal &vector,
        const double breakdownTol,
        const char *errorMessage) {
    const double normSquared =
            vector.realdotProduct(vector);
    if (!std::isfinite(normSquared)
        || normSquared <= breakdownTol * breakdownTol) {
        throw std::runtime_error(stdLogger.fatal(errorMessage));
    }
    vector *= COMPLEX(floatT)(
            static_cast<floatT>(
                    1.0 / std::sqrt(normSquared)),
            static_cast<floatT>(0.0));
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::diagonalizeProjected(
        const std::vector<std::vector<double>> &projected,
        const int dimension,
        std::vector<double> &eigenvalues,
        std::vector<std::vector<double>> &eigenvectors) {
    bool isTridiagonal = true;
    for (int row = 0; row < dimension && isTridiagonal; ++row) {
        for (int column = 0; column < dimension; ++column) {
            if (std::abs(row - column) > 1
                && projected[row][column] != 0.0) {
                isTridiagonal = false;
                break;
            }
        }
    }

    std::vector<std::vector<double>> work(
            dimension + 1,
            std::vector<double>(dimension + 1, 0.0));
    std::vector<double> diagonal(
            dimension + 1, 0.0);
    std::vector<double> offDiagonal(
            dimension + 1, 0.0);

    if (isTridiagonal) {
        for (int row = 0; row < dimension; ++row) {
            const double entry = projected[row][row];
            if (!std::isfinite(entry)) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan projected matrix contains a non-finite value"));
            }
            diagonal[row + 1] = entry;
            work[row + 1][row + 1] = 1.0;
        }
        for (int row = 1; row < dimension; ++row) {
            const double lower = projected[row][row - 1];
            const double upper = projected[row - 1][row];
            if (!std::isfinite(lower) || !std::isfinite(upper)) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan projected matrix contains a non-finite value"));
            }
            offDiagonal[row + 1] =
                    0.5 * (lower + upper);
        }
    } else {
        for (int row = 0; row < dimension; ++row) {
            for (int column = 0;
                 column < dimension;
                 ++column) {
                const double left = projected[row][column];
                const double right = projected[column][row];
                if (!std::isfinite(left)
                    || !std::isfinite(right)) {
                    throw std::runtime_error(stdLogger.fatal(
                            "TRLan projected matrix contains a non-finite value"));
                }
                work[row + 1][column + 1] =
                        0.5 * (left + right);
            }
        }
        householderTridiagonalize(
                work,
                dimension,
                diagonal,
                offDiagonal);
    }
    tridiagonalQL(
            diagonal,
            offDiagonal,
            dimension,
            work);

    eigenvalues.assign(dimension, 0.0);
    eigenvectors.assign(
            dimension,
            std::vector<double>(dimension, 0.0));
    for (int column = 0;
         column < dimension;
         ++column) {
        eigenvalues[column] =
                diagonal[column + 1];
        if (!std::isfinite(eigenvalues[column])) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan projected eigensolver returned a non-finite value"));
        }
        for (int row = 0; row < dimension; ++row) {
            eigenvectors[row][column] =
                    work[row + 1][column + 1];
        }
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
std::vector<int> TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::targetOrder(
        const std::vector<double> &filteredEigenvalues,
        const TRLanRestartParams &params) {
    std::vector<int> order(filteredEigenvalues.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
            order.begin(),
            order.end(),
            [&](const int left, const int right) {
                double leftKey = filteredEigenvalues[left];
                double rightKey = filteredEigenvalues[right];
                if (params.chebyshev.enabled) {
                    leftKey = std::fabs(leftKey);
                    rightKey = std::fabs(rightKey);
                }
                if (leftKey == rightKey) {
                    return left < right;
                }
                if (params.exponential.enabled
                    || params.chebyshev.enabled) {
                    return leftKey > rightKey;
                }
                return leftKey < rightKey;
            });
    return order;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::correctedExponentialEigenvalue(
        const double filteredEigenvalue,
        const TRLanExponentialFilterParams &filter) {
    if (!std::isfinite(filteredEigenvalue)) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan cannot correct a non-finite filtered eigenvalue"));
    }

    double root = 0.0;
    if (filteredEigenvalue < 0.0) {
        if (filter.order % 2 == 0) {
            const double roundoffTolerance =
                    64.0
                    * static_cast<double>(
                            std::numeric_limits<floatT>::epsilon());
            if (filteredEigenvalue < -roundoffTolerance) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan even exponential order produced a materially negative Ritz value"));
            }
            root = 0.0;
        } else {
            root = -std::pow(
                    -filteredEigenvalue,
                    1.0 / static_cast<double>(filter.order));
        }
    } else if (filteredEigenvalue == 0.0) {
        root = 0.0;
    } else {
        const double logRatio =
                std::log(filteredEigenvalue)
                        / static_cast<double>(filter.order)
                - std::log(filter.beta);
        const double corrected =
                -static_cast<double>(filter.order)
                * std::expm1(logRatio)
                / filter.alpha;
        if (!std::isfinite(corrected)) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan exponential correction is non-finite"));
        }
        return corrected;
    }

    const double corrected =
            static_cast<double>(filter.order)
            * (filter.beta - root)
            / (filter.alpha * filter.beta);
    if (!std::isfinite(corrected)) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan exponential correction is non-finite"));
    }
    return corrected;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::exponentialDerivativeMagnitude(
        const double filteredEigenvalue,
        const TRLanExponentialFilterParams &filter) {
    const double exponent =
            static_cast<double>(filter.order - 1)
            / static_cast<double>(filter.order);
    return filter.alpha * filter.beta
            * std::pow(
                    std::fabs(filteredEigenvalue),
                    exponent);
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
bool TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::inexpensiveConvergenceGate(
        const std::vector<double> &filteredEigenvalues,
        const std::vector<double> &filteredResiduals,
        const int requestedEigenpairs,
        const double residualTol,
        const TRLanRestartParams &params) {
    if (static_cast<int>(filteredEigenvalues.size())
            < requestedEigenpairs
        || filteredResiduals.size()
                < filteredEigenvalues.size()) {
        return false;
    }

    constexpr double gateSlack = 10.0;
    for (int i = 0; i < requestedEigenpairs; ++i) {
        double expectedPhysicalEigenvalue =
                filteredEigenvalues[i];
        double derivative = 1.0;
        if (params.exponential.enabled) {
            expectedPhysicalEigenvalue =
                    correctedExponentialEigenvalue(
                            filteredEigenvalues[i],
                            params.exponential);
            derivative =
                    exponentialDerivativeMagnitude(
                            filteredEigenvalues[i],
                            params.exponential);
        }
        const double allowedPhysical =
                residualTol
                * std::max(
                        1.0,
                        std::fabs(expectedPhysicalEigenvalue));
        const double allowedFiltered =
                gateSlack
                * std::max(
                        derivative,
                        std::numeric_limits<double>::epsilon())
                * allowedPhysical;
        if (!std::isfinite(filteredResiduals[i])
            || filteredResiduals[i] > allowedFiltered) {
            return false;
        }
    }
    return true;
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::physicalResidual(
        LinearOperator<Spinor_external> &op,
        Spinor_internal &vector,
        const double physicalEigenvalue,
        const bool densecodePositiveConvention,
        const double operatorShift,
        const double operatorScale,
        double &rayleighQuotient,
        double *vectorNorm,
        OperatorWorkspace &workspace) {
    Spinor_internal &applied = workspace.stage2;
    applyMdaggMSingle(op, applied, vector, workspace);
    const COMPLEX(double) numerator =
            vector.dotProduct(applied);
    const double denominator =
            vector.realdotProduct(vector);
    if (!std::isfinite(numerator.cREAL)
        || !std::isfinite(numerator.cIMAG)
        || !std::isfinite(denominator)
        || denominator
                <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan cannot validate a zero or non-finite Ritz vector"));
    }
    rayleighQuotient = numerator.cREAL / denominator;
    if (vectorNorm != nullptr) {
        *vectorNorm = std::sqrt(denominator);
    }

    double expectedOperatorEigenvalue =
            physicalEigenvalue;
    if (densecodePositiveConvention) {
        expectedOperatorEigenvalue =
                operatorShift
                - physicalEigenvalue / operatorScale;
    } else if (!std::isfinite(expectedOperatorEigenvalue)) {
        expectedOperatorEigenvalue = rayleighQuotient;
    }
    applied.axpyThis(
            static_cast<floatT>(
                    -expectedOperatorEigenvalue),
            vector);
    const double residualSquared =
            applied.realdotProduct(applied);
    if (!std::isfinite(residualSquared)
        || residualSquared < 0.0) {
        throw std::runtime_error(stdLogger.fatal(
                "TRLan physical residual is non-finite"));
    }
    return std::sqrt(residualSquared / denominator);
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::householderTridiagonalize(
        std::vector<std::vector<double>> &matrix,
        const int dimension,
        std::vector<double> &diagonal,
        std::vector<double> &offDiagonal) {
    for (int i = dimension; i >= 2; --i) {
        const int l = i - 1;
        double scale = 0.0;
        double h = 0.0;
        if (l > 1) {
            for (int k = 1; k <= l; ++k) {
                scale += std::fabs(matrix[i][k]);
            }
            if (scale == 0.0) {
                offDiagonal[i] = matrix[i][l];
            } else {
                for (int k = 1; k <= l; ++k) {
                    matrix[i][k] /= scale;
                    h += matrix[i][k] * matrix[i][k];
                }
                double f = matrix[i][l];
                double g =
                        f >= 0.0
                                ? -std::sqrt(h)
                                : std::sqrt(h);
                offDiagonal[i] = scale * g;
                h -= f * g;
                matrix[i][l] = f - g;
                f = 0.0;
                for (int j = 1; j <= l; ++j) {
                    matrix[j][i] = matrix[i][j] / h;
                    g = 0.0;
                    for (int k = 1; k <= j; ++k) {
                        g += matrix[j][k]
                                * matrix[i][k];
                    }
                    for (int k = j + 1;
                         k <= l;
                         ++k) {
                        g += matrix[k][j]
                                * matrix[i][k];
                    }
                    offDiagonal[j] = g / h;
                    f += offDiagonal[j]
                            * matrix[i][j];
                }
                const double hh = f / (h + h);
                for (int j = 1; j <= l; ++j) {
                    f = matrix[i][j];
                    const double g =
                            offDiagonal[j] - hh * f;
                    offDiagonal[j] = g;
                    for (int k = 1; k <= j; ++k) {
                        matrix[j][k] -=
                                f * offDiagonal[k]
                                + g * matrix[i][k];
                    }
                }
            }
        } else {
            offDiagonal[i] = matrix[i][l];
        }
        diagonal[i] = h;
    }
    diagonal[1] = 0.0;
    offDiagonal[1] = 0.0;

    for (int i = 1; i <= dimension; ++i) {
        const int l = i - 1;
        if (diagonal[i] != 0.0) {
            for (int j = 1; j <= l; ++j) {
                double g = 0.0;
                for (int k = 1; k <= l; ++k) {
                    g += matrix[i][k]
                            * matrix[k][j];
                }
                for (int k = 1; k <= l; ++k) {
                    matrix[k][j] -=
                            g * matrix[k][i];
                }
            }
        }
        diagonal[i] = matrix[i][i];
        matrix[i][i] = 1.0;
        for (int j = 1; j <= l; ++j) {
            matrix[j][i] = 0.0;
            matrix[i][j] = 0.0;
        }
    }
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
double TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::pythag(
        const double a,
        const double b) {
    const double absoluteA = std::fabs(a);
    const double absoluteB = std::fabs(b);
    if (absoluteA > absoluteB) {
        const double ratio = absoluteB / absoluteA;
        return absoluteA * std::sqrt(1.0 + ratio * ratio);
    }
    if (absoluteB == 0.0) {
        return 0.0;
    }
    const double ratio = absoluteA / absoluteB;
    return absoluteB * std::sqrt(1.0 + ratio * ratio);
}

template<class floatT, bool onDevice, Layout LatticeLayout,
         size_t HaloDepthSpin, size_t NStacks>
void TRLanSpinorSolver<
        floatT, onDevice, LatticeLayout,
        HaloDepthSpin, NStacks>::tridiagonalQL(
        std::vector<double> &diagonal,
        std::vector<double> &offDiagonal,
        const int dimension,
        std::vector<std::vector<double>> &eigenvectors) {
    for (int i = 2; i <= dimension; ++i) {
        offDiagonal[i - 1] = offDiagonal[i];
    }
    offDiagonal[dimension] = 0.0;

    const double epsilon =
            std::numeric_limits<double>::epsilon();
    for (int l = 1; l <= dimension; ++l) {
        int iteration = 0;
        int m = l;
        do {
            for (m = l; m <= dimension - 1; ++m) {
                const double scale =
                        std::fabs(diagonal[m])
                        + std::fabs(diagonal[m + 1]);
                if (std::fabs(offDiagonal[m])
                        <= epsilon * scale) {
                    break;
                }
            }
            if (m == l) {
                continue;
            }
            if (iteration++ >= 100) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan tridiagonal QL iteration did not converge"));
            }

            double g =
                    (diagonal[l + 1] - diagonal[l])
                    / (2.0 * offDiagonal[l]);
            double r = pythag(g, 1.0);
            g = diagonal[m] - diagonal[l]
                    + offDiagonal[l]
                            / (g + (g >= 0.0
                                    ? std::fabs(r)
                                    : -std::fabs(r)));
            double sine = 1.0;
            double cosine = 1.0;
            double shift = 0.0;
            int i = m - 1;
            for (; i >= l; --i) {
                const double f =
                        sine * offDiagonal[i];
                const double b =
                        cosine * offDiagonal[i];
                r = pythag(f, g);
                offDiagonal[i + 1] = r;
                if (r == 0.0) {
                    diagonal[i + 1] -= shift;
                    offDiagonal[m] = 0.0;
                    break;
                }
                sine = f / r;
                cosine = g / r;
                g = diagonal[i + 1] - shift;
                r = (diagonal[i] - g) * sine
                        + 2.0 * cosine * b;
                shift = sine * r;
                diagonal[i + 1] = g + shift;
                g = cosine * r - b;
                for (int k = 1;
                     k <= dimension;
                     ++k) {
                    const double upper =
                            eigenvectors[k][i + 1];
                    const double lower =
                            eigenvectors[k][i];
                    eigenvectors[k][i + 1] =
                            sine * lower
                            + cosine * upper;
                    eigenvectors[k][i] =
                            cosine * lower
                            - sine * upper;
                }
            }
            if (r == 0.0 && i >= l) {
                continue;
            }
            diagonal[l] -= shift;
            offDiagonal[l] = g;
            offDiagonal[m] = 0.0;
        } while (m != l);
    }
}

#define LANCZOS_INIT_PLHHSN( \
        floatT, LO, HaloDepth, HaloDepthSpin, STACKS) \
template class TRLanSpinorSolver< \
        floatT, false, LO, HaloDepthSpin, STACKS>; \
template class TRLanSpinorSolver< \
        floatT, true, LO, HaloDepthSpin, STACKS>;
INIT_PLHHSN(LANCZOS_INIT_PLHHSN)
