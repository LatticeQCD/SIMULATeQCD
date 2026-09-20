#pragma once

#include "../../base/latticeContainer.h"
#include "../../base/math/vect3array.h"
#include "../../base/wrapper/gpu_wrapper.h"
#include "../../define.h"
#include "../../spinor/spinorfield.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trlan_detail {

constexpr unsigned int dotBlockSize = 128;
constexpr unsigned int rotateSiteTile = 16;
constexpr unsigned int rotateVectorTile = 8;

template<class floatT, Layout LatticeLayout, size_t HaloDepthSpin>
__host__ __device__ inline gSite basisSite(
        const size_t bulkSite,
        const size_t vector,
        const size_t fullVolume) {
    gSite site = GIndexer<LatticeLayout, HaloDepthSpin>::getSite(bulkSite);
    site.isiteFull += vector * fullVolume;
    return site;
}

template<class floatT, Layout LatticeLayout, size_t HaloDepthSpin>
__host__ __device__ inline gSite basisFullSite(
        const size_t fullSite,
        const size_t vector,
        const size_t fullVolume) {
    gSite site;
    site.isiteFull = fullSite + vector * fullVolume;
    return site;
}

// TEMPORARY DIAGNOSTIC (2026-09-18): mirrors TRLanLinearCombination
// (lanczos.cpp) but subtracts one complex-scaled stored basis vector from
// `output` in place, via the generic (already-validated) iterateOverFull
// path instead of the hand-rolled subtractBasisCombinationKernel below.
template<class floatT>
struct TRLanSubtractSingleVector {
    Vect3arrayAcc<floatT> outputAcc;
    Vect3arrayAcc<floatT> storedAcc;
    COMPLEX(floatT) coefficient;

    TRLanSubtractSingleVector(
            Vect3arrayAcc<floatT> outputAcc,
            Vect3arrayAcc<floatT> storedAcc,
            COMPLEX(floatT) coefficient)
        : outputAcc(outputAcc), storedAcc(storedAcc), coefficient(coefficient) {}

    __host__ __device__ Vect3<floatT> operator()(gSiteStack &site) const {
        return outputAcc.getElement(site)
                - coefficient * storedAcc.getElement(site);
    }
};

#ifdef __GPUCC__

template<class floatT, Layout LatticeLayout, size_t HaloDepthSpin>
__global__ void basisDotPartialKernel(
        const Vect3arrayAcc<floatT> basis,
        const Vect3arrayAcc<floatT> vector,
        LatticeContainerAccessor partial,
        const size_t bulkVolume,
        const size_t fullVolume) {
    const size_t basisVector = blockIdx.y;
    COMPLEX(double) local(0.0, 0.0);

    for (size_t bulkSite = blockIdx.x * blockDim.x + threadIdx.x;
         bulkSite < bulkVolume;
         bulkSite += gridDim.x * blockDim.x) {
        const gSite vectorSite =
                GIndexer<LatticeLayout, HaloDepthSpin>::getSite(bulkSite);
        const gSite storedSite =
                basisSite<floatT, LatticeLayout, HaloDepthSpin>(
                        bulkSite, basisVector, fullVolume);
        const COMPLEX(floatT) product =
                basis.getElement(storedSite)
                * vector.getElement(vectorSite);
        local += COMPLEX(double)(
                static_cast<double>(product.cREAL),
                static_cast<double>(product.cIMAG));
    }

    __shared__ COMPLEX(double) blockSum[dotBlockSize];
    blockSum[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial.setElement(
                basisVector * gridDim.x + blockIdx.x,
                blockSum[0]);
    }
}

template<class floatT, Layout LatticeLayout, size_t HaloDepthSpin>
__global__ void subtractBasisCombinationKernel(
        Vect3arrayAcc<floatT> vector,
        const Vect3arrayAcc<floatT> basis,
        const COMPLEX(double) *coefficients,
        const size_t vectorCount,
        const size_t fullVolume) {
    const size_t fullSite = blockIdx.x * blockDim.x + threadIdx.x;
    if (fullSite >= fullVolume) {
        return;
    }

    gSite vectorSite;
    vectorSite.isiteFull = fullSite;
    Vect3<floatT> value = vector.getElement(vectorSite);
    for (size_t j = 0; j < vectorCount; ++j) {
        const gSite storedSite =
                basisFullSite<floatT, LatticeLayout, HaloDepthSpin>(
                        fullSite, j, fullVolume);
        const COMPLEX(floatT) coefficient(
                static_cast<floatT>(coefficients[j].cREAL),
                static_cast<floatT>(coefficients[j].cIMAG));
        value -= coefficient * basis.getElement(storedSite);
    }
    vector.setElement(vectorSite, value);
}

template<class floatT, Layout LatticeLayout, size_t HaloDepthSpin>
__global__ void rotateBasisKernel(
        const Vect3arrayAcc<floatT> source,
        Vect3arrayAcc<floatT> destination,
        const double *coefficients,
        const size_t sourceCount,
        const size_t destinationCount,
        const size_t fullVolume) {
    const size_t fullSite =
            blockIdx.x * rotateSiteTile + threadIdx.x;
    const size_t destinationVector =
            blockIdx.y * rotateVectorTile + threadIdx.y;

    __shared__ Vect3<floatT>
            sourceTile[rotateVectorTile][rotateSiteTile];
    __shared__ floatT
            coefficientTile[rotateVectorTile][rotateVectorTile];

    Vect3<floatT> sum(static_cast<floatT>(0.0));
    for (size_t tile = 0; tile < sourceCount;
         tile += rotateVectorTile) {
        const size_t sourceVector = tile + threadIdx.y;
        if (fullSite < fullVolume && sourceVector < sourceCount) {
            const gSite storedSite =
                    basisFullSite<floatT, LatticeLayout, HaloDepthSpin>(
                            fullSite, sourceVector, fullVolume);
            sourceTile[threadIdx.y][threadIdx.x] =
                    source.getElement(storedSite);
        } else {
            sourceTile[threadIdx.y][threadIdx.x] =
                    Vect3<floatT>(static_cast<floatT>(0.0));
        }

        if (threadIdx.x < rotateVectorTile) {
            const size_t coefficientVector = tile + threadIdx.x;
            coefficientTile[threadIdx.y][threadIdx.x] =
                    (destinationVector < destinationCount
                     && coefficientVector < sourceCount)
                            ? static_cast<floatT>(
                                    coefficients[
                                            coefficientVector
                                                    * destinationCount
                                            + destinationVector])
                            : static_cast<floatT>(0.0);
        }
        __syncthreads();

        if (fullSite < fullVolume
            && destinationVector < destinationCount) {
            const size_t remaining = sourceCount - tile;
            const size_t active =
                    remaining < rotateVectorTile
                            ? remaining
                            : rotateVectorTile;
            for (size_t j = 0; j < active; ++j) {
                sum += coefficientTile[threadIdx.y][j]
                        * sourceTile[j][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (fullSite < fullVolume
        && destinationVector < destinationCount) {
        const gSite outputSite =
                basisFullSite<floatT, LatticeLayout, HaloDepthSpin>(
                        fullSite, destinationVector, fullVolume);
        destination.setElement(outputSite, sum);
    }
}

#endif

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin>
class Basis {
public:
    using Spinor =
            Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, 1>;

    Basis(
            CommunicationBase &comm,
            const size_t capacity,
            const size_t rotationCapacity)
        : _comm(comm),
          _capacity(capacity),
          _rotationCapacity(rotationCapacity),
          _bulkVolume(
                  LatticeLayout == Layout::All
                          ? GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getLatData().vol4
                          : GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getLatData().sizeh),
          _fullVolume(
                  LatticeLayout == Layout::All
                          ? GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getLatData().vol4Full
                          : GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getLatData().sizehFull),
          _vectors(std::make_unique<
                  Vect3array<floatT, onDevice>>(
                          checkedElementCount(
                                  capacity, _fullVolume),
                          "TRLanBasis_vectors")),
          _rotationScratch(
                  checkedElementCount(
                          rotationCapacity, _fullVolume),
                  "TRLanBasis_rotationScratch"),
          _partialDots(
                  comm,
                  "TRLanBasis_partialDots",
                  "TRLanBasis_reduceHelp",
                  "TRLanBasis_reduceResult",
                  "TRLanBasis_reduceHost"),
          _coefficients(
                  MemoryManagement::getMemAt<onDevice>(
                          "TRLanBasis_coefficients")),
          _hostCoefficients(
                  MemoryManagement::getMemAt<false>(
                          "TRLanBasis_hostCoefficients")),
          _globalReductions(0),
          _rotations(0),
          _rotationIsActive(false),
          _rotatedVectorCount(0) {
        if (capacity == 0
            || rotationCapacity == 0
            || rotationCapacity > capacity
            || _bulkVolume == 0
            || _fullVolume == 0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis dimensions are invalid"));
        }
    }

    size_t capacity() const {
        return _capacity;
    }

    static size_t requiredStorageBytes(
            const size_t capacity,
            const size_t rotationCapacity) {
        const size_t volume =
                LatticeLayout == Layout::All
                        ? GIndexer<LatticeLayout, HaloDepthSpin>
                                  ::getLatData().vol4Full
                        : GIndexer<LatticeLayout, HaloDepthSpin>
                                  ::getLatData().sizehFull;
        const size_t basisElements =
                checkedElementCount(capacity, volume);
        const size_t rotationElements =
                checkedElementCount(rotationCapacity, volume);
        const size_t elements =
                checkedSum(basisElements, rotationElements);
        return elements * 3 * sizeof(COMPLEX(floatT));
    }

    size_t storageBytes() const {
        return requiredStorageBytes(
                _capacity, _rotationCapacity);
    }

    uint64_t globalReductions() const {
        return _globalReductions;
    }

    uint64_t rotations() const {
        return _rotations;
    }

    void store(const size_t index, const Spinor &vector) {
        checkIndex(index);
        requireMainStorage();
        if (_rotationIsActive) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan cannot store into an uncommitted rotated basis"));
        }
        _vectors->copyFromPartial(
                vector.getArray(),
                _fullVolume,
                index * _fullVolume,
                0);
    }

    void load(const size_t index, Spinor &vector) const {
        checkIndex(index);
        if (_rotationIsActive) {
            if (index >= _rotatedVectorCount) {
                throw std::runtime_error(stdLogger.fatal(
                        "TRLan rotated basis index is out of range"));
            }
            vector.copyFromArray(
                    _rotationScratch,
                    index * _fullVolume);
        } else {
            requireMainStorage();
            vector.copyFromArray(
                    *_vectors,
                    index * _fullVolume);
        }
    }

    std::vector<COMPLEX(double)> dot(
            const Spinor &vector,
            const size_t vectorCount) {
        checkCount(vectorCount);
        std::vector<COMPLEX(double)> result(
                vectorCount, COMPLEX(double)(0.0, 0.0));
        if (vectorCount == 0) {
            return result;
        }
        requireUnrotatedMainStorage();

        if constexpr (onDevice) {
#ifdef __GPUCC__
            const size_t requestedBlocks =
                    (_bulkVolume + dotBlockSize - 1) / dotBlockSize;
            const unsigned int partialBlockCount =
                    static_cast<unsigned int>(
                            std::max<size_t>(
                                    1, std::min<size_t>(256, requestedBlocks)));
            _partialDots.adjustSize(
                    vectorCount * partialBlockCount);

            const dim3 block(dotBlockSize);
            const dim3 grid(
                    partialBlockCount,
                    static_cast<unsigned int>(vectorCount));
#ifdef USE_CUDA
            basisDotPartialKernel<
                    floatT, LatticeLayout, HaloDepthSpin>
                    <<<grid, block>>>(
                            _vectors->getAccessor(),
                            vector.getAccessor(),
                            _partialDots.getAccessor(),
                            _bulkVolume,
                            _fullVolume);
#elif defined USE_HIP
            hipLaunchKernelGGL(
                    (basisDotPartialKernel<
                            floatT, LatticeLayout, HaloDepthSpin>),
                    grid,
                    block,
                    0,
                    0,
                    _vectors->getAccessor(),
                    vector.getAccessor(),
                    _partialDots.getAccessor(),
                    _bulkVolume,
                    _fullVolume);
#endif
            checkLastKernel("TRLan basis dot product");

            // TEMPORARY DIAGNOSTIC (2026-09-20): a projection magnitude of
            // ~1e179 was observed at column 0 -- wildly inconsistent with
            // the field magnitudes seen elsewhere (storedVector ~1e-4,
            // applied ~1e6-1e7, so a legitimate sum over ~2e6 sites should
            // land around 1e8-1e9, not 1e179). That points at the device
            // reduction pipeline (this kernel's per-block partial sums, or
            // the CUB DeviceSegmentedReduce inside reduceStacked) returning
            // garbage rather than a genuine numerical overflow. Dump every
            // per-block partial sum here, before the segmented reduce, so
            // we can tell whether the garbage is already present per-block
            // (kernel/shared-memory bug) or only appears after
            // reduceStacked (buffer-sizing/indexing bug in the CUB call).
            {
                const size_t diagCount =
                        static_cast<size_t>(vectorCount)
                        * static_cast<size_t>(partialBlockCount);
                auto diagHost = MemoryManagement::getMemAt<false>(
                        "TRLanDiag_partialDotsHost");
                diagHost->template adjustSize<COMPLEX(double)>(diagCount);
                diagHost->copyFrom(
                        _partialDots.getMemPointer(),
                        diagCount * sizeof(COMPLEX(double)));
                LatticeContainerAccessor diagAcc(diagHost->getPointer());
                for (size_t idx = 0; idx < diagCount; ++idx) {
                    COMPLEX(double) value;
                    diagAcc.getValue(idx, value);
                    const double magnitude =
                            std::hypot(value.cREAL, value.cIMAG);
                    std::fprintf(stderr,
                            "DIAG lanczosKernels.h: dot partial idx=%zu "
                            "(basisVector=%zu block=%zu) value=(%g,%g) "
                            "magnitude=%g\n",
                            idx,
                            idx / partialBlockCount,
                            idx % partialBlockCount,
                            value.cREAL, value.cIMAG, magnitude);
                    std::fflush(stderr);
                }
            }

            _partialDots.reduceStacked(
                    result,
                    vectorCount,
                    partialBlockCount,
                    false);
#else
            static_assert(
                    !onDevice,
                    "Device Lanczos basis requires GPU compilation");
#endif
        } else {
            const Vect3arrayAcc<floatT> basis =
                    _vectors->getAccessor();
            const Vect3arrayAcc<floatT> input = vector.getAccessor();
            for (size_t j = 0; j < vectorCount; ++j) {
                COMPLEX(double) sum(0.0, 0.0);
                for (size_t siteIndex = 0;
                     siteIndex < _bulkVolume;
                     ++siteIndex) {
                    const gSite site =
                            GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getSite(siteIndex);
                    const gSite storedSite =
                            basisSite<
                                    floatT,
                                    LatticeLayout,
                                    HaloDepthSpin>(
                                            siteIndex,
                                            j,
                                            _fullVolume);
                    const COMPLEX(floatT) product =
                            basis.getElement(storedSite)
                            * input.getElement(site);
                    sum += COMPLEX(double)(
                            static_cast<double>(product.cREAL),
                            static_cast<double>(product.cIMAG));
                }
                result[j] = sum;
            }
            _comm.reduce(result.data(), static_cast<int>(result.size()));
        }
        ++_globalReductions;
        return result;
    }

    void subtractCombination(
            Spinor &vector,
            const std::vector<COMPLEX(double)> &coefficients) {
        const size_t vectorCount = coefficients.size();
        checkCount(vectorCount);
        if (vectorCount == 0) {
            return;
        }
        requireUnrotatedMainStorage();

        // TEMPORARY DIAGNOSTIC (2026-09-18): on JLab 21g (AMD), the custom
        // subtractBasisCombinationKernel path below (disabled via #if 0)
        // produces a NaN residual at column 0, even though basis.load() and
        // basis.dot() -- which read the exact same stored data -- do not.
        // This loop bypasses that kernel entirely: it reuses the
        // already-validated Basis::load, plus a new TRLanSubtractSingleVector
        // functor run through the generic (already-validated) iterateOverFull
        // path instead of a hand-rolled kernel launch. If this stops the
        // crash, the custom kernel itself is the bug; if it doesn't, the
        // corruption is coming from somewhere else and this diagnostic
        // should be reverted.
        std::fprintf(stderr,
                "DIAG lanczosKernels.h: subtractCombination entered, "
                "vectorCount=%zu onDevice=%d bulkVolume=%zu\n",
                vectorCount, static_cast<int>(onDevice),
                static_cast<size_t>(_bulkVolume));
        std::fflush(stderr);

        // TEMPORARY DIAGNOSTIC (2026-09-19): scan a host-side field over the
        // *entire* full volume (bulk + halo) for the first non-finite site,
        // and report whether that site falls in the bulk range
        // ([0, _bulkVolume)) or the halo range ([_bulkVolume, _fullVolume)).
        // Motivation: basis.dot()'s isfinite check (the one guarding entry
        // into this function) only scans the bulk, while this function's
        // kernels -- both the original hand-rolled one and the
        // iterateOverFull replacement below -- iterate the full volume. If
        // the halo of `applied`/`_vectors` was never explicitly initialized
        // (no halo exchange after the exponential/Chebyshev filter), this
        // would be the first read of that garbage memory, and this scan
        // will show the offending index landing at/after _bulkVolume.
        auto scanFullVolumeForNonFinite = [this](
                const Spinorfield<
                        floatT, false, LatticeLayout, HaloDepthSpin, 1>
                        &hostField,
                const char *label) {
            const Vect3arrayAcc<floatT> acc = hostField.getAccessor();
            for (size_t fullSite = 0; fullSite < _fullVolume; ++fullSite) {
                gSite site;
                site.isiteFull = fullSite;
                const Vect3<floatT> element = acc.getElement(site);
                const bool finite =
                        std::isfinite(element.getElement0().cREAL)
                        && std::isfinite(element.getElement0().cIMAG)
                        && std::isfinite(element.getElement1().cREAL)
                        && std::isfinite(element.getElement1().cIMAG)
                        && std::isfinite(element.getElement2().cREAL)
                        && std::isfinite(element.getElement2().cIMAG);
                if (!finite) {
                    std::fprintf(stderr,
                            "DIAG TRLan scan: %s first non-finite at "
                            "fullSite=%zu region=%s bulkVolume=%zu "
                            "fullVolume=%zu\n",
                            label, fullSite,
                            (fullSite < _bulkVolume ? "BULK" : "HALO"),
                            static_cast<size_t>(_bulkVolume),
                            static_cast<size_t>(_fullVolume));
                    std::fflush(stderr);
                    return;
                }
            }
            std::fprintf(stderr,
                    "DIAG TRLan scan: %s all finite over full volume "
                    "(bulkVolume=%zu fullVolume=%zu)\n",
                    label, static_cast<size_t>(_bulkVolume),
                    static_cast<size_t>(_fullVolume));
            std::fflush(stderr);
        };

        for (size_t j = 0; j < vectorCount; ++j) {
            Spinor storedVector(_comm);
            load(j, storedVector);
            const COMPLEX(floatT) coefficient(
                    static_cast<floatT>(coefficients[j].cREAL),
                    static_cast<floatT>(coefficients[j].cIMAG));
            std::fprintf(stderr,
                    "DIAG lanczosKernels.h: loop j=%zu, reaching dump block\n",
                    j);
            std::fflush(stderr);

            // TEMPORARY DIAGNOSTIC (2026-09-18): dump raw element values
            // (not just an aggregate isfinite check) at a handful of sites,
            // for both operands going into this subtraction and for the
            // result coming out, to see directly whether the corruption is
            // already present in storedVector's second read of basis vector
            // j (the same index the top-of-loop basis.load() just validated
            // clean), or only appears in `vector` after the subtraction
            // despite clean inputs.
            if constexpr (onDevice) {
                Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1>
                        hostStored(_comm, "TRLanDiag_hostStored");
                hostStored = storedVector;
                Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1>
                        hostBefore(_comm, "TRLanDiag_hostBefore");
                hostBefore = vector;
                for (size_t siteIndex = 0;
                     siteIndex < 4 && siteIndex < _bulkVolume;
                     ++siteIndex) {
                    const gSite site =
                            GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getSite(siteIndex);
                    const Vect3<floatT> storedElement =
                            hostStored.getAccessor().getElement(site);
                    const Vect3<floatT> beforeElement =
                            hostBefore.getAccessor().getElement(site);
                    std::fprintf(stderr,
                            "DIAG TRLan raw dump (pre-subtract): j=%zu "
                            "site=%zu coeff=(%g,%g) storedVector[0]=(%g,%g) "
                            "vectorBefore[0]=(%g,%g)\n",
                            j, siteIndex,
                            static_cast<double>(coefficient.cREAL),
                            static_cast<double>(coefficient.cIMAG),
                            static_cast<double>(
                                    storedElement.getElement0().cREAL),
                            static_cast<double>(
                                    storedElement.getElement0().cIMAG),
                            static_cast<double>(
                                    beforeElement.getElement0().cREAL),
                            static_cast<double>(
                                    beforeElement.getElement0().cIMAG));
                    std::fflush(stderr);
                }
                scanFullVolumeForNonFinite(
                        hostStored, "storedVector (pre-subtract)");
                scanFullVolumeForNonFinite(
                        hostBefore, "vector (pre-subtract)");
            }

            vector.iterateOverFull(
                    TRLanSubtractSingleVector<floatT>(
                            vector.getAccessor(),
                            storedVector.getAccessor(),
                            coefficient));

            if constexpr (onDevice) {
                Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, 1>
                        hostAfter(_comm, "TRLanDiag_hostAfter");
                hostAfter = vector;
                for (size_t siteIndex = 0;
                     siteIndex < 4 && siteIndex < _bulkVolume;
                     ++siteIndex) {
                    const gSite site =
                            GIndexer<LatticeLayout, HaloDepthSpin>
                                    ::getSite(siteIndex);
                    const Vect3<floatT> afterElement =
                            hostAfter.getAccessor().getElement(site);
                    std::fprintf(stderr,
                            "DIAG TRLan raw dump (post-subtract): j=%zu "
                            "site=%zu vectorAfter[0]=(%g,%g)\n",
                            j, siteIndex,
                            static_cast<double>(
                                    afterElement.getElement0().cREAL),
                            static_cast<double>(
                                    afterElement.getElement0().cIMAG));
                    std::fflush(stderr);
                }
                scanFullVolumeForNonFinite(
                        hostAfter, "vector (post-subtract)");
            }
        }
        return;
    }

#if 0
        upload(coefficients);

        if constexpr (onDevice) {
#ifdef __GPUCC__
            const dim3 block(128);
            const dim3 grid(
                    static_cast<unsigned int>(
                            (_fullVolume + block.x - 1) / block.x));
#ifdef USE_CUDA
            subtractBasisCombinationKernel<
                    floatT, LatticeLayout, HaloDepthSpin>
                    <<<grid, block>>>(
                            vector.getAccessor(),
                            _vectors->getAccessor(),
                            _coefficients
                                    ->template getPointer<COMPLEX(double)>(),
                            vectorCount,
                            _fullVolume);
#elif defined USE_HIP
            hipLaunchKernelGGL(
                    (subtractBasisCombinationKernel<
                            floatT, LatticeLayout, HaloDepthSpin>),
                    grid,
                    block,
                    0,
                    0,
                    vector.getAccessor(),
                    _vectors->getAccessor(),
                    _coefficients
                            ->template getPointer<COMPLEX(double)>(),
                    vectorCount,
                    _fullVolume);
#endif
            checkLastKernel("TRLan basis subtraction");
#else
            static_assert(
                    !onDevice,
                    "Device Lanczos basis requires GPU compilation");
#endif
        } else {
            Vect3arrayAcc<floatT> output = vector.getAccessor();
            const Vect3arrayAcc<floatT> basis =
                    _vectors->getAccessor();
            for (size_t fullSite = 0;
                 fullSite < _fullVolume;
                 ++fullSite) {
                gSite site;
                site.isiteFull = fullSite;
                Vect3<floatT> value = output.getElement(site);
                for (size_t j = 0; j < vectorCount; ++j) {
                    const gSite storedSite =
                            basisFullSite<
                                    floatT,
                                    LatticeLayout,
                                    HaloDepthSpin>(
                                            fullSite,
                                            j,
                                            _fullVolume);
                    const COMPLEX(floatT) coefficient(
                            static_cast<floatT>(
                                    coefficients[j].cREAL),
                            static_cast<floatT>(
                                    coefficients[j].cIMAG));
                    value -= coefficient
                            * basis.getElement(storedSite);
                }
                output.setElement(site, value);
            }
        }
    }
#endif

    double orthogonalize(
            Spinor &vector,
            const size_t vectorCount,
            const int passes) {
        checkCount(vectorCount);
        if (passes < 0) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan reorthogonalization passes cannot be negative"));
        }

        double maximumProjection = 0.0;
        for (int pass = 0; pass < passes; ++pass) {
            std::vector<COMPLEX(double)> coefficients =
                    dot(vector, vectorCount);
            for (const COMPLEX(double) coefficient : coefficients) {
                maximumProjection = std::max(
                        maximumProjection,
                        std::hypot(
                                coefficient.cREAL,
                                coefficient.cIMAG));
            }
            subtractCombination(vector, coefficients);
        }
        return maximumProjection;
    }

    void rotate(
            const std::vector<std::vector<double>> &eigenvectors,
            const std::vector<int> &columns,
            const size_t sourceCount) {
        checkCount(sourceCount);
        const size_t destinationCount = columns.size();
        checkCount(destinationCount);
        if (destinationCount > _rotationCapacity) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan Ritz rotation exceeds scratch capacity"));
        }
        if (destinationCount == 0) {
            return;
        }
        requireUnrotatedMainStorage();
        if (eigenvectors.size() < sourceCount) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan Ritz coefficient matrix has too few rows"));
        }

        std::vector<double> packed(
                sourceCount * destinationCount, 0.0);
        for (size_t source = 0; source < sourceCount; ++source) {
            for (size_t destination = 0;
                 destination < destinationCount;
                 ++destination) {
                const int column = columns[destination];
                if (column < 0
                    || static_cast<size_t>(column)
                            >= eigenvectors[source].size()) {
                    throw std::runtime_error(stdLogger.fatal(
                            "TRLan Ritz coefficient column is out of range"));
                }
                packed[source * destinationCount + destination] =
                        eigenvectors[source][column];
            }
        }
        upload(packed);

        if constexpr (onDevice) {
#ifdef __GPUCC__
            const dim3 block(rotateSiteTile, rotateVectorTile);
            const dim3 grid(
                    static_cast<unsigned int>(
                            (_fullVolume + rotateSiteTile - 1)
                            / rotateSiteTile),
                    static_cast<unsigned int>(
                            (destinationCount + rotateVectorTile - 1)
                            / rotateVectorTile));
#ifdef USE_CUDA
            rotateBasisKernel<
                    floatT, LatticeLayout, HaloDepthSpin>
                    <<<grid, block>>>(
                            _vectors->getAccessor(),
                            _rotationScratch.getAccessor(),
                            _coefficients->template getPointer<double>(),
                            sourceCount,
                            destinationCount,
                            _fullVolume);
#elif defined USE_HIP
            hipLaunchKernelGGL(
                    (rotateBasisKernel<
                            floatT, LatticeLayout, HaloDepthSpin>),
                    grid,
                    block,
                    0,
                    0,
                    _vectors->getAccessor(),
                    _rotationScratch.getAccessor(),
                    _coefficients->template getPointer<double>(),
                    sourceCount,
                    destinationCount,
                    _fullVolume);
#endif
            checkLastKernel("TRLan basis rotation");
#else
            static_assert(
                    !onDevice,
                    "Device Lanczos basis requires GPU compilation");
#endif
        } else {
            const Vect3arrayAcc<floatT> source =
                    _vectors->getAccessor();
            Vect3arrayAcc<floatT> destination =
                    _rotationScratch.getAccessor();
            for (size_t output = 0;
                 output < destinationCount;
                 ++output) {
                for (size_t fullSite = 0;
                     fullSite < _fullVolume;
                     ++fullSite) {
                    Vect3<floatT> sum(
                            static_cast<floatT>(0.0));
                    for (size_t input = 0;
                         input < sourceCount;
                         ++input) {
                        const gSite inputSite =
                                basisFullSite<
                                        floatT,
                                        LatticeLayout,
                                        HaloDepthSpin>(
                                                fullSite,
                                                input,
                                                _fullVolume);
                        sum += static_cast<floatT>(
                                packed[
                                        input * destinationCount
                                        + output])
                                * source.getElement(inputSite);
                    }
                    const gSite outputSite =
                            basisFullSite<
                                    floatT,
                                    LatticeLayout,
                                    HaloDepthSpin>(
                                            fullSite,
                                            output,
                                            _fullVolume);
                    destination.setElement(outputSite, sum);
                }
            }
        }
        _rotationIsActive = true;
        _rotatedVectorCount = destinationCount;
        ++_rotations;
    }

    void commitRotation() {
        if (!_rotationIsActive) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan has no Ritz rotation to commit"));
        }
        requireMainStorage();
        _vectors->copyFromPartial(
                _rotationScratch,
                _rotatedVectorCount * _fullVolume,
                0,
                0);
        _rotationIsActive = false;
        _rotatedVectorCount = 0;
    }

    void releaseMainStorage() {
        if (!_rotationIsActive) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan cannot release its basis before a final Ritz rotation"));
        }
        _vectors.reset();
    }

private:
    CommunicationBase &_comm;
    size_t _capacity;
    size_t _rotationCapacity;
    size_t _bulkVolume;
    size_t _fullVolume;
    std::unique_ptr<Vect3array<floatT, onDevice>> _vectors;
    Vect3array<floatT, onDevice> _rotationScratch;
    LatticeContainer<onDevice, COMPLEX(double)> _partialDots;
    gMemoryPtr<onDevice> _coefficients;
    gMemoryPtr<false> _hostCoefficients;
    uint64_t _globalReductions;
    uint64_t _rotations;
    bool _rotationIsActive;
    size_t _rotatedVectorCount;

    static size_t checkedElementCount(
            const size_t vectors,
            const size_t volume) {
        if (vectors == 0
            || volume == 0
            || vectors
                    > std::numeric_limits<size_t>::max()
                            / volume) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis allocation size overflows size_t"));
        }
        const size_t elements = vectors * volume;
        const size_t bytesPerElement =
                3 * sizeof(COMPLEX(floatT));
        if (elements
                > std::numeric_limits<size_t>::max()
                        / bytesPerElement) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis allocation byte count overflows size_t"));
        }
        return elements;
    }

    static size_t checkedSum(
            const size_t left,
            const size_t right) {
        if (left
                > std::numeric_limits<size_t>::max()
                        - right) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan combined basis allocation overflows size_t"));
        }
        const size_t elements = left + right;
        const size_t bytesPerElement =
                3 * sizeof(COMPLEX(floatT));
        if (elements
                > std::numeric_limits<size_t>::max()
                        / bytesPerElement) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan combined basis byte count overflows size_t"));
        }
        return elements;
    }

    void checkIndex(const size_t index) const {
        if (index >= _capacity) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis index exceeds allocated capacity"));
        }
    }

    void checkCount(const size_t count) const {
        if (count > _capacity) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan basis vector count exceeds allocated capacity"));
        }
    }

    void requireMainStorage() const {
        if (!_vectors) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan main basis storage has been released"));
        }
    }

    void requireUnrotatedMainStorage() const {
        requireMainStorage();
        if (_rotationIsActive) {
            throw std::runtime_error(stdLogger.fatal(
                    "TRLan operation requires a committed basis rotation"));
        }
    }

    template<class valueT>
    void upload(const std::vector<valueT> &values) {
        const size_t bytes = values.size() * sizeof(valueT);
        _coefficients->template adjustSize<valueT>(values.size());
        if constexpr (onDevice) {
            _hostCoefficients
                    ->template adjustSize<valueT>(values.size());
            std::copy(
                    values.begin(),
                    values.end(),
                    _hostCoefficients->template getPointer<valueT>());
            _coefficients->copyFrom(_hostCoefficients, bytes);
        } else {
            std::copy(
                    values.begin(),
                    values.end(),
                    _coefficients->template getPointer<valueT>());
        }
    }

    static void checkLastKernel(const char *operation) {
        if constexpr (onDevice) {
            const gpuError_t error = gpuGetLastError();
            if (error != gpuSuccess) {
                GpuError(operation, error);
            }
        }
    }
};

} // namespace trlan_detail
