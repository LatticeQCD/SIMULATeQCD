#pragma once

#include "../../base/latticeContainer.h"
#include "../../base/math/vect3array.h"
#include "../../base/wrapper/gpu_wrapper.h"
#include "../../spinor/spinorfield.h"

#include <algorithm>
#include <cmath>
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
