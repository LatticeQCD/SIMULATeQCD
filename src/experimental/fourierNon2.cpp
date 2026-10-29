#include "fourierNon2.h"
#include "../base/stopWatch.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif


template<typename floatT>
void FourierClass<floatT>::moveMatrix4x4SymComponentToContainer(
    LatticeContainer<true, Matrix4x4SymComplex<floatT>>& emt,
    LatticeContainer<true, COMPLEX(floatT)>& redBase,
    int emtComponent
) {

    gpuError_t gpuErr;
    // copy information from spinor over to redbase 
    size_t elems = lx*ly*lz*lt;
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

    #ifdef USE_CUDA
        copyMatrix4x4SymComponentToContainerLocal<floatT><<<gridDim, blockDim>>>(redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lx, (int) ly, (int) lz, (int) lt);
    #endif

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<typename floatT>
void FourierClass<floatT>::moveTensor4x4Symx4x4SymComplexComponentToContainer(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensor,
    LatticeContainer<true, COMPLEX(floatT)>& redBase,
    int firstIndexPair, int secondIndexPair
) {

    gpuError_t gpuErr;
    // copy information from spinor over to redbase 
    size_t elems = lx*ly*lz*lt;
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

    #ifdef USE_CUDA
        copyTensor4x4Symx4x4SymComplexComponentToContainerLocal<floatT><<<gridDim, blockDim>>>(redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lz, (int) lt);
    #endif

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<class floatT>
struct CopyMatrix4x4SymToContainer {

    LatticeContainerAccessor matrixAccessor;
    int index;
    typedef GIndexer<All> GInd;

    CopyMatrix4x4SymToContainer(LatticeContainerAccessor _matrixAccessor, int _index) : matrixAccessor(_matrixAccessor) {
        index = _index;
    }

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        Matrix4x4SymComplex<floatT> matrix(matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        return matrix(index);
    }

};

template<class floatT>
struct CopyTensor4x4Symx4x4SymHalfMatrixToContainer {

    LatticeContainerAccessor tensorAccessor;
    int firstIndexPair;
    typedef GIndexer<All> GInd;

    CopyTensor4x4Symx4x4SymHalfMatrixToContainer(LatticeContainerAccessor _tensorAccessor, int _firstIndexPair) : tensorAccessor(_tensorAccessor) {
        firstIndexPair = _firstIndexPair;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT> operator()(gSite site) {

        Tensor4x4Symx4x4SymComplex<floatT> tensor(tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site));

        return tensor.getSecondMatrix4x4SymComplex(firstIndexPair);
    }

};

template<class floatT>
struct CopyContainerToMatrix4x4Sym {

    LatticeContainerAccessor complexAccessor;
    LatticeContainerAccessor matrixAccessor;
    int index;
    typedef GIndexer<All> GInd;

    CopyContainerToMatrix4x4Sym(LatticeContainerAccessor _complexAccessor, LatticeContainerAccessor _matrixAccessor, int _index) :
        complexAccessor(_complexAccessor), matrixAccessor(_matrixAccessor) {
        index = _index;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT> operator()(gSite site) {

        COMPLEX(floatT) complexValue(complexAccessor.getElement<COMPLEX(floatT)>(site));

        Matrix4x4SymComplex<floatT> matrix(matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        matrix(index, complexValue);

        return matrix;
    }

};

template<class floatT>
struct CopyContainerToTensor4x4Symx4x4SymHalfMatrix {

    LatticeContainerAccessor matrixAccessor;
    LatticeContainerAccessor tensorAccessor;
    int firstIndexPair;
    typedef GIndexer<All> GInd;

    CopyContainerToTensor4x4Symx4x4SymHalfMatrix(LatticeContainerAccessor _matrixAccessor, LatticeContainerAccessor _tensorAccessor, int _firstIndexPair) :
        matrixAccessor(_matrixAccessor), tensorAccessor(_tensorAccessor) {
        firstIndexPair = _firstIndexPair;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> matrix(matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> tensor(tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site));

        tensor.setSecondMatrix4x4SymComplex(firstIndexPair, matrix);

        return tensor;
    }

};


template<typename floatT>
template<int dir>
void FourierClass<floatT>::moveContainerToMatrix4x4SymDirection(
    LatticeContainer<true, COMPLEX(floatT)>& redBase,
    LatticeContainer<true, Matrix4x4SymComplex<floatT>>& matrix,
    int component
) {

    gpuError_t gpuErr;
    // copy information from spinor over to redbase 
    size_t elems = lx*ly*lz*lt;
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

    if (dir == 0) {
        #ifdef USE_CUDA
            copyContainerToMatrix4x4Local<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), (int) component, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToMatrix4x4Local<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), component, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #endif
    }

    if (dir == 1) {
        #ifdef USE_CUDA
            copyContainerToMatrix4x4Local<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), (int) component, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToMatrix4x4Local<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), component, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #endif
    }

    if (dir == 2) {
        #ifdef USE_CUDA
            copyContainerToMatrix4x4Local<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), (int) component, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToMatrix4x4Local<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), component, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #endif
    }

    if (dir == 3) {
        #ifdef USE_CUDA
            copyContainerToMatrix4x4Local<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), (int) component, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToMatrix4x4Local<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), matrix.getAccessor(), (size_t) (lx*ly*lz*lt), component, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #endif
    }

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<typename floatT>
template<int dir>
void FourierClass<floatT>::moveContainerToTensor4x4Symx4x4SymComplexDirection(
    LatticeContainer<true, COMPLEX(floatT)>& redBase,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensor,
    int firstIndexPair, int secondIndexPair
) {

    gpuError_t gpuErr;
    // copy information from spinor over to redbase 
    size_t elems = lx*ly*lz*lt;
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

    if (dir == 0) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #endif
    }

    if (dir == 1) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #endif
    }

    if (dir == 2) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #endif
    }

    if (dir == 3) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplexLocal<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), tensor.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #endif
    }

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<typename floatT>
template<int dir>
void FourierClass<floatT>::performFourierTransformDirection(
    LatticeContainer<true, COMPLEX(floatT)> &redBaseDevice,
    LatticeContainer<false, COMPLEX(floatT)> &redBaseHost,
    int sign
) {

    gpuError_t gpuErr;

    size_t elems;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    dim3 gridDim;

    if (dir == 0) {

        elems = lx*ly*lz*lt; // TODO: extractable

        if (nodes[0] > 1) {

            gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            gatherHostXYZ<floatT, 0>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commX, lxL, ly, lz);

            gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[0]*elems, gpuMemcpyHostToDevice);

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in x direction
        elems = ly*lz*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));
        
        #ifdef USE_CUDA
            fourier<floatT, 0><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourier<floatT, 0>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
        #endif

        gpuErr = gpuGetLastError(); // TODO: extractable
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr); // TODO: extractable

    }

    if (dir == 1) {

        elems = lx*ly*lz*lt;

        if (nodes[1] > 1) {

            gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            gatherHostXYZ<floatT, 1>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commY, lx, lyL, lz);

            gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[1]*elems, gpuMemcpyHostToDevice);

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in y direction
        elems = lx*lz*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x))); // TODO: why blockDim.x, not .y?
        
        #ifdef USE_CUDA
            fourier<floatT, 1><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourier<floatT, 1>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    if (dir == 2) {

        elems = lx*ly*lz*lt;
        
        if (nodes[2] > 1) {

            gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            gatherHostXYZ<floatT, 2>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commZ, lx, ly, lzL);
            // gatherAllHost((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr() ->getPointer(), commBase); // Why is this here?

            gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[2]*elems, gpuMemcpyHostToDevice);

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in z direction
        elems = lx*ly*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));
        
        #ifdef USE_CUDA
            fourier<floatT, 2><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourier<floatT, 2>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    if (dir == 3) {

        elems = lx*ly*lz*lt;
        
        if (nodes[3] > 1) { // TODO: Not nodes nodes[3]? Done

            gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            gatherHostXYZT<floatT, 3>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commT, lx, ly, lz, ltL);

            gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[3]*elems, gpuMemcpyHostToDevice);

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in t direction
        elems = lx*ly*lz;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

        #ifdef USE_CUDA
            fourier<floatT, 3><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, ltL, lz, lsT, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourier<floatT, 3>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, ltL, lz, lsT, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

   }

}

template<typename floatT>
template<int dir, typename elemType>
void FourierClass<floatT>::performFourierTransformDirectionPolymorph(
    LatticeContainer<true, elemType>& redBaseDevice,
    int sign
) {

    gpuError_t gpuErr;

    size_t elems;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    dim3 gridDim;

    if (dir == 0) {

        elems = lx*ly*lz*lt;

        if (nodes[0] > 1) {

            throw std::runtime_error(stdLogger.fatal("Function performFourierTransformDirectionPolymorph does only support gpu topology 1x1x1x1, not ", nodes[0], "x", nodes[1], "x", nodes[2], "x", nodes[3]));

            // TODO: extend this function to work with multiple gpus

            // gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(elemType)*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            // gatherHostXYZ<floatT, 0>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commX, lxL, ly, lz);

            // gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(elemType)*nodes[0]*elems, gpuMemcpyHostToDevice);

            // gpuErr = gpuGetLastError();
            // if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in x direction
        elems = ly*lz*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

        #ifdef USE_CUDA
        fourierPolymorph<floatT, elemType, 0><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
        #elif defined USE_HIP
        hipLaunchKernelGGL((fourierPolymorph<floatT, elemType, 0>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
        #endif

        gpuErr = gpuGetLastError(); // TODO: extractable
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr); // TODO: extractable

    }

    if (dir == 1) {

        elems = lx*ly*lz*lt;

        if (nodes[1] > 1) {

            throw std::runtime_error(stdLogger.fatal("Function performFourierTransformDirectionPolymorph does only support gpu topology 1x1x1x1, not ", nodes[0], "x", nodes[1], "x", nodes[2], "x", nodes[3]));

            // gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            // gatherHostXYZ<floatT, 1>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commY, lx, lyL, lz);

            // gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[1]*elems, gpuMemcpyHostToDevice);

            // gpuErr = gpuGetLastError();
            // if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in y direction
        elems = lx*lz*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x))); // TODO: why blockDim.x, not .y?
        
        #ifdef USE_CUDA
            fourierPolymorph<floatT, elemType, 1><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourierPolymorph<floatT, elemType, 1>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    if (dir == 2) {

        elems = lx*ly*lz*lt;
        
        if (nodes[2] > 1) {

            throw std::runtime_error(stdLogger.fatal("Function performFourierTransformDirectionPolymorph does only support gpu topology 1x1x1x1, not ", nodes[0], "x", nodes[1], "x", nodes[2], "x", nodes[3]));

            // gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            // gatherHostXYZ<floatT, 2>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commZ, lx, ly, lzL);
            // // gatherAllHost((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr() ->getPointer(), commBase); // Why is this here?

            // gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[2]*elems, gpuMemcpyHostToDevice);

            // gpuErr = gpuGetLastError();
            // if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in z direction
        elems = lx*ly*lt;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));
        
        #ifdef USE_CUDA
            fourierPolymorph<floatT, elemType, 2><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourierPolymorph<floatT, elemType, 2>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    if (dir == 3) {

        elems = lx*ly*lz*lt;
        
        if (nodes[3] > 1) { // TODO: Not nodes nodes[3]? Done

            throw std::runtime_error(stdLogger.fatal("Function performFourierTransformDirectionPolymorph does only support gpu topology 1x1x1x1, not ", nodes[0], "x", nodes[1], "x", nodes[2], "x", nodes[3]));

            // gpuMemcpy(redBaseHost.get_ContainerArrayPtr()->getPointer(), redBaseDevice.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

            // gatherHostXYZT<floatT, 3>((std::complex<floatT> *) redBaseHost.get_ContainerArrayPtr()->getPointer(), commT, lx, ly, lz, ltL);

            // gpuMemcpy(redBaseDevice.get_ContainerArrayPtr()->getPointer(), redBaseHost.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*nodes[3]*elems, gpuMemcpyHostToDevice);

            // gpuErr = gpuGetLastError();
            // if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

        }

        // perform the fourier transformation in t direction
        elems = lx*ly*lz;
        gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

        #ifdef USE_CUDA
            fourierPolymorph<floatT, elemType, 3><<<gridDim, blockDim>>>(redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, ltL, lz, lsT, sign);
        #elif defined USE_HIP
            hipLaunchKernelGGL((fourierPolymorph<floatT, elemType, 3>), dim3(gridDim), dim3(blockDim), 0, 0, redBaseDevice.getAccessor(), redBaseDevice.getAccessor(), elems, lx, ly, ltL, lz, lsT, sign);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

   }

}


template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourierTransformMatrix4x4SymComponentwise(
    LatticeContainer<true, Matrix4x4SymComplex<floatT>>& matrixIn,
    LatticeContainer<true, Matrix4x4SymComplex<floatT>>& matrixOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    // create helper lattice containers
    LatticeContainer<true, COMPLEX(floatT)> redBaseDevice(matrixIn.get_CommBase(), "RedBaseDevice", "RedBaseDevice", "RedBaseDevice", "RedBaseDevice");
    LatticeContainer<false, COMPLEX(floatT)> redBaseHost(matrixIn.get_CommBase(), "RedBaseHost", "RedBaseHost", "RedBaseHost", "RedBaseHost");
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> matrixIntermediate(matrixIn.get_CommBase(), "matrixIntermediate", "matrixIntermediate", "matrixIntermediate", "matrixIntermediate");
    redBaseDevice.adjustSize(lxL*lyL*lzL*ltL);
    redBaseHost.adjustSize(lxL*lyL*lzL*ltL);
    matrixIntermediate.adjustSize(GInd::getLatData().vol4);

    // copy initial matrix into working-base lattice container
    matrixIntermediate.copyFromLatticeContainer(matrixIn);

    for (int index = 0; index < 10; index++) {

        // TODO: Doesn't work with this yet, but performance increase not needed here
        // timerMoveIn.start();
        // redBaseDevice.template iterateOverBulk<All, 0>(CopyMatrix4x4SymToContainer<floatT>(matrixIntermediate.getAccessor(), index));
        // timerMoveIn.stop();

        if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
            timerMoveIn.start();
            moveMatrix4x4SymComponentToContainer(matrixIntermediate, redBaseDevice, index);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<0>(redBaseDevice, redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToMatrix4x4SymDirection<0>(redBaseDevice, matrixIntermediate, index);
            timerMoveOut.stop();
            
            timerMoveIn.start();
            moveMatrix4x4SymComponentToContainer(matrixIntermediate, redBaseDevice, index);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<1>(redBaseDevice, redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToMatrix4x4SymDirection<1>(redBaseDevice, matrixIntermediate, index);
            timerMoveOut.stop();
            
            timerMoveIn.start();
            moveMatrix4x4SymComponentToContainer(matrixIntermediate, redBaseDevice, index);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<2>(redBaseDevice, redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToMatrix4x4SymDirection<2>(redBaseDevice, matrixIntermediate, index);
            timerMoveOut.stop();
        }
        
        if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
            timerMoveIn.start();
            moveMatrix4x4SymComponentToContainer(matrixIntermediate, redBaseDevice, index);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<3>(redBaseDevice, redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToMatrix4x4SymDirection<3>(redBaseDevice, matrixIntermediate, index);
            timerMoveOut.stop();
        }
        
        // TODO: Doesn't work with this yet, but performance increase not needed here
        // timerMoveOut.start();
        // matrixIntermediate.template iterateOverBulk<All, 0>(CopyContainerToMatrix4x4Sym<floatT>(redBaseDevice.getAccessor(), matrixIntermediate.getAccessor(), index));
        // timerMoveOut.stop();

        // handle timers
        timePerComponentMoveIn.push_back(timerMoveIn.seconds());
        timePerComponentFT.push_back(timerFT.seconds());
        timePerComponentMoveOut.push_back(timerMoveOut.seconds());

        timerMoveIn.reset();
        timerFT.reset();
        timerMoveOut.reset();

    }

    // summarize timers
    auto const countMoveIn = static_cast<double>(timePerComponentMoveIn.size());
    double timePerComponentAverageMoveIn = std::reduce(timePerComponentMoveIn.begin(), timePerComponentMoveIn.end()) / countMoveIn;
    auto const countFT = static_cast<double>(timePerComponentFT.size());
    double timePerComponentAverageFT = std::reduce(timePerComponentFT.begin(), timePerComponentFT.end()) / countFT;
    auto const countMoveOut = static_cast<double>(timePerComponentMoveOut.size());
    double timePerComponentAverageMoveOut = std::reduce(timePerComponentMoveOut.begin(), timePerComponentMoveOut.end()) / countMoveOut;

    rootLogger.debug("       Fourier EMT move-in   took           ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.debug("       Fourier EMT scalar FT took           ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.debug("       Fourier EMT move-out  took           ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    // copy working-base lattice container into output matrix
    matrixOut.copyFromLatticeContainer(matrixIntermediate);

}

template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    // create helper lattice containers
    LatticeContainer<true, COMPLEX(floatT)> _redBaseDevice(tensorIn.get_CommBase(), "RedBaseDevice", "RedBaseDevice", "RedBaseDevice", "RedBaseDevice");
    LatticeContainer<false, COMPLEX(floatT)> _redBaseHost(tensorIn.get_CommBase(), "RedBaseHost", "RedBaseHost", "RedBaseHost", "RedBaseHost");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> tensorIntermediate(tensorIn.get_CommBase(), "tensorIntermediate", "tensorIntermediate", "tensorIntermediate", "tensorIntermediate");
    _redBaseDevice.adjustSize(lxL*lyL*lzL*ltL);
    _redBaseHost.adjustSize(lxL*lyL*lzL*ltL);
    tensorIntermediate.adjustSize(GInd::getLatData().vol4);
    
    // copy initial tensor into working-base lattice container
    tensorIntermediate.copyFromLatticeContainer(tensorIn);

    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {

            if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
                timerMoveIn.start();
                moveTensor4x4Symx4x4SymComplexComponentToContainer(tensorIntermediate, _redBaseDevice, firstIndexPair, secondIndexPair);
                timerMoveIn.stop();
                timerFT.start();
                performFourierTransformDirection<0>(_redBaseDevice, _redBaseHost, sign);
                timerFT.stop();
                timerMoveOut.start();
                moveContainerToTensor4x4Symx4x4SymComplexDirection<0>(_redBaseDevice, tensorIntermediate, firstIndexPair, secondIndexPair);
                timerMoveOut.stop();
                
                timerMoveIn.start();
                moveTensor4x4Symx4x4SymComplexComponentToContainer(tensorIntermediate, _redBaseDevice, firstIndexPair, secondIndexPair);
                timerMoveIn.stop();
                timerFT.start();
                performFourierTransformDirection<1>(_redBaseDevice, _redBaseHost, sign);
                timerFT.stop();
                timerMoveOut.start();
                moveContainerToTensor4x4Symx4x4SymComplexDirection<1>(_redBaseDevice, tensorIntermediate, firstIndexPair, secondIndexPair);
                timerMoveOut.stop();
                
                timerMoveIn.start();
                moveTensor4x4Symx4x4SymComplexComponentToContainer(tensorIntermediate, _redBaseDevice, firstIndexPair, secondIndexPair);
                timerMoveIn.stop();
                timerFT.start();
                performFourierTransformDirection<2>(_redBaseDevice, _redBaseHost, sign);
                timerFT.stop();
                timerMoveOut.start();
                moveContainerToTensor4x4Symx4x4SymComplexDirection<2>(_redBaseDevice, tensorIntermediate, firstIndexPair, secondIndexPair);
                timerMoveOut.stop();
            }
            
            if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
                timerMoveIn.start();
                moveTensor4x4Symx4x4SymComplexComponentToContainer(tensorIntermediate, _redBaseDevice, firstIndexPair, secondIndexPair);
                timerMoveIn.stop();
                timerFT.start();
                performFourierTransformDirection<3>(_redBaseDevice, _redBaseHost, sign);
                timerFT.stop();
                timerMoveOut.start();
                moveContainerToTensor4x4Symx4x4SymComplexDirection<3>(_redBaseDevice, tensorIntermediate, firstIndexPair, secondIndexPair);
                timerMoveOut.stop();
            }
        
            timePerComponentMoveIn.push_back(timerMoveIn.seconds());
            timePerComponentFT.push_back(timerFT.seconds());
            timePerComponentMoveOut.push_back(timerMoveOut.seconds());

            timerMoveIn.reset();
            timerFT.reset();
            timerMoveOut.reset();

        }
    }

    auto const countMoveIn = static_cast<double>(timePerComponentMoveIn.size());
    double timePerComponentAverageMoveIn = std::reduce(timePerComponentMoveIn.begin(), timePerComponentMoveIn.end()) / countMoveIn;
    auto const countFT = static_cast<double>(timePerComponentFT.size());
    double timePerComponentAverageFT = std::reduce(timePerComponentFT.begin(), timePerComponentFT.end()) / countFT;
    auto const countMoveOut = static_cast<double>(timePerComponentMoveOut.size());
    double timePerComponentAverageMoveOut = std::reduce(timePerComponentMoveOut.begin(), timePerComponentMoveOut.end()) / countMoveOut;

    rootLogger.debug("       Fourier 4x4symx4x4sym move-in   took ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.debug("       Fourier 4x4symx4x4sym scalar FT took ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.debug("       Fourier 4x4symx4x4sym move-out  took ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    tensorOut.copyFromLatticeContainer(tensorIntermediate);

}

template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourierTransformTensor4x4Symx4x4SymHalfPolymorph(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    // create helper lattice containers
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> _redBaseDevice(tensorIn.get_CommBase(), "RedBaseDevice", "RedBaseDevice", "RedBaseDevice", "RedBaseDevice");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> tensorIntermediate(tensorIn.get_CommBase(), "tensorIntermediate", "tensorIntermediate", "tensorIntermediate", "tensorIntermediate");
    _redBaseDevice.adjustSize(GInd::getLatData().vol4);
    tensorIntermediate.adjustSize(GInd::getLatData().vol4);
    
    // copy initial tensor into working-base lattice container
    tensorIntermediate.copyFromLatticeContainer(tensorIn);

    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {

        // copy the firstIndexPair-submatrix of tensor to the reduction base
        timerMoveIn.start();
        _redBaseDevice.template iterateOverBulk<All, 0>(CopyTensor4x4Symx4x4SymHalfMatrixToContainer<floatT>(tensorIntermediate.getAccessor(), firstIndexPair));
        timerMoveIn.stop();
        
        // perform polymorph Fourier transformation of the Matrix4x4Sym reduction base
        timerFT.start();
        if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
            performFourierTransformDirectionPolymorph<0, Matrix4x4SymComplex<floatT>>(_redBaseDevice, sign);
            performFourierTransformDirectionPolymorph<1, Matrix4x4SymComplex<floatT>>(_redBaseDevice, sign);
            performFourierTransformDirectionPolymorph<2, Matrix4x4SymComplex<floatT>>(_redBaseDevice, sign);
        }
        if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
            performFourierTransformDirectionPolymorph<3, Matrix4x4SymComplex<floatT>>(_redBaseDevice, sign);
        }
        timerFT.stop();
        
        // copy the matrix of the reduction base to the firstIndexPair-submatrix of tensor
        timerMoveOut.start();
        tensorIntermediate.template iterateOverBulk<All, 0>(CopyContainerToTensor4x4Symx4x4SymHalfMatrix<floatT>(_redBaseDevice.getAccessor(), tensorIntermediate.getAccessor(), firstIndexPair));
        timerMoveOut.stop();
        
        // handle timers
        timePerComponentMoveIn.push_back(timerMoveIn.seconds());
        timePerComponentFT.push_back(timerFT.seconds());
        timePerComponentMoveOut.push_back(timerMoveOut.seconds());

        timerMoveIn.reset();
        timerFT.reset();
        timerMoveOut.reset();

    }

    // summarize timers
    auto const countMoveIn = static_cast<double>(timePerComponentMoveIn.size());
    double timePerComponentAverageMoveIn = std::reduce(timePerComponentMoveIn.begin(), timePerComponentMoveIn.end()) / countMoveIn;
    auto const countFT = static_cast<double>(timePerComponentFT.size());
    double timePerComponentAverageFT = std::reduce(timePerComponentFT.begin(), timePerComponentFT.end()) / countFT;
    auto const countMoveOut = static_cast<double>(timePerComponentMoveOut.size());
    double timePerComponentAverageMoveOut = std::reduce(timePerComponentMoveOut.begin(), timePerComponentMoveOut.end()) / countMoveOut;

    rootLogger.debug("       Fourier 4x4symx4x4sym move-in   took ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.debug("       Fourier 4x4symx4x4sym scalar FT took ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.debug("       Fourier 4x4symx4x4sym move-out  took ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    // copy working-base lattice container into output tensor
    tensorOut.copyFromLatticeContainer(tensorIntermediate);

}

template<typename floatT>
template<typename elemType, SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourierTransformPolymorph(
    LatticeContainer<true, elemType>& latticeIn,
    LatticeContainer<true, elemType>& latticeOut,
    int sign
) {

    typedef GIndexer<All> GInd;

    LatticeContainer<true, elemType> latticeIntermediate(latticeIn.get_CommBase(), "latticeIntermediate", "latticeIntermediate", "latticeIntermediate", "latticeIntermediate");
    latticeIntermediate.adjustSize(GInd::getLatData().vol4);

    // copy input lattice into working-base lattice container
    latticeIntermediate.copyFromLatticeContainer(latticeIn);

    // perform polymorph Fourier transform on elemType reduction base
    if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
        performFourierTransformDirectionPolymorph<0, elemType>(latticeIntermediate, sign);
        performFourierTransformDirectionPolymorph<1, elemType>(latticeIntermediate, sign);
        performFourierTransformDirectionPolymorph<2, elemType>(latticeIntermediate, sign);
    }
    if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
        performFourierTransformDirectionPolymorph<3, elemType>(latticeIntermediate, sign);
    }

    // copy working-base lattice container into output lattice
    latticeOut.copyFromLatticeContainer(latticeIntermediate);

}

template<typename floatT>
void gatherAllHost(std::complex<floatT> *in, CommunicationBase & commBase) {

    int coord[4];
    // all gather 4d
    typedef GIndexer<All, 0> GInd;
    int glx = GInd::getLatData().globLX;
    int lx  = GInd::getLatData().lx;
    int gly = GInd::getLatData().globLY;
    int ly  = GInd::getLatData().ly;
    int glz = GInd::getLatData().globLZ;
    int lz  = GInd::getLatData().lz;
    int glt = GInd::getLatData().globLT;
    int lt  = GInd::getLatData().lt;
    int myrank, rankSize;
    MPI_Comm_rank(commBase.getCart_comm(), &myrank);
    MPI_Comm_size(commBase.getCart_comm(), &rankSize);

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz*glt];

    if(std::is_same<floatT, double>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, commBase.getCart_comm() );
    }
    else if(std::is_same<floatT, float>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_COMPLEX, buf, lx*ly*lt*lz, MPI_COMPLEX, commBase.getCart_comm() );
    }

    for (int r=0; r<rankSize; r++) {
    MPI_Cart_coords(commBase.getCart_comm(), r, 4, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            in[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])+glz*((t+lt*coord[3]))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
        }
    }

    delete[] buf;

}


template<typename floatT, int direction>
void gatherHostXYZ(std::complex<floatT> *in, MPI_Comm &comm, int glx, int gly, int glz) {

    int coord[1];
    // gather 4d for extended directions
    typedef GIndexer<All, 0> GInd;
    int lx  = GInd::getLatData().lx;
    int ly  = GInd::getLatData().ly;
    int lz  = GInd::getLatData().lz;
    int glt = GInd::getLatData().globLT;
    int lt  = GInd::getLatData().lt;
    int myrank, rankSize;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &rankSize);

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz*glt];
    
    if(std::is_same<floatT, double>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, comm);
    }
    else if(std::is_same<floatT, float>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_COMPLEX, buf, lx*ly*lt*lz, MPI_COMPLEX, comm);
    }

    
    for (int r=0; r<rankSize; r++) {
        MPI_Cart_coords(comm, r, 1, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            if (direction == 0) {
                in[(x + lx*coord[0]) + glx*((y) + gly*((z) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            }
            else if (direction == 1) {
                in[(x) + glx*((y + ly*coord[0]) + gly*((z) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            }
            else if (direction == 2) {
                in[(x) + glx*((y) + gly*((z + lz*coord[0]) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            } 
        }
    }

    delete[] buf;

}

// TODO: Is there a difference between gatherHostXYZT(glt=globLT) and gatherHostXYZ?
template<typename floatT, int direction>
void gatherHostXYZT(std::complex<floatT> *in, MPI_Comm &comm, int glx, int gly, int glz, int glt) {

    int coord[1];
    // gather 4d for extended directions
    typedef GIndexer<All, 0> GInd;
    int lx  = GInd::getLatData().lx;
    int ly  = GInd::getLatData().ly;
    int lz  = GInd::getLatData().lz;
    int lt  = GInd::getLatData().lt;
    int myrank, rankSize;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &rankSize);

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz*glt];

    if(std::is_same<floatT, double>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, comm);
    }
    else if(std::is_same<floatT, float>::value) {
        MPI_Allgather(in, lx*ly*lt*lz, MPI_COMPLEX, buf, lx*ly*lt*lz, MPI_COMPLEX, comm);
    }


    for (int r=0; r<rankSize; r++) {
    MPI_Cart_coords(comm, r, 1, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            if (direction == 0) {
                in[(x + lx*coord[0]) + glx*((y) + gly*((z) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))]; // why always coord[0] not other coord components?
            }
            else if (direction == 1) {
                in[(x) + glx*((y + ly*coord[0]) + gly*((z) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            }
            else if (direction == 2) {
                in[(x) + glx*((y) + gly*((z + lz*coord[0]) + glz*((t))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            }
            else if (direction == 3) {
                in[(x) + glx*((y) + gly*((z) + glz*((t + lt*coord[0]))))] = buf[x + lx*(y + ly*(z + lz*(t + lt*r)))];
            }
        }
    }

    delete[] buf;

}


/// template declarations

template class FourierClass<double>;

template void FourierClass<double>::moveContainerToMatrix4x4SymDirection<0>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToMatrix4x4SymDirection<1>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToMatrix4x4SymDirection<2>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToMatrix4x4SymDirection<3>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);

template void FourierClass<double>::moveContainerToTensor4x4Symx4x4SymComplexDirection<0>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emt, int firstIndexPair, int secondIndexPair);
template void FourierClass<double>::moveContainerToTensor4x4Symx4x4SymComplexDirection<1>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emt, int firstIndexPair, int secondIndexPair);
template void FourierClass<double>::moveContainerToTensor4x4Symx4x4SymComplexDirection<2>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emt, int firstIndexPair, int secondIndexPair);
template void FourierClass<double>::moveContainerToTensor4x4Symx4x4SymComplexDirection<3>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emt, int firstIndexPair, int secondIndexPair);

template void FourierClass<double>::performFourierTransformDirection<0>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<false, COMPLEX(double)> & redBase2, int sign);
template void FourierClass<double>::performFourierTransformDirection<1>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<false, COMPLEX(double)> & redBase2, int sign);
template void FourierClass<double>::performFourierTransformDirection<2>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<false, COMPLEX(double)> & redBase2, int sign);
template void FourierClass<double>::performFourierTransformDirection<3>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<false, COMPLEX(double)> & redBase2, int sign);

template void FourierClass<double>::performFourierTransformDirectionPolymorph<0, Matrix4x4SymComplex<double>>(LatticeContainer<true, Matrix4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<1, Matrix4x4SymComplex<double>>(LatticeContainer<true, Matrix4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<2, Matrix4x4SymComplex<double>>(LatticeContainer<true, Matrix4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<3, Matrix4x4SymComplex<double>>(LatticeContainer<true, Matrix4x4SymComplex<double>> & redBase, int sign);

/* 
* The polymorphic Fourier instantiations are disabled for the Tensor4x4Symx4x4SymComplex<double>.
* The called polymorphic Fourier method fourierPolymorph in fourierNon2.h reserves 2 * LZ * size(elemType) of memory per GPU thread.
* For elemType = Tensor4x4Symx4x4SymComplex<double> and LZ = 64, this reaches 32kB per thread, which is too much for V100 (32GB) cards.
*/
// template void FourierClass<double>::performFourierTransformDirectionPolymorph<0, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
// template void FourierClass<double>::performFourierTransformDirectionPolymorph<1, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
// template void FourierClass<double>::performFourierTransformDirectionPolymorph<2, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
// template void FourierClass<double>::performFourierTransformDirectionPolymorph<3, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);


template void FourierClass<double>::performFourierTransformMatrix4x4SymComponentwise<SpatialTemporal::Both>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    int sign
);
template void FourierClass<double>::performFourierTransformMatrix4x4SymComponentwise<SpatialTemporal::Spatial>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    int sign
);
template void FourierClass<double>::performFourierTransformMatrix4x4SymComponentwise<SpatialTemporal::Temporal>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    int sign
);

template void FourierClass<double>::performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<SpatialTemporal::Both>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    int sign
);
template void FourierClass<double>::performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<SpatialTemporal::Spatial>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    int sign
);
template void FourierClass<double>::performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<SpatialTemporal::Temporal>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    int sign
);

template void FourierClass<double>::performFourierTransformPolymorph<COMPLEX(double), SpatialTemporal::Both>(
    LatticeContainer<true, COMPLEX(double)> &latticeIn,
    LatticeContainer<true, COMPLEX(double)> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformPolymorph<COMPLEX(double), SpatialTemporal::Spatial>(
    LatticeContainer<true, COMPLEX(double)> &latticeIn,
    LatticeContainer<true, COMPLEX(double)> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformPolymorph<COMPLEX(double), SpatialTemporal::Temporal>(
    LatticeContainer<true, COMPLEX(double)> &latticeIn,
    LatticeContainer<true, COMPLEX(double)> &latticeOut,
    int sign
);

template void FourierClass<double>::performFourierTransformPolymorph<Matrix4x4SymComplex<double>, SpatialTemporal::Both>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformPolymorph<Matrix4x4SymComplex<double>, SpatialTemporal::Spatial>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformPolymorph<Matrix4x4SymComplex<double>, SpatialTemporal::Temporal>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeOut,
    int sign
);

/* 
* The polymorphic Fourier instantiations are disabled for the Tensor4x4Symx4x4SymComplex<double>.
* The called polymorphic Fourier method fourierPolymorph in fourierNon2.h reserves 2 * LZ * size(elemType) of memory per GPU thread.
* For elemType = Tensor4x4Symx4x4SymComplex<double> and LZ = 64, this reaches 32kB per thread, which is too much for V100 (32GB) cards.
*/
// template void FourierClass<double>::performFourierTransformPolymorph<Tensor4x4Symx4x4SymComplex<double>, SpatialTemporal::Both>(
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
//     int sign
// );
// template void FourierClass<double>::performFourierTransformPolymorph<Tensor4x4Symx4x4SymComplex<double>, SpatialTemporal::Spatial>(
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
//     int sign
// );
// template void FourierClass<double>::performFourierTransformPolymorph<Tensor4x4Symx4x4SymComplex<double>, SpatialTemporal::Temporal>(
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
//     LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
//     int sign
// );

template void FourierClass<double>::performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<SpatialTemporal::Both>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<SpatialTemporal::Spatial>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
    int sign
);
template void FourierClass<double>::performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<SpatialTemporal::Temporal>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
    int sign
);
