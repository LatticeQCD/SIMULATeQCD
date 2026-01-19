#include "fourierNon2.h"
#include "source.h"
#include "../base/stopWatch.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif


///////////

template<typename floatT>
template<size_t HaloDepth>
void FourierClass<floatT>::moveSpinor1212ToContainer(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    int spincolor1, int spincolor2
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
        copySpinorToContainerLocal<floatT, HaloDepth><<<gridDim, blockDim>>>(redBase.getAccessor(), spinor_in.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt);
    #elif defined USE_HIP
        hipLaunchKernelGGL((copySpinorToContainerLocal<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_in.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt);
    #endif

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<typename floatT>
void FourierClass<floatT>::moveEMTComponentToContainer(
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> & emt,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
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
        copyEMTComponentToContainerLocal<floatT><<<gridDim, blockDim>>>(redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lx, (int) ly, (int) lz, (int) lt);
    #endif

    gpuErr = gpuGetLastError();
    if (gpuErr) {
        GpuError("Failed to launch kernel", gpuErr);
    }

}

template<typename floatT>
void FourierClass<floatT>::moveTensor4x4Symx4x4SymComplexComponentToContainer(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> & tensor,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
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
struct copyToContainer {

    LatticeContainerAccessor _tensorAccessor;
    int _firstIndexPair;
    typedef GIndexer<All> GInd;

    copyToContainer(LatticeContainerAccessor tensorAccessor, int firstIndexPair) : _tensorAccessor(tensorAccessor) {
        _firstIndexPair = firstIndexPair;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT> operator()(gSite site) {

        Tensor4x4Symx4x4SymComplex<floatT> tensor(_tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site));

        return tensor.getSecondMatrix4x4SymComplex(_firstIndexPair);
    }

};

template<class floatT>
struct copyFromContainer {

    LatticeContainerAccessor _matrixAccessor;
    LatticeContainerAccessor _tensorAccessor;
    int _firstIndexPair;
    typedef GIndexer<All> GInd;

    copyFromContainer(LatticeContainerAccessor matrixAccessor, LatticeContainerAccessor tensorAccessor, int firstIndexPair) :
        _matrixAccessor(matrixAccessor), _tensorAccessor(tensorAccessor) {
        _firstIndexPair = firstIndexPair;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> matrix(_matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> tensor(_tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(site));

        tensor.setSecondMatrix4x4SymComplex(_firstIndexPair, matrix);

        return tensor;
    }

};


template<typename floatT>
template<size_t HaloDepth, int dir>
void FourierClass<floatT>::moveContainerToSpinor1212Direction(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    int spincolor1, int spincolor2
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
            copyContainerToSpinor<floatT, HaloDepth><<<gridDim, blockDim>>>(
                spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0);
        #endif
    }

    if (dir == 1) {
        #ifdef USE_CUDA
            copyContainerToSpinor<floatT, HaloDepth><<<gridDim, blockDim>>>(
                spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0
            );
        #endif
    }

    if (dir == 2) {
        #ifdef USE_CUDA
            copyContainerToSpinor<floatT, HaloDepth><<<gridDim, blockDim>>>(
                spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2]
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2]
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
void FourierClass<floatT>::moveContainerToEMTDirection(
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> &emt,
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

    gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

    if (dir == 0) {
        #ifdef USE_CUDA
            copyContainerToEMT<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToEMT<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), emtComponent, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #endif
    }

    if (dir == 1) {
        #ifdef USE_CUDA
            copyContainerToEMT<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToEMT<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), emtComponent, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #endif
    }

    if (dir == 2) {
        #ifdef USE_CUDA
            copyContainerToEMT<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToEMT<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), emtComponent, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #endif
    }

    if (dir == 3) {
        #ifdef USE_CUDA
            copyContainerToEMT<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) emtComponent, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToEMT<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), emtComponent, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
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
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &emt,
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
            copyContainerToTensor4x4Symx4x4SymComplex<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplex<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lxL, (int) ly, (int) lz, (int) lt, mycoords[0], 0, 0, 0
            );
        #endif
    }

    if (dir == 1) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplex<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplex<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) lyL, (int) lz, (int) lt, 0, mycoords[1], 0, 0
            );
        #endif
    }

    if (dir == 2) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplex<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplex<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, mycoords[2], 0
            );
        #endif
    }

    if (dir == 3) {
        #ifdef USE_CUDA
            copyContainerToTensor4x4Symx4x4SymComplex<floatT><<<gridDim, blockDim>>>(
                redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
            );
        #elif defined USE_HIP
            hipLaunchKernelGGL(
                (copyContainerToTensor4x4Symx4x4SymComplex<floatT>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), emt.getAccessor(), (size_t) (lx*ly*lz*lt), (int) firstIndexPair, (int) secondIndexPair, (int) lx, (int) ly, (int) lz, (int) ltL, 0, 0, 0, mycoords[3]
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
    LatticeContainer<true, elemType> &redBaseDevice,
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

            throw std::runtime_error(stdLogger.fatal("Function performFourierTransformDirectionPolymorph does only support gpu topology 1x1x1x1, not ", nodes[0], "x", nodes[1], "x", nodes[2], "x", nodes[3]));

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
template<size_t HaloDepth>
void FourierClass<floatT>::performFourier3DSpinor1212(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    LatticeContainer<false, COMPLEX(floatT)> & redBase2, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign, int maxColorSpin
) {

    redBase.adjustSize(lxL*lyL*lzL*lt);
    redBase2.adjustSize(lxL*lyL*lzL*lt);

    for(int spincolor1 = 0; spincolor1 < maxColorSpin; spincolor1++) {
        for(int spincolor2 = 0; spincolor2 < maxColorSpin; spincolor2++) {

            moveSpinor1212ToContainer(spinor_in, redBase, spincolor1, spincolor2);
            performFourierTransformDirection<0>(redBase, redBase2, sign);
            moveContainerToSpinor1212Direction<HaloDepth, 0>(spinor_out, redBase, spincolor1, spincolor2);

            moveSpinor1212ToContainer(spinor_out, redBase, spincolor1, spincolor2);
            performFourierTransformDirection<1>(redBase, redBase2, sign);
            moveContainerToSpinor1212Direction<HaloDepth, 1>(spinor_out, redBase, spincolor1, spincolor2);

            moveSpinor1212ToContainer(spinor_out, redBase, spincolor1, spincolor2);
            performFourierTransformDirection<2>(redBase, redBase2, sign);
            moveContainerToSpinor1212Direction<HaloDepth, 2>(spinor_out, redBase, spincolor1, spincolor2);

        }
    }

}

template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourier3DEMT(
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> &emtOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    LatticeContainer<true, COMPLEX(floatT)> _redBaseDevice(emtIn.get_CommBase(), "RedBaseDevice", "RedBaseDevice", "RedBaseDevice", "RedBaseDevice");
    LatticeContainer<false, COMPLEX(floatT)> _redBaseHost(emtIn.get_CommBase(), "RedBaseHost", "RedBaseHost", "RedBaseHost", "RedBaseHost");

    _redBaseDevice.adjustSize(lxL*lyL*lzL*ltL); // TODO: Not ltL?
    _redBaseHost.adjustSize(lxL*lyL*lzL*ltL);

    LatticeContainer<true, Matrix4x4SymComplex<floatT>> emtIntermediate(emtIn.get_CommBase(), "emtIntermediate", "emtIntermediate", "emtIntermediate", "emtIntermediate");
    emtIntermediate.adjustSize(GInd::getLatData().vol4);

    emtIntermediate.copyFromLatticeContainer(emtIn);

    for (int emtComponent = 0; emtComponent < 10; emtComponent++) {

        
        if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
            timerMoveIn.start();
            moveEMTComponentToContainer(emtIntermediate, _redBaseDevice, emtComponent);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<0>(_redBaseDevice, _redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToEMTDirection<0>(_redBaseDevice, emtIntermediate, emtComponent);
            timerMoveOut.stop();
            
            timerMoveIn.start();
            moveEMTComponentToContainer(emtIntermediate, _redBaseDevice, emtComponent);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<1>(_redBaseDevice, _redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToEMTDirection<1>(_redBaseDevice, emtIntermediate, emtComponent);
            timerMoveOut.stop();
            
            timerMoveIn.start();
            moveEMTComponentToContainer(emtIntermediate, _redBaseDevice, emtComponent);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<2>(_redBaseDevice, _redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToEMTDirection<2>(_redBaseDevice, emtIntermediate, emtComponent);
            timerMoveOut.stop();
        }
        
        if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
            timerMoveIn.start();
            moveEMTComponentToContainer(emtIntermediate, _redBaseDevice, emtComponent);
            timerMoveIn.stop();
            timerFT.start();
            performFourierTransformDirection<3>(_redBaseDevice, _redBaseHost, sign);
            timerFT.stop();
            timerMoveOut.start();
            moveContainerToEMTDirection<3>(_redBaseDevice, emtIntermediate, emtComponent);
            timerMoveOut.stop();
        }

        timePerComponentMoveIn.push_back(timerMoveIn.seconds());
        timePerComponentFT.push_back(timerFT.seconds());
        timePerComponentMoveOut.push_back(timerMoveOut.seconds());
        // rootLogger.info("       Fourier EMT took ", timer.seconds(), "s for one component.");

        timerMoveIn.reset();
        timerFT.reset();
        timerMoveOut.reset();

    }

    auto const countMoveIn = static_cast<double>(timePerComponentMoveIn.size());
    double timePerComponentAverageMoveIn = std::reduce(timePerComponentMoveIn.begin(), timePerComponentMoveIn.end()) / countMoveIn;
    auto const countFT = static_cast<double>(timePerComponentFT.size());
    double timePerComponentAverageFT = std::reduce(timePerComponentFT.begin(), timePerComponentFT.end()) / countFT;
    auto const countMoveOut = static_cast<double>(timePerComponentMoveOut.size());
    double timePerComponentAverageMoveOut = std::reduce(timePerComponentMoveOut.begin(), timePerComponentMoveOut.end()) / countMoveOut;

    rootLogger.info("       Fourier EMT move-in   took           ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.info("       Fourier EMT scalar FT took           ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.info("       Fourier EMT move-out  took           ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    emtOut.copyFromLatticeContainer(emtIntermediate);

}


template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourier3DTensor4x4Symx4x4SymComplex(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &tensorIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &tensorOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    LatticeContainer<true, COMPLEX(floatT)> _redBaseDevice(tensorIn.get_CommBase(), "RedBaseDevice", "RedBaseDevice", "RedBaseDevice", "RedBaseDevice");
    LatticeContainer<false, COMPLEX(floatT)> _redBaseHost(tensorIn.get_CommBase(), "RedBaseHost", "RedBaseHost", "RedBaseHost", "RedBaseHost");

    _redBaseDevice.adjustSize(lxL*lyL*lzL*ltL); // TODO: Not ltL?
    _redBaseHost.adjustSize(lxL*lyL*lzL*ltL);

    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> tensorIntermediate(tensorIn.get_CommBase(), "tensorIntermediate", "tensorIntermediate", "tensorIntermediate", "tensorIntermediate");
    tensorIntermediate.adjustSize(GInd::getLatData().vol4);
    
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
            // rootLogger.info("       Fourier 4x4Symx4x4Sym took ", timer.seconds(), "s for one component.");

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

    rootLogger.info("       Fourier 4x4symx4x4sym move-in   took ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.info("       Fourier 4x4symx4x4sym scalar FT took ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.info("       Fourier 4x4symx4x4sym move-out  took ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    tensorOut.copyFromLatticeContainer(tensorIntermediate);

}

template<typename floatT>
template<SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourier3DHalfPolymorph(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &tensorIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &tensorOut,
    int sign
) {

    std::vector<double> timePerComponentMoveIn;
    std::vector<double> timePerComponentFT;
    std::vector<double> timePerComponentMoveOut;
    StopWatch<true> timerMoveIn;
    StopWatch<true> timerFT;
    StopWatch<true> timerMoveOut;

    typedef GIndexer<All> GInd;

    LatticeContainer<true, Matrix4x4SymComplex<floatT>> _redBaseDevice(tensorIn.get_CommBase(), "RedBaseDeviceHalf", "RedBaseDeviceHalf", "RedBaseDeviceHalf", "RedBaseDeviceHalf");

    _redBaseDevice.adjustSize(lxL*lyL*lzL*ltL); // TODO: Not ltL?
    // _redBaseDevice.adjustSize(GInd::getLatData().vol4); // TODO: Not ltL?

    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> tensorIntermediate(tensorIn.get_CommBase(), "tensorIntermediateHalf", "tensorIntermediateHalf", "tensorIntermediateHalf", "tensorIntermediateHalf");
    tensorIntermediate.adjustSize(GInd::getLatData().vol4);
    
    tensorIntermediate.copyFromLatticeContainer(tensorIn);

    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {

        timerMoveIn.start();
        _redBaseDevice.template iterateOverBulk<All, 0>(copyToContainer<floatT>(tensorIntermediate.getAccessor(), firstIndexPair));
        timerMoveIn.stop();

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

        timerMoveOut.start();
        tensorIntermediate.template iterateOverBulk<All, 0>(copyFromContainer<floatT>(_redBaseDevice.getAccessor(), tensorIntermediate.getAccessor(), firstIndexPair));
        timerMoveOut.stop();

        timePerComponentMoveIn.push_back(timerMoveIn.seconds());
        timePerComponentFT.push_back(timerFT.seconds());
        timePerComponentMoveOut.push_back(timerMoveOut.seconds());

        timerMoveIn.reset();
        timerFT.reset();
        timerMoveOut.reset();

    }

    auto const countMoveIn = static_cast<double>(timePerComponentMoveIn.size());
    double timePerComponentAverageMoveIn = std::reduce(timePerComponentMoveIn.begin(), timePerComponentMoveIn.end()) / countMoveIn;
    auto const countFT = static_cast<double>(timePerComponentFT.size());
    double timePerComponentAverageFT = std::reduce(timePerComponentFT.begin(), timePerComponentFT.end()) / countFT;
    auto const countMoveOut = static_cast<double>(timePerComponentMoveOut.size());
    double timePerComponentAverageMoveOut = std::reduce(timePerComponentMoveOut.begin(), timePerComponentMoveOut.end()) / countMoveOut;

    rootLogger.info("       Fourier 4x4symx4x4sym move-in   took ", timePerComponentAverageMoveIn, "s on average over ", countMoveIn, " components.");
    rootLogger.info("       Fourier 4x4symx4x4sym scalar FT took ", timePerComponentAverageFT, "s on average over ", countFT, " components.");
    rootLogger.info("       Fourier 4x4symx4x4sym move-out  took ", timePerComponentAverageMoveOut, "s on average over ", countMoveOut, " components.");

    tensorOut.copyFromLatticeContainer(tensorIntermediate);

}

template<typename floatT>
template<typename elemType, SpatialTemporal spatialTemporal>
void FourierClass<floatT>::performFourier3DPolymorph(
    LatticeContainer<true, elemType> &latticeIn,
    LatticeContainer<true, elemType> &latticeOut,
    int sign
) {

    typedef GIndexer<All> GInd;

    LatticeContainer<true, elemType> latticeIntermediate(latticeIn.get_CommBase(), "latticeIntermediate", "latticeIntermediate", "latticeIntermediate", "latticeIntermediate");
    latticeIntermediate.adjustSize(GInd::getLatData().vol4);

    latticeIntermediate.copyFromLatticeContainer(latticeIn);

    rootLogger.info("latticeIntermediate initiated!");
    
    if (spatialTemporal == SpatialTemporal::Spatial || spatialTemporal == SpatialTemporal::Both) {
        performFourierTransformDirectionPolymorph<0, elemType>(latticeIntermediate, sign);
        performFourierTransformDirectionPolymorph<1, elemType>(latticeIntermediate, sign);
        performFourierTransformDirectionPolymorph<2, elemType>(latticeIntermediate, sign);
    }
    
    if (spatialTemporal == SpatialTemporal::Temporal || spatialTemporal == SpatialTemporal::Both) {
        performFourierTransformDirectionPolymorph<3, elemType>(latticeIntermediate, sign);
    }

    latticeOut.copyFromLatticeContainer(latticeIntermediate);

}


////////////

template<class floatT, size_t HaloDepth>
void fourier3D(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
    LatticeContainer<true, COMPLEX(floatT)> & redBase,
    LatticeContainer<false, COMPLEX(floatT)> & redBase2,
    CommunicationBase & commBase,
    int sign, int maxColorSpin
) {

    StopWatch<true> timer;

    MPI_Comm commX, commY, commZ;
    int remain[4];
    remain[0] = 1;
    remain[1] = 0;
    remain[2] = 0;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(), remain, &commX);
    remain[0] = 0;
    remain[1] = 1;
    remain[2] = 0;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(), remain, &commY);
    remain[0] = 0;
    remain[1] = 0;
    remain[2] = 1;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(), remain, &commZ);

    typedef GIndexer<All, 0> GInd;
    size_t lx = GInd::getLatData().lx;
    size_t ly = GInd::getLatData().ly;
    size_t lz = GInd::getLatData().lz;
    size_t lt = GInd::getLatData().lt;

    size_t elems;
    dim3 blockDim;
    blockDim.x = 32; // TODO: Why these dimensions?
    blockDim.y = 1;
    blockDim.z = 1;

    dim3 gridDim;

    size_t lxL = GInd::getLatData().globLX;
    size_t lyL = GInd::getLatData().globLY;
    size_t lzL = GInd::getLatData().globLZ;

    size_t lsX = lxL;
    size_t lsY = lyL;
    size_t lsZ = lzL;

    while(abs(round( ((floatT) lsX)/2.0 )-(floatT) (lsX/2) ) < 0.00001) {
        lsX = lsX/2;
    }
    while(abs(round( ((floatT) lsY)/2.0 )-(floatT) (lsY/2) ) < 0.00001) {
        lsY = lsY/2;
    }
    while(abs(round( ((floatT) lsZ)/2.0 )-(floatT) (lsZ/2) ) < 0.00001) {
        lsZ = lsZ/2;
    }

    // std::cout << "lsX " << lsX << " lsY " << lsY << " lsZ " << lsZ << std::endl;

    redBase.adjustSize(lxL*lyL*lzL*lt);
    redBase2.adjustSize(lxL*lyL*lzL*lt);

    for(int spincolor1 = 0; spincolor1 < maxColorSpin; spincolor1 ++) {
        for(int spincolor2 = 0; spincolor2 < maxColorSpin; spincolor2 ++) {

            gpuError_t gpuErr;

            // TODO: Can't the three x,y,z parts be combined?
            // start x direction

            // copy information from spinor over to redbase 
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

            #ifdef USE_CUDA
                copySpinorToContainerLocal<floatT, HaloDepth><<<gridDim, blockDim>>>(
                    redBase.getAccessor(), spinor_in.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt
                );
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copySpinorToContainerLocal<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_in.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt
                );
            #endif

            // FourierClass<floatT> fourierClass(commBase);
            // fourierClass.moveSpinor1212ToContainer(spinor_in, redBase, spincolor1, spincolor2);

            if ( commBase.nodes()[0] > 1 ) {

                gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

                gatherHostXYZ<floatT, 0>((std::complex<floatT> *) redBase2.get_ContainerArrayPtr()->getPointer(), commX, lxL, ly, lz);

                gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*commBase.nodes()[0]*elems, gpuMemcpyHostToDevice);

                gpuErr = gpuGetLastError();
                if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

            }

            // perform the fourier transformation in x direction
            elems = ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

            #ifdef USE_CUDA
                fourier<floatT, 0><<<gridDim, blockDim>>>(redBase.getAccessor(), redBase.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
            #elif defined USE_HIP
                hipLaunchKernelGGL((fourier<floatT, 0>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), redBase.getAccessor(), elems, ly, lz, lxL, lt, lsX, sign);
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);
            
            // fourierClass.template performFourierTransformDirection<0>(redBase, redBase2, sign);
            
            // move back into spinor
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

            #ifdef USE_CUDA
                copyContainerToSpinor<floatT, HaloDepth><<< gridDim, blockDim>>>(
                    spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lxL, (int) ly, (int) lz, (int) lt, commBase.mycoords()[0], 0, 0
                );
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(),  redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lxL, (int) ly, (int) lz, (int) lt, commBase.mycoords()[0], 0, 0
                );
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

            // fourierClass.template moveContainerToSpinor1212Direction<HaloDepth, 0>(spinor_out, redBase, spincolor1, spincolor2);

            // start y direction

            // copy information from spinor over to redbase 
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x))); // TODO: Why also blockDim.x, not .y?

            #ifdef USE_CUDA
                copySpinorToContainerLocal<floatT, HaloDepth><<<gridDim, blockDim>>>(
                    redBase.getAccessor(), spinor_out.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt
                ); // TODO: Why is it spin_out here?
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copySpinorToContainerLocal<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_out.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt
                ); // TODO: Why is it spin_out here?
            #endif

            // fourierClass.moveSpinor1212ToContainer(spinor_out, redBase, spincolor1, spincolor2);

            if ( commBase.nodes()[1] > 1 ) {

                gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

                gatherHostXYZ<floatT, 1>((std::complex<floatT> *) redBase2.get_ContainerArrayPtr()->getPointer(), commY, lx, lyL, lz);

                gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*commBase.nodes()[1]*elems, gpuMemcpyHostToDevice);

                gpuErr = gpuGetLastError();
                if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

            }

            // perform the fourier transformation in y direction
            elems = lx*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x))); // TODO: Why blockdim.x, not .y?

            #ifdef USE_CUDA
                fourier<floatT, 1><<<gridDim, blockDim>>>(redBase.getAccessor(), redBase.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
            #elif defined USE_HIP
                hipLaunchKernelGGL((fourier<floatT, 1>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), redBase.getAccessor(), elems, lx, lz, lyL, lt, lsY, sign);
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

            // fourierClass.template performFourierTransformDirection<1>(redBase, redBase2, sign);

            // move back into spinor
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x))); // TODO: Why blockdim.x, not .y?

            #ifdef USE_CUDA
                copyContainerToSpinor<floatT, HaloDepth><<< gridDim, blockDim>>>(
                    spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) lyL, (int) lz, (int) lt, 0, commBase.mycoords()[1], 0
                );
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) lyL, (int) lz, (int) lt, 0, commBase.mycoords()[1], 0
                );
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr)GpuError("performFunctor: Failed to launch kernel", gpuErr);

            // fourierClass.template moveContainerToSpinor1212Direction<HaloDepth, 1>(spinor_out, redBase, spincolor1, spincolor2);

            // start z direction

            // copy information from spinor over to redbase 
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

            #ifdef USE_CUDA
                copySpinorToContainerLocal<floatT, HaloDepth><<< gridDim, blockDim>>>(
                    redBase.getAccessor(), spinor_out.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) lt
                );
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copySpinorToContainerLocal<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_out.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lz, (int) ltL
                );
            #endif
            
            // fourierClass.moveSpinor1212ToContainer(spinor_in, redBase, spincolor1, spincolor2);

            
            if ( commBase.nodes()[2] > 1 ) {

                gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

                gatherHostXYZ<floatT, 2>((std::complex<floatT> *) redBase2.get_ContainerArrayPtr()->getPointer(), commZ, lx, ly, lzL);
                // gatherAllHost((std::complex<floatT> *) redBase2.get_ContainerArrayPtr()->getPointer(), commBase);

                gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(), sizeof(COMPLEX(floatT))*commBase.nodes()[2]*elems, gpuMemcpyHostToDevice);

                gpuErr = gpuGetLastError();
                if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);

            }
            
            // perform the fourier transformation in z direction
            elems = lx*ly*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));
            
            #ifdef USE_CUDA
                fourier<floatT, 2><<<gridDim, blockDim>>>(redBase.getAccessor(), redBase.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
            #elif defined USE_HIP
                hipLaunchKernelGGL((fourier<floatT, 2>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), redBase.getAccessor(), elems, lx, ly, lzL, lt, lsZ, sign);
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);
            
            // fourierClass.template performFourierTransformDirection<2>(redBase, redBase2, sign);
            
            // move back into spinor
            elems = lx*ly*lz*lt;
            gridDim = static_cast<int> (ceilf(static_cast<float> (elems) / static_cast<float> (blockDim.x)));

            #ifdef USE_CUDA
                copyContainerToSpinor<floatT, HaloDepth><<< gridDim, blockDim>>>(
                    spinor_out.getAccessor(), redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, commBase.mycoords()[2]
                );
            #elif defined USE_HIP
                hipLaunchKernelGGL(
                    (copyContainerToSpinor<floatT, HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, spinor_out.getAccessor(),  redBase.getAccessor(), (size_t) (lx*ly*lz*lt), spincolor1, spincolor2, (int) lx, (int) ly, (int) lzL, (int) lt, 0, 0, commBase.mycoords()[2]
                );
            #endif

            gpuErr = gpuGetLastError();
            if (gpuErr) GpuError("performFunctor: Failed to launch kernel", gpuErr);
            
            // fourierClass.template moveContainerToSpinor1212Direction<HaloDepth, 2>(spinor_out, redBase, spincolor1, spincolor2);

        }

    }

    // std::cout << "Finished " << std::endl;

}

////////////

template<typename floatT, bool onDevice, size_t HaloDepthSpin>
void tr_spinorXspinor(
    Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;
    size_t _elems = GInd::getLatData().vol4;
    ReadIndex<All, HaloDepthSpin> index;

    iterateFunctorNoReturn<true, BLOCKSIZE>(Tr_SpinorXspinor<floatT, HaloDepthSpin, 12>(spinorInDagger, spinorIn), index, _elems);

}


/// val = S_in * S_in but only at spatial time t
template<typename floatT, bool onDevice, size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerM(
    int t,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
    LatticeContainer<true, COMPLEX(floatT)> & _redBase
) {

        typedef GIndexer<All, HaloDepthSpin> GInd;

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol3;

        _redBase.adjustSize(elems_);

        _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(SumXYZ_TrMdaggerM2<floatT, HaloDepthSpin, 12>(t, spinorInDagger, spinorIn));

        _redBase.reduce(result, elems_);
        return result;

}


/// val = S_in * S_in but only at spatial time t
template<typename floatT, bool onDevice, size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerMwave(
    int t,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 3, 1> & spinor_wave,
    LatticeContainer<true, COMPLEX(floatT)> & _redBase,
    int time, int col, int conjON
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;

    COMPLEX(double) result = 0;

    size_t elems_ = GInd::getLatData().vol3;

    _redBase.adjustSize(elems_);

    if(conjON == 2) {
        _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin, 12, 2>(t, spinorInDagger, spinorIn, spinor_wave, time, col));
    }
    else if(conjON == 1) {
        _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin, 12, 1>(t, spinorInDagger, spinorIn, spinor_wave, time, col));
    }
    else{
        _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin, 12, 0>(t, spinorInDagger, spinorIn, spinor_wave, time, col));
    }

    _redBase.reduce(result, elems_);
    return result;

}

template<typename floatT, size_t HaloDepthSpin>
void loadWave(
    std::string fname,
    Spinorfield<floatT, true, All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    int time, int col,
    CommunicationBase & commBase
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();
    global[3] = 1;
    local[3] = 1;

    commBase.initIOBinary(fname, 0, 2*sizeof(floatT), 0, global, local, READ);

    std::vector<char> buf;
    buf.resize(local[0]*local[1]*local[2]*2*sizeof(floatT));
    commBase.readBinary(&buf[0], local[0]*local[1]*local[2]);
    int ps = 0;
    Vect3<floatT> tmp3;
    // for ( int i = 0; i < 3; i ++) {
        // tmp3.data[i] = 0.0;
    // }
    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++)
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        floatT *dataRe = (floatT *) &buf[ps];
        ps += sizeof(floatT);
        floatT *dataIm = (floatT *) &buf[ps];
        ps += sizeof(floatT);
        tmp3.data[col] = COMPLEX(floatT) (dataRe[0], dataIm[0]);
        //std::cout << "data " << data[0] << std::endl;
        spinor_host.getAccessor().setElement(GInd::getSite(x, y, z, time), tmp3);
    }

    commBase.closeIOBinary();

    spinor_device = spinor_host;
    spinor_device.updateAll();
}

template<typename floatT, size_t HaloDepthSpin>
void moveWave(
    Spinorfield<floatT, true, All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    int posX, int posY, int posZ, int timeOut, int colOut, int timeIn, int colIn ,
    CommunicationBase & commBase
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;


    int coord[4];
    // all gather 4d
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

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz];
    std::complex<floatT> *buf2 = new std::complex<floatT>[glx*gly*glz];

        spinor_host = spinor_device;

        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            buf[x+lx*(y+ly*(z))] = std::complex<floatT>(
                ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, timeIn))).data[colIn]).cREAL,
                ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, timeIn))).data[colIn]).cIMAG
            );
        }


    if(std::is_same<floatT, double>::value) {
        MPI_Allgather(buf, lx*ly*lz, MPI_DOUBLE_COMPLEX, buf2, lx*ly*lz, MPI_DOUBLE_COMPLEX, commBase.getCart_comm());
    } else if(std::is_same<floatT, float>::value) {
        MPI_Allgather(buf, lx*ly*lz, MPI_COMPLEX, buf2, lx*ly*lz, MPI_COMPLEX, commBase.getCart_comm());
    }
    

    for (int r=0; r<rankSize; r++) {
        MPI_Cart_coords(commBase.getCart_comm(), r, 4, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            buf[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])))] = buf2[x+lx*(y+ly*(z+lz*(r)))];
        }
    }

    for (int z=0; z<lz; z++)
    for (int y=0; y<ly; y++)
    for (int x=0; x<lx; x++) {
        Vect3<floatT> tmp3 = spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, timeOut));
        tmp3.data[colOut] = COMPLEX(floatT) (
            real(buf[((x+lx*commBase.mycoords()[0]+glx-posX)%glx)+glx*(((y+ly*commBase.mycoords()[1]+gly-posY)%gly)+gly*(((z+lz*commBase.mycoords()[2]+glz-posZ)%glz)))]),
            imag(buf[((x+lx*commBase.mycoords()[0]+glx-posX)%glx)+glx*(((y+ly*commBase.mycoords()[1]+gly-posY)%gly)+gly*(((z+lz*commBase.mycoords()[2]+glz-posZ)%glz)))])
        );
        spinor_host.getAccessor().setElement(GInd::getSite(x, y, z, timeOut), tmp3);
    }

    spinor_device = spinor_host;
    delete[] buf;
    delete[] buf2;

    spinor_device.updateAll();

}

template<typename floatT, size_t HaloDepthSpin>
void gatherMomentum(
    COMPLEX(floatT) * CC, Spinorfield<floatT, true, All, HaloDepthSpin, 12, 12> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 12, 12> & spinor_host,
    int timeIn, int colIn , int savePos, int nMomentum,
    CommunicationBase & commBase
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;

    int coord[4];
    // all gather 4d
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

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz];
    std::complex<floatT> *buf2 = new std::complex<floatT>[glx*gly*glz];

    spinor_host = spinor_device;

    for (int z=0; z<lz; z++)
    for (int y=0; y<ly; y++)
    for (int x=0; x<lx; x++) {
        buf[x+lx*(y+ly*(z))] = std::complex<floatT>(
            ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, timeIn))).data[colIn]).cREAL,
            ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, timeIn))).data[colIn]).cIMAG
        );
    }


    if(std::is_same<floatT, double>::value) {
        MPI_Allgather(buf, lx*ly*lz, MPI_DOUBLE_COMPLEX, buf2, lx*ly*lz, MPI_DOUBLE_COMPLEX, commBase.getCart_comm());
    }
    else if(std::is_same<floatT, float>::value) {
        MPI_Allgather(buf, lx*ly*lz, MPI_COMPLEX, buf2, lx*ly*lz, MPI_COMPLEX, commBase.getCart_comm());
    }


    for (int r=0; r<rankSize; r++) {
        MPI_Cart_coords(commBase.getCart_comm(), r, 4, coord);
        // for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            buf[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])))] = buf2[x+lx*(y+ly*(z+lz*(r)))];
        }
    }
    
    int ktotal = -1;
    for(int kz = -1; kz < 2; kz ++)
    for(int ky = -1; ky < 2; ky ++)
    for(int kx = -1; kx < 2; kx ++) {
        ktotal++;
        CC[ktotal + savePos] =  CC[ktotal + savePos]+ COMPLEX(floatT) (real(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))]),
                                                                       imag(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))]));
    }
    
    delete[] buf;
    delete[] buf2;

}

template<typename floatT, size_t HaloDepthSpin>
void gatherMomentumT(
    COMPLEX(floatT) * CC, Spinorfield<floatT, true, All, HaloDepthSpin, 12, 12> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 12, 12> & spinor_host,
    int colIn , int savePos, int nP, int * pos,
    CommunicationBase & commBase
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;

    int coord[4];
    // all gather 4d
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

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz];
    std::complex<floatT> *buf2 = new std::complex<floatT>[glx*gly*glz];

    spinor_host = spinor_device;

    for (int t=0; t<glt; t++) {

        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++) {
            buf[x+lx*(y+ly*(z))] = std::complex<floatT>(
                ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, t))).data[colIn]).cREAL,
                ((spinor_host.getAccessor().getElement(GInd::getSite(x, y, z, t))).data[colIn]).cIMAG
            );
        }


        if(std::is_same<floatT, double>::value) {
            MPI_Allgather(buf, lx*ly*lz, MPI_DOUBLE_COMPLEX, buf2, lx*ly*lz, MPI_DOUBLE_COMPLEX, commBase.getCart_comm());
        }
        else if(std::is_same<floatT, float>::value) {
            MPI_Allgather(buf, lx*ly*lz, MPI_COMPLEX, buf2, lx*ly*lz, MPI_COMPLEX, commBase.getCart_comm());
        }


        for (int r=0; r<rankSize; r++) {
            MPI_Cart_coords(commBase.getCart_comm(), r, 4, coord);
            // for (int t=0; t<lt; t++)
            for (int z=0; z<lz; z++)
            for (int y=0; y<ly; y++)
            for (int x=0; x<lx; x++) {
                buf[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])))] = buf2[x+lx*(y+ly*(z+lz*(r)))];
            }
        }

        int ktotal = -1;
        for(int kz = -nP; kz < nP+1; kz ++)
        for(int ky = -nP; ky < nP+1; ky ++) 
        for(int kx = -nP; kx < nP+1; kx ++) {
            ktotal++;
            floatT phase = -2.0*M_PI*((floatT) (kx*pos[0])/glx+(floatT) (ky*pos[1])/gly+(floatT) (kz*pos[2])/glz);
            CC[(t-pos[3]+glt)%(glt)+glt*ktotal + savePos] = CC[(t-pos[3]+glt)%(glt)+glt*ktotal + savePos] + COMPLEX(floatT) (real(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))]),
                                                                                                                            imag(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))])) *
                                                                                                            COMPLEX(floatT) (cos(phase), sin(phase));
        CC[(t-pos[3]+glt)%(glt)+glt*ktotal + savePos] =  CC[(t-pos[3]+glt)%(glt)+glt*ktotal + savePos] + COMPLEX(floatT) (real(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))]),
                                                                                                                            imag(buf[(kx+glx)%glx+glx*((ky+gly)%gly+gly*((kz+glz)%glz))]));
        }

        }

    delete[] buf;
    delete[] buf2;

}


template<typename floatT, size_t HaloDepthSpin>
void loadWavePos(
    std::string fname,
    Spinorfield<floatT, true, All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    size_t posX, size_t posY, size_t posZ,
    int time, int col,
    CommunicationBase & commBase
) {

    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();
    global[3] = 1;
    // local[3] = 1;

    commBase.initIOBinary(fname, 0, 2*sizeof(floatT), 0, global, global, READ);

    std::vector<char> buf;
    buf.resize(global[0]*global[1]*global[2]*2*sizeof(floatT));
    commBase.readBinary(&buf[0], global[0]*global[1]*global[2]);
    int ps = 0;
    Vect3<floatT> tmp3;

    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++)
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        size_t xg = (x+commBase.mycoords()[0]*local[0]+posX)%global[0];
        size_t yg = (y+commBase.mycoords()[1]*local[1]+posY)%global[1];
        size_t zg = (z+commBase.mycoords()[2]*local[2]+posZ)%global[2];
        
        ps = 2*sizeof(floatT)*(xg+global[0]*(yg+global[1]*(zg)));
        floatT *dataRe = (floatT *) &buf[ps];
        ps += sizeof(floatT);
        floatT *dataIm = (floatT *) &buf[ps];
        tmp3.data[col] = COMPLEX(floatT) (dataRe[0], dataIm[0]);
        
        std::cout << "x "<< xg << " y "<< yg << " z "<< zg <<" dataRe " << dataRe[0] << " dataIm " << dataIm[0] << std::endl;
        
        spinor_host.getAccessor().setElement(GInd::getSite(x, y, z, time), tmp3);
    }

    commBase.closeIOBinary();

    spinor_device = spinor_host;

}

template<typename floatT, size_t HaloDepth>
void makeWaveSource(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
    const Spinorfield<floatT, true, All, HaloDepth, 3, 1> &spinor_wave,
    size_t time, size_t col, size_t post
) {

    typedef GIndexer<All, HaloDepth> GInd;
    size_t _elems = GInd::getLatData().vol4;
    ReadIndex<All, HaloDepth> index;

    iterateFunctorNoReturn<true, BLOCKSIZE>(MakeWaveSource12<floatT, HaloDepth>( spinorIn, spinor_wave, time, col, post), index, _elems);

    spinorIn.updateAll();

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

template void FourierClass<double>::moveSpinor1212ToContainer<2>(Spinorfield<double, true, All, 2, 12, 12> & spinor_in, LatticeContainer<true, COMPLEX(double)> & redBase, int spincolor1, int spincolor2);

template void FourierClass<double>::moveContainerToSpinor1212Direction<2, 0>(Spinorfield<double, true, All, 2, 12, 12> & spinor_out, LatticeContainer<true, COMPLEX(double)> & redBase, int spincolor1, int spincolor2);
template void FourierClass<double>::moveContainerToSpinor1212Direction<2, 1>(Spinorfield<double, true, All, 2, 12, 12> & spinor_out, LatticeContainer<true, COMPLEX(double)> & redBase, int spincolor1, int spincolor2);
template void FourierClass<double>::moveContainerToSpinor1212Direction<2, 2>(Spinorfield<double, true, All, 2, 12, 12> & spinor_out, LatticeContainer<true, COMPLEX(double)> & redBase, int spincolor1, int spincolor2);

template void FourierClass<double>::moveContainerToEMTDirection<0>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToEMTDirection<1>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToEMTDirection<2>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);
template void FourierClass<double>::moveContainerToEMTDirection<3>(LatticeContainer<true, COMPLEX(double)> & redBase, LatticeContainer<true, Matrix4x4SymComplex<double>> &emt, int emtComponent);

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

template void FourierClass<double>::performFourierTransformDirectionPolymorph<0, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<1, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<2, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);
template void FourierClass<double>::performFourierTransformDirectionPolymorph<3, Tensor4x4Symx4x4SymComplex<double>>(LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> & redBase, int sign);


template void FourierClass<double>::performFourier3DSpinor1212<2>( // what is this last <2> here? because it's the standard HaloDepth
    Spinorfield<double, true, All, 2, 12, 12> & spinor_in,
    Spinorfield<double, true, All, 2, 12, 12> & spinor_out,
    LatticeContainer<true, COMPLEX(double)> & redBase,
    LatticeContainer<false, COMPLEX(double)> & redBase2,
    int sign, int maxColorSpin
);

template void FourierClass<double>::performFourier3DEMT<SpatialTemporal::Spatial>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DEMT<SpatialTemporal::Both>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DEMT<SpatialTemporal::Temporal>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);

template void FourierClass<double>::performFourier3DTensor4x4Symx4x4SymComplex<SpatialTemporal::Spatial>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DTensor4x4Symx4x4SymComplex<SpatialTemporal::Both>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DTensor4x4Symx4x4SymComplex<SpatialTemporal::Temporal>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &emtOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DPolymorph<Matrix4x4SymComplex<double>, SpatialTemporal::Both>(
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Matrix4x4SymComplex<double>> &latticeOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DPolymorph<Tensor4x4Symx4x4SymComplex<double>, SpatialTemporal::Both>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
    // LatticeContainer<true, COMPLEX(double)> & _redBaseDevice,
    // LatticeContainer<false, COMPLEX(double)> & _redBaseHost, // TODO: Why this not onDevice? Memory on cpu for mpi handling
    int sign
);
template void FourierClass<double>::performFourier3DHalfPolymorph<SpatialTemporal::Both>(
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeIn,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<double>> &latticeOut,
    int sign
);

////

template void fourier3D(
    Spinorfield<double, true, All, 2, 12, 12> & spinor_out,
    Spinorfield<double, true, All, 2, 12, 12> & spinor_in,
    LatticeContainer<true, COMPLEX(double)> & redBase,
    LatticeContainer<false, COMPLEX(double)> & redBase2,
    CommunicationBase & commBase,
    int sign, int maxColorSpin
);


template void tr_spinorXspinor(
    Spinorfield<double, true, All, 2, 12, 12> & spinorInDagger,
    const Spinorfield<double, true, All, 2, 12, 12> & spinorIn
);


template COMPLEX(double) sumXYZ_TrMdaggerM(int t,
    const Spinorfield<double, true, All, 2, 12, 12> & spinorInDagger,
    const Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
    LatticeContainer<true, COMPLEX(double)> & _redBase
);

template COMPLEX(double) sumXYZ_TrMdaggerMwave(int t,
    const Spinorfield<double, true, All, 2, 12, 12> & spinorInDagger,
    const Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
    const Spinorfield<double, true, All, 2, 3 ,  1> & spinor_wave,
    LatticeContainer<true, COMPLEX(double)> & _redBase,
    int time, int col, int conjON
);

template void loadWave(
    std::string fname,
    Spinorfield<double, true , All, 2, 3, 1> & spinor_device,
    Spinorfield<double, false, All, 2, 3, 1> & spinor_host,
    int time, int col,
    CommunicationBase & commBase
);

template void loadWavePos(
    std::string fname,
    Spinorfield<double, true , All, 2, 3, 1> & spinor_device,
    Spinorfield<double, false, All, 2, 3, 1> & spinor_host,
    size_t posX, size_t posY, size_t posZ,
    int time, int col,
    CommunicationBase & commBase
);

template void makeWaveSource(
    Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
    const Spinorfield<double, true, All, 2, 3, 1> &spinor_wave,
    size_t time, size_t col, size_t post
);

template void moveWave(
    Spinorfield<double, true, All, 2, 3, 1> & spinor_device,
    Spinorfield<double, false, All, 2, 3, 1> & spinor_host,
    int posX, int posY, int posZ,
    int timeOut, int colOut, int timeIn, int colIn,
    CommunicationBase & commBase
);

template void gatherMomentum(
    COMPLEX(double) * CC,
    Spinorfield<double, true, All, 2, 12, 12> & spinor_device,
    Spinorfield<double, false, All, 2, 12, 12> & spinor_host,
    int timeIn, int colIn, int savePos , int nMomentum,
    CommunicationBase & commBase
);

template void gatherMomentumT(
    COMPLEX(double) * CC,
    Spinorfield<double, true, All, 2, 12, 12> & spinor_device,
    Spinorfield<double, false, All, 2, 12, 12> & spinor_host,
    int colIn, int savePos , int nP, int * pos,
    CommunicationBase & commBase
);

