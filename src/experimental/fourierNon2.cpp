#include "fourierNon2.h"
#include "source.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif

template<class floatT, size_t HaloDepth>
void fourier3D(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,LatticeContainer<true,COMPLEX(floatT)> & redBase,LatticeContainer<false,COMPLEX(floatT)> & redBase2,CommunicationBase & commBase){

    StopWatch<true> timer;

    MPI_Comm commX, commY, commZ;
    int remain[4];
    remain[0] = 1;
    remain[1] = 0;
    remain[2] = 0;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(),remain, &commX);
    remain[0] = 0;
    remain[1] = 1;
    remain[2] = 0;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(),remain, &commY);
    remain[0] = 0;
    remain[1] = 0;
    remain[2] = 1;
    remain[3] = 0;
    MPI_Cart_sub(commBase.getCart_comm(),remain, &commZ);


    typedef GIndexer<All,0> GInd;
    size_t lx = GInd::getLatData().lx;
    size_t ly = GInd::getLatData().ly;
    size_t lz = GInd::getLatData().lz;
    size_t lt = GInd::getLatData().lt;

    size_t elems;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1;
    blockDim.z = 1;

    dim3 gridDim;


    size_t lxL = GInd::getLatData().globLX;
    size_t lyL = GInd::getLatData().globLY;
    size_t lzL = GInd::getLatData().globLZ;

    size_t lsX = lxL;
    size_t lsY = lyL;
    size_t lsZ = lzL;
    while(abs(round(  ((floatT)lsX)/2.0 )-(floatT)(lsX/2)  ) < 0.00001){
        lsX = lsX/2;
    }
    while(abs(round(  ((floatT)lsY)/2.0 )-(floatT)(lsY/2)  ) < 0.00001){
        lsY = lsY/2;
    }
    while(abs(round(  ((floatT)lsZ)/2.0 )-(floatT)(lsZ/2)  ) < 0.00001){
        lsZ = lsZ/2;
    }

    std::cout << "lsX " << lsX << " lsY " << lsY << " lsZ " << lsZ << std::endl;

    redBase.adjustSize(lxL*lyL*lzL*lt);
    redBase2.adjustSize(lxL*lyL*lzL*lt);

    for(int spincolor1 =0; spincolor1 < 12; spincolor1 ++){
        for(int spincolor2 =0; spincolor2 < 12; spincolor2 ++){

    // start x direction
    gpuError_t gpuErr;
    // copy information from spinor over to redbase 
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));
    
    #ifdef USE_CUDA
    copySpinorToContainerLocal<floatT,HaloDepth><<< gridDim, blockDim>>>(redBase.getAccessor(), spinor_in.getAccessor(), 
                                                                        (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #elif defined USE_HIP
    hipLaunchKernelGGL((copySpinorToContainerLocal<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_in.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #endif
     
    if( commBase.nodes()[0] > 1 ){

       gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

       gatherHostXYZ<floatT,0>((std::complex<floatT> *)redBase2.get_ContainerArrayPtr()->getPointer(),commX,lxL,ly,lz);

       gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*commBase.nodes()[0]*elems, gpuMemcpyHostToDevice);

       gpuErr = gpuGetLastError();
          if (gpuErr)
             GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    // perform the fourier transformation in x direction
    elems = ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));
#ifdef USE_CUDA
    fourier<floatT,0><<<gridDim, blockDim>>>(redBase.getAccessor(),redBase.getAccessor(),elems,ly,lz,lxL,lt,lsX);
#elif defined USE_HIP
    hipLaunchKernelGGL((fourier<floatT,0>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(),redBase.getAccessor(),elems,ly,lz,lxL,lt,lsX);
#endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);

    // move back into spinor
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));


        #ifdef USE_CUDA
        copyContainerToSpinor<floatT,HaloDepth><<< gridDim, blockDim>>>(spinor_out.getAccessor(),redBase.getAccessor(),
                                                                       (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lxL,(int)ly,(int)lz,(int)lt,
                                                                       commBase.mycoords()[0],0,0);
        #elif defined USE_HIP
        hipLaunchKernelGGL((copyContainerToSpinor<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim),0,0, spinor_out.getAccessor(),  redBase.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lxL,(int)ly,(int)lz,(int)lt,
                                                                      commBase.mycoords()[0],0,0);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);

    // start y direction

    // copy information from spinor over to redbase 
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

    #ifdef USE_CUDA
    copySpinorToContainerLocal<floatT,HaloDepth><<< gridDim, blockDim>>>(redBase.getAccessor(), spinor_out.getAccessor(),
                                                                        (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #elif defined USE_HIP
    hipLaunchKernelGGL((copySpinorToContainerLocal<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_out.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #endif

    if( commBase.nodes()[1] > 1 ){

       gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

       gatherHostXYZ<floatT,1>((std::complex<floatT> *)redBase2.get_ContainerArrayPtr()->getPointer(),commY,lx,lyL,lz);

       gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*commBase.nodes()[1]*elems, gpuMemcpyHostToDevice);

       gpuErr = gpuGetLastError();
          if (gpuErr)
             GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    // perform the fourier transformation in y direction
    elems = lx*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));
#ifdef USE_CUDA
    fourier<floatT,1><<<gridDim, blockDim>>>(redBase.getAccessor(),redBase.getAccessor(),elems,lx,lz,lyL,lt,lsY);
#elif defined USE_HIP
    hipLaunchKernelGGL((fourier<floatT,1>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(),redBase.getAccessor(),elems,lx,lz,lyL,lt,lsY);
#endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);

    // move back into spinor
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));


        #ifdef USE_CUDA
        copyContainerToSpinor<floatT,HaloDepth><<< gridDim, blockDim>>>(spinor_out.getAccessor(),redBase.getAccessor(),
                                                                       (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)lyL,(int)lz,(int)lt,
                                                                       0,commBase.mycoords()[1],0);
        #elif defined USE_HIP
        hipLaunchKernelGGL((copyContainerToSpinor<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim),0,0, spinor_out.getAccessor(),  redBase.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)lyL,(int)lz,(int)lt,
                                                                      0,commBase.mycoords()[1],0);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);


    // start z direction

    // copy information from spinor over to redbase 
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)/ static_cast<float> (blockDim.x)));

    #ifdef USE_CUDA
    copySpinorToContainerLocal<floatT,HaloDepth><<< gridDim, blockDim>>>(redBase.getAccessor(), spinor_out.getAccessor(),
                                                                        (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #elif defined USE_HIP
    hipLaunchKernelGGL((copySpinorToContainerLocal<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(), spinor_out.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lz,(int)lt);
    #endif

    if( commBase.nodes()[2] > 1 ){

       gpuMemcpy(redBase2.get_ContainerArrayPtr()->getPointer(), redBase.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*(lx*ly*lz*lt), gpuMemcpyDeviceToHost);

       gatherHostXYZ<floatT,2>((std::complex<floatT> *)redBase2.get_ContainerArrayPtr()->getPointer(),commZ,lx,ly,lzL);
       //gatherAllHost((std::complex<floatT> *)redBase2.get_ContainerArrayPtr()->getPointer(),commBase);

       gpuMemcpy(redBase.get_ContainerArrayPtr()->getPointer(), redBase2.get_ContainerArrayPtr()->getPointer(),sizeof(COMPLEX(floatT))*commBase.nodes()[2]*elems, gpuMemcpyHostToDevice);

       gpuErr = gpuGetLastError();
          if (gpuErr)
             GpuError("performFunctor: Failed to launch kernel", gpuErr);

    }

    // perform the fourier transformation in z direction
    elems = lx*ly*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));
#ifdef USE_CUDA
    fourier<floatT,2><<<gridDim, blockDim>>>(redBase.getAccessor(),redBase.getAccessor(),elems,lx,ly,lzL,lt,lsZ);
#elif defined USE_HIP
    hipLaunchKernelGGL((fourier<floatT,2>), dim3(gridDim), dim3(blockDim), 0, 0, redBase.getAccessor(),redBase.getAccessor(),elems,lx,ly,lzL,lt,lsZ);
#endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);

    // move back into spinor
    elems = lx*ly*lz*lt;
    gridDim = static_cast<int> (ceilf(static_cast<float> (elems)
                    / static_cast<float> (blockDim.x)));


        #ifdef USE_CUDA
        copyContainerToSpinor<floatT,HaloDepth><<< gridDim, blockDim>>>(spinor_out.getAccessor(),redBase.getAccessor(),
                                                                       (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lzL,(int)lt,
                                                                       0,0,commBase.mycoords()[2]);
        #elif defined USE_HIP
        hipLaunchKernelGGL((copyContainerToSpinor<floatT,HaloDepth>), dim3(gridDim), dim3(blockDim),0,0, spinor_out.getAccessor(),  redBase.getAccessor(),
                                                                      (size_t)(lx*ly*lz*lt),spincolor1,spincolor2,(int)lx,(int)ly,(int)lzL,(int)lt,
                                                                      0,0,commBase.mycoords()[2]);
        #endif

        gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);




        }
    }

    //std::cout << "Finished " << std::endl;

}


////////////

/// val = S_in * S_in but only at spatial time t
template<typename floatT, bool onDevice,size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerM(int t,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
        LatticeContainer<true,COMPLEX(floatT)> & _redBase){

        typedef GIndexer<All, HaloDepthSpin> GInd;

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol3;

        _redBase.adjustSize(elems_);

        _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(
                SumXYZ_TrMdaggerM2<floatT, HaloDepthSpin,12>(t, spinorInDagger,spinorIn));

        _redBase.reduce(result, elems_);
        return result;
}


/// val = S_in * S_in but only at spatial time t
template<typename floatT, bool onDevice,size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerMwave(int t,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 3,1> & spinor_wave,
        LatticeContainer<true,COMPLEX(floatT)> & _redBase, int time, int col, int conjON){

        typedef GIndexer<All, HaloDepthSpin> GInd;

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol3;

        _redBase.adjustSize(elems_);

        if(conjON == 2){
            _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(
                SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin,12,2>(t, spinorInDagger,spinorIn,spinor_wave,time,col));
        }
        else if(conjON == 1){
            _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(
                SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin,12,1>(t, spinorInDagger,spinorIn,spinor_wave,time,col));
        }
        else{
            _redBase.template iterateOverSpatialBulk<All, HaloDepthSpin>(
                SumXYZ_TrMdaggerMwave<floatT, HaloDepthSpin,12,0>(t, spinorInDagger,spinorIn,spinor_wave,time,col));
        }
   
        _redBase.reduce(result, elems_);
        return result;
}

template<typename floatT, size_t HaloDepthSpin>
void loadWave(std::string fname, Spinorfield<floatT, true, All, HaloDepthSpin, 3,1> & spinor_device,
                                 Spinorfield<floatT, false, All, HaloDepthSpin, 3,1> & spinor_host,
                                 int time, int col,CommunicationBase & commBase){
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
 //   for ( int i = 0; i < 3; i ++){
 //       tmp3.data[i] = 0.0;
 //   }
    for (size_t z = 0; z < GInd::getLatData().lz; z++)
    for (size_t y = 0; y < GInd::getLatData().ly; y++)
    for (size_t x = 0; x < GInd::getLatData().lx; x++) {
        floatT *dataRe = (floatT *) &buf[ps];
        ps += sizeof(floatT);
        floatT *dataIm = (floatT *) &buf[ps];
        ps += sizeof(floatT);
        tmp3.data[col] = COMPLEX(floatT)(dataRe[0],dataIm[0]);
        //std::cout << "data " << data[0] << std::endl;
        spinor_host.getAccessor().setElement(GInd::getSite(x,y, z, time),tmp3);
    }

    commBase.closeIOBinary();

    spinor_device = spinor_host;
    spinor_device.updateAll();
}

template<typename floatT, size_t HaloDepthSpin>
void moveWave(Spinorfield<floatT, true, All, HaloDepthSpin, 3,1> & spinor_device,Spinorfield<floatT, false, All, HaloDepthSpin, 3,1> & spinor_host,
                                 int posX, int posY, int posZ,
                                 int timeOut, int colOut,int timeIn, int colIn ,CommunicationBase & commBase){
    typedef GIndexer<All, HaloDepthSpin> GInd;


    int coord[4];
    //all gather 4d
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
        for (int x=0; x<lx; x++){
            buf[x+lx*(y+ly*(z))] = std::complex<floatT>(((spinor_host.getAccessor().getElement(GInd::getSite(x,y, z, timeIn))).data[colIn]).cREAL,
                                   ((spinor_host.getAccessor().getElement(GInd::getSite(x,y, z, timeIn))).data[colIn]).cIMAG);
        }


    if(std::is_same<floatT,double>::value){
        MPI_Allgather(buf, lx*ly*lz, MPI_DOUBLE_COMPLEX, buf2, lx*ly*lz, MPI_DOUBLE_COMPLEX,commBase.getCart_comm() );
    }
    else if(std::is_same<floatT,float>::value){
        MPI_Allgather(buf, lx*ly*lz, MPI_COMPLEX, buf2, lx*ly*lz, MPI_COMPLEX,commBase.getCart_comm() );
    }
    

    for (int r=0; r<rankSize; r++){
    MPI_Cart_coords(commBase.getCart_comm(), r,4, coord);
   //     for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++){
            buf[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])))] = buf2[x+lx*(y+ly*(z+lz*(r)))];
        }
    }

        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++){
            Vect3<floatT> tmp3 = spinor_host.getAccessor().getElement(GInd::getSite(x,y, z, timeOut));
            tmp3.data[colOut] = COMPLEX(floatT)(real(buf[((x+lx*commBase.mycoords()[0]+glx-posX)%glx)
                                                   +glx*(((y+ly*commBase.mycoords()[1]+gly-posY)%gly)
                                                   +gly*(((z+lz*commBase.mycoords()[2]+glz-posZ)%glz)))]),
                                                imag(buf[((x+lx*commBase.mycoords()[0]+glx-posX)%glx)
                                                   +glx*(((y+ly*commBase.mycoords()[1]+gly-posY)%gly)
                                                   +gly*(((z+lz*commBase.mycoords()[2]+glz-posZ)%glz)))]));
            spinor_host.getAccessor().setElement(GInd::getSite(x,y, z, timeOut),tmp3);
        }

    spinor_device = spinor_host;
    delete[] buf;
    delete[] buf2;

    spinor_device.updateAll();

}

template<typename floatT, size_t HaloDepthSpin>
void loadWavePos(std::string fname, Spinorfield<floatT, true, All, HaloDepthSpin, 3,1> & spinor_device,
                                 Spinorfield<floatT, false, All, HaloDepthSpin, 3,1> & spinor_host,
                                 size_t posX, size_t posY, size_t posZ,
                                 int time, int col,CommunicationBase & commBase){
    typedef GIndexer<All, HaloDepthSpin> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();
    global[3] = 1;
//    local[3] = 1;

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
        tmp3.data[col] = COMPLEX(floatT)(dataRe[0], dataIm[0] );
        std::cout << "x "<< xg << " y "<< yg << " z "<< zg <<" dataRe " << dataRe[0] << " dataIm " << dataIm[0] << std::endl;
        spinor_host.getAccessor().setElement(GInd::getSite(x,y, z, time),tmp3);
    }

    commBase.closeIOBinary();

    spinor_device = spinor_host;

}

template<typename floatT, size_t HaloDepth>
void makeWaveSource(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3,1> &spinor_wave,
                      size_t time, size_t col,size_t post){

    typedef GIndexer<All, HaloDepth> GInd;
    size_t _elems = GInd::getLatData().vol4;
    ReadIndex<All,HaloDepth> index;

    iterateFunctorNoReturn<true,BLOCKSIZE>(MakeWaveSource12<floatT,HaloDepth>( spinorIn, spinor_wave,time,col,post),index,_elems);

    spinorIn.updateAll();

}

template<typename floatT>
void gatherAllHost(std::complex<floatT> *in,CommunicationBase & commBase){

    int coord[4];
    //all gather 4d
    typedef GIndexer<All,0> GInd;
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

    if(std::is_same<floatT,double>::value){
        MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX,commBase.getCart_comm() );
    }
    else if(std::is_same<floatT,float>::value){
        MPI_Allgather(in, lx*ly*lt*lz, MPI_COMPLEX, buf, lx*ly*lt*lz, MPI_COMPLEX,commBase.getCart_comm() );
    }


    for (int r=0; r<rankSize; r++){
    MPI_Cart_coords(commBase.getCart_comm(), r,4, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++){
            in[(x+lx*coord[0])+glx*((y+ly*coord[1])+gly*((z+lz*coord[2])+glz*((t+lt*coord[3]))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
        }
    }

    delete[] buf;


}


template<typename floatT,int direction>
void gatherHostXYZ(std::complex<floatT> *in,MPI_Comm & comm,int glx,int gly,int glz){

    int coord[1];
    //gather 4d for extended directions
    typedef GIndexer<All,0> GInd;
    int lx  = GInd::getLatData().lx;
    int ly  = GInd::getLatData().ly;
    int lz  = GInd::getLatData().lz;
    int glt = GInd::getLatData().globLT;
    int lt  = GInd::getLatData().lt;
    int myrank, rankSize;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &rankSize);

    std::complex<floatT> *buf = new std::complex<floatT>[glx*gly*glz*glt];
    
    if(std::is_same<floatT,double>::value){
        MPI_Allgather(in, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX, buf, lx*ly*lt*lz, MPI_DOUBLE_COMPLEX,comm );
    }
    else if(std::is_same<floatT,float>::value){
        MPI_Allgather(in, lx*ly*lt*lz, MPI_COMPLEX, buf, lx*ly*lt*lz, MPI_COMPLEX,comm );
    }

    
    for (int r=0; r<rankSize; r++){
    MPI_Cart_coords(comm, r,1, coord);
        for (int t=0; t<lt; t++)
        for (int z=0; z<lz; z++)
        for (int y=0; y<ly; y++)
        for (int x=0; x<lx; x++){
            if (direction == 0){
                in[(x+lx*coord[0])+glx*((y)+gly*((z)+glz*((t))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
            }
            else if (direction == 1){
                in[(x)+glx*((y+ly*coord[0])+gly*((z)+glz*((t))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
            }
            else if (direction == 2){
                in[(x)+glx*((y)+gly*((z+lz*coord[0])+glz*((t))))] = buf[x+lx*(y+ly*(z+lz*(t+lt*r)))];
            } 
        }
    }

    delete[] buf;


}


//////////template declarations

template void fourier3D(Spinorfield<double, true, All, 2, 12, 12> & spinor_out,Spinorfield<double, true, All, 2, 12, 12> & spinor_in,LatticeContainer<true,COMPLEX(double)> & redBase,LatticeContainer<false,COMPLEX(double)> & redBase2,CommunicationBase & commBase);

template COMPLEX(double) sumXYZ_TrMdaggerM(int t,
        const Spinorfield<double, true, All, 2, 12, 12> & spinorInDagger,
        const Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
        LatticeContainer<true,COMPLEX(double)> & _redBase);

template COMPLEX(double) sumXYZ_TrMdaggerMwave(int t,
        const Spinorfield<double, true, All, 2, 12, 12> & spinorInDagger,
        const Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
        const Spinorfield<double, true, All, 2, 3 ,  1> & spinor_wave,
        LatticeContainer<true,COMPLEX(double)> & _redBase, int time, int col, int conjON);

template void loadWave(std::string fname, Spinorfield<double, true , All, 2, 3,1> & spinor_device,
                                          Spinorfield<double, false, All, 2, 3,1> & spinor_host,
                                          int time, int col,CommunicationBase & commBase);

template void loadWavePos(std::string fname, Spinorfield<double, true , All, 2, 3,1> & spinor_device,
                                             Spinorfield<double, false, All, 2, 3,1> & spinor_host,
                                             size_t posX, size_t posY, size_t posZ,
                                             int time, int col,CommunicationBase & commBase);

template void makeWaveSource(Spinorfield<double, true, All, 2, 12, 12> & spinorIn, const Spinorfield<double, true, All, 2, 3,1> &spinor_wave,
                      size_t time, size_t col,size_t post);

template void moveWave(Spinorfield<double, true, All, 2, 3,1> & spinor_device,Spinorfield<double, false, All, 2, 3,1> & spinor_host,
                                 int posX, int posY, int posZ,
                                 int timeOut, int colOut,int timeIn, int colIn ,CommunicationBase & commBase);

