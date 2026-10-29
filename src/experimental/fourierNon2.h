#pragma once

#include "../simulateqcd.h"
#include "../base/math/matrix4x4SymComplex.h"

// hard coded
#define LZ 64


template<typename floatT>
class FourierClass {

    // CommunicationBase &_commBase;
    MPI_Comm commX, commY, commZ, commT;
    size_t lx, ly, lz, lt; // local lattice sizes
    size_t lxL, lyL, lzL, ltL; // global lattice sizes
    size_t lsX, lsY, lsZ, lsT; // global lattice sizes as halved as integerly possible, used for the FFT algorithm
    int mycoords[4]; // communication information (current (?) coordinate?)
    int nodes[4]; // communication information (number of nodes in direction?)

public:
    FourierClass(CommunicationBase &commBase) {

        // _commBase = commBase;

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
        remain[0] = 0;
        remain[1] = 0;
        remain[2] = 0;
        remain[3] = 1;
        MPI_Cart_sub(commBase.getCart_comm(), remain, &commT);

        mycoords[0] = commBase.mycoords()[0];
        mycoords[1] = commBase.mycoords()[1];
        mycoords[2] = commBase.mycoords()[2];
        mycoords[3] = commBase.mycoords()[3];

        nodes[0] = commBase.nodes()[0];
        nodes[1] = commBase.nodes()[1];
        nodes[2] = commBase.nodes()[2];
        nodes[3] = commBase.nodes()[3];

        typedef GIndexer<All, 0> GInd;

        lx = GInd::getLatData().lx;
        ly = GInd::getLatData().ly;
        lz = GInd::getLatData().lz;
        lt = GInd::getLatData().lt;

        lxL = GInd::getLatData().globLX;
        lyL = GInd::getLatData().globLY;
        lzL = GInd::getLatData().globLZ;
        ltL = GInd::getLatData().globLT;

        lsX = lxL;
        lsY = lyL;
        lsZ = lzL;
        lsT = ltL;

        while(abs(round(((floatT)lsX)/2.0 )-(floatT)(lsX/2)) < 0.00001) {
            lsX = lsX/2;
        }
        while(abs(round(((floatT)lsY)/2.0 )-(floatT)(lsY/2)) < 0.00001) {
            lsY = lsY/2;
        }
        while(abs(round(((floatT)lsZ)/2.0 )-(floatT)(lsZ/2)) < 0.00001) {
            lsZ = lsZ/2;
        }
        while(abs(round(((floatT)lsT)/2.0 )-(floatT)(lsT/2)) < 0.00001) {
            lsT = lsT/2;
        }

    }

    void checkLatticeExtents() {
        if (lxL > LZ || lyL > LZ || lzL > LZ || ltL > LZ) {
            throw std::runtime_error(rootLogger.fatal(
                "Global lattice extents exceed the hard-coded maximum LZ", LZ,
                "used within the fourier(Polymorph) routine. Re-compile with a larger LZ."
            ));
        }
    }

    // template<size_t HaloDepth> // didn't need this
    void moveMatrix4x4SymComponentToContainer(
        LatticeContainer<true, Matrix4x4SymComplex<floatT>>& emt,
        LatticeContainer<true, COMPLEX(floatT)>& redBase,
        int emt_component
    );

    void moveTensor4x4Symx4x4SymComplexComponentToContainer(
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensor,
        LatticeContainer<true, COMPLEX(floatT)>& redBase,
        int firstIndexPair, int secondIndexPair
    );

    template<int dir>
    // template<size_t HaloDepth, int dir> // didn't need the HaloDepth
    void moveContainerToMatrix4x4SymDirection(
        LatticeContainer<true, COMPLEX(floatT)>& redBase,
        LatticeContainer<true, Matrix4x4SymComplex<floatT>>& emt,
        int emt_component
    );
    
    template<int dir>
    void moveContainerToTensor4x4Symx4x4SymComplexDirection(
        LatticeContainer<true, COMPLEX(floatT)>& redBase,
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensor,
        int firstIndexPair, int secondIndexPair
    );

    template<int dir>
    void performFourierTransformDirection(
        LatticeContainer<true, COMPLEX(floatT)>& redBaseDevice,
        LatticeContainer<false, COMPLEX(floatT)>& redBaseHost,
        int sign
    );

    template<int dir, typename elemType>
    void performFourierTransformDirectionPolymorph(
        LatticeContainer<true, elemType>& redBase,
        int sign
    );

    template<typename elemType, SpatialTemporal spatialTemporal>
    void performFourierTransformPolymorph(
        LatticeContainer<true, elemType>& latticeIn,
        LatticeContainer<true, elemType>& latticeOut,
        int sign
    );

    template<SpatialTemporal spatialTemporal>
    void performFourierTransformTensor4x4Symx4x4SymHalfPolymorph(
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorIn,
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorOut,
        int sign
    );

    template<SpatialTemporal spatialTemporal>
    void performFourierTransformMatrix4x4SymComponentwise(
        LatticeContainer<true, Matrix4x4SymComplex<floatT>>& matrixIn,
        LatticeContainer<true, Matrix4x4SymComplex<floatT>>& matrixOut,
        int sign
    );

    template<SpatialTemporal spatialTemporal>
    void performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise(
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorIn,
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& tensorOut,
        int sign
    );

};

template<typename floatT>
void gatherAllHost(std::complex<floatT> *in, CommunicationBase & commBase);

template<typename floatT, int direction>
void gatherHostXYZ(std::complex<floatT> *in, MPI_Comm & comm, int glx, int gly, int glz);

template<typename floatT, int direction>
void gatherHostXYZT(std::complex<floatT> *in, MPI_Comm & comm, int glx, int gly, int glz, int glt);

// gpu functions

template<class floatT, int direction>
__global__ void fourier(
    LatticeContainerAccessor _redBaseOut,
    LatticeContainerAccessor _redBaseIn,
    const size_t size, const size_t lx, const size_t ly, const size_t lz, const size_t lt,
    size_t lsIn,
    int sign = 1
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, it;
    int tmp;
    int ls=lsIn;
    int hf=lz/ls;

    divmod(site, lx*ly, it, tmp);
    divmod(tmp,  lx   , iy, ix );

    COMPLEX(floatT) v[LZ];
    COMPLEX(floatT) v0[LZ]; 

    // COMPLEX(floatT) * v = new COMPLEX(floatT)[lz];
    
    if(direction == 0) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(z+lz*(ix+lx*(iy+ly*it)));
        }
    }
    if(direction == 1) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(ix+lx*(z+lz*(iy+ly*it)));
        }
    }
    if(direction == 2) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(ix+lx*(iy+ly*(z+lz*it)));
        }
    }
    if(direction == 3) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(ix+lx*(iy+ly*(it+lt*z)));
        }
    }

    // standard fourier transformation
    for(int z = 0; z < lz ; z++) {
        v0[z] = v[z];
    }
    for(int i = 0; i < hf ; i++) {
        for(int k = 0; k < ls ; k++) {
            COMPLEX(floatT) sum = 0.0;
            for(int z = 0; z < ls ; z++) {
                sum = sum + v0[z*hf+i]*COMPLEX(floatT)(cos(sign*2.0*k*z*M_PI/ls), sin(sign*2.0*k*z*M_PI/ls));
            }
            v[i+k*hf] = sum;
        }
    }


    // fast part
    for(int j = 0; j < (int)(log2(lz/lsIn)+0.1) ; j++) {
        for(int z = 0; z < lz ; z++) {
            v0[z] = v[z];
        }
        ls=ls*2;
        hf=hf/2;
        for(int s = 0; s < hf ; s++) {
            for(int k = 0; k < ls/2 ; k++) {
                COMPLEX(floatT) phase = COMPLEX(floatT)(cos(sign*2.0*k*M_PI/ls), sin(sign*2.0*k*M_PI/ls));

                COMPLEX(floatT) even = v0[s + k*hf*2];
                COMPLEX(floatT) odd  = v0[s + k*hf*2 + hf];

                v[s + k*hf] = even + phase*odd;
                v[s + k*hf + hf*ls/2] = even - phase*odd;
            }
        }
    }

    for(int z = 0; z < lz ; z++) {
        v[z] = v[z]/sqrt(lz);
        // printf("%f %f \n", v[z].cREAL , v[z].cIMAG);
        if(direction == 0) {
            _redBaseOut.setValue<COMPLEX(floatT)>(z+lz*(ix+lx*(iy+ly*it)), v[z]);
        }
        if(direction == 1) {
            _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(z+lz*(iy+ly*it)), v[z]);
        }
        if(direction == 2) {
            // if(it == 0 && ix == 0 && iy == 0)
            // printf("i  %d %d %d %f %f \n", (int)ix, (int)iy, (int)z, v[z].cREAL , v[z].cIMAG);
            _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(z+lz*it)), v[z]);
        }
        if(direction == 3) {
            _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(it+lt*z)), v[z]);
        }
    }

}

template<class floatT, typename elemType, int direction>
__global__ void fourierPolymorph(
    LatticeContainerAccessor _redBaseOut,
    LatticeContainerAccessor _redBaseIn,
    const size_t size,
    const size_t lx, const size_t ly, const size_t lz, const size_t lt,
    size_t lsIn,
    int sign = 1
) {

    // every thread holds 2*LZ elements in local memory (v and v0 below), which the GPU reserves for all resident threads
    // this should be kept below 32kB, such that it stays below most GPU VRAM limits
    static_assert(
        2 * LZ * sizeof(elemType) <= 32 * 1024,
        "fourierPolymorph: element type too large for the per-thread buffers, "
        "use performFourierTransformTensor4x4Symx4x4SymHalfPolymorph instead"
    );

    typedef GIndexer<All> GInd;

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, it;
    int tmp;
    int ls=lsIn;
    int hf=lz/ls;

    divmod(site, lx*ly, it, tmp);
    divmod(tmp,  lx   , iy, ix );

    elemType v[LZ];
    elemType v0[LZ]; 

    // COMPLEX(floatT) * v = new COMPLEX(floatT)[lz];

    // fills the first lz values of the array v
    if(direction == 0) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<elemType>(GInd::getSite(sitexyzt(z, ix, iy, it)));
        }
    }
    if(direction == 1) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<elemType>(GInd::getSite(sitexyzt(ix, z, iy, it)));
        }
    }
    if(direction == 2) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<elemType>(GInd::getSite(sitexyzt(ix, iy, z, it)));
        }
    }
    if(direction == 3) {
        for(int z = 0; z < lz ; z++) {
            v[z] = _redBaseOut.getElement<elemType>(GInd::getSite(sitexyzt(ix, iy, it, z)));
        }
    }

    // standard fourier transformation
    for(int z = 0; z < lz ; z++) {
        v0[z] = v[z];
    }
    
    for(int i = 0; i < hf ; i++) {
        for(int k = 0; k < ls ; k++) {
            elemType sum = 0.0;
            for(int z = 0; z < ls ; z++) {
                sum = sum + v0[z*hf+i]*COMPLEX(floatT)(cos(sign*2.0*k*z*M_PI/ls), sin(sign*2.0*k*z*M_PI/ls));
            }
            v[i+k*hf] = sum;
        }
    }

    // fast part
    for(int j = 0; j < (int)(log2(lz/lsIn)+0.1) ; j++) {
        for(int z = 0; z < lz ; z++) {
            v0[z] = v[z];
        }
        ls=ls*2;
        hf=hf/2;
        for(int s = 0; s < hf ; s++) {
            for(int k = 0; k < ls/2 ; k++) {
                COMPLEX(floatT) phase = COMPLEX(floatT)(cos(sign*2.0*k*M_PI/ls), sin(sign*2.0*k*M_PI/ls));

                elemType even = v0[s + k*hf*2];
                elemType odd  = v0[s + k*hf*2 + hf];

                v[s + k*hf] = even + phase*odd;
                v[s + k*hf + hf*ls/2] = even - phase*odd;
            }
        }
    }
    
    for(int z = 0; z < lz ; z++) {
        v[z] = v[z]/sqrt(lz);
        if(direction == 0) {
            _redBaseOut.setElement(GInd::getSite(sitexyzt(z, ix, iy, it)), v[z]);
        }
        if(direction == 1) {
            _redBaseOut.setElement(GInd::getSite(sitexyzt(ix, z, iy, it)), v[z]);
        }
        if(direction == 2) {
            _redBaseOut.setElement(GInd::getSite(sitexyzt(ix, iy, z, it)), v[z]);
        }
        if(direction == 3) {
            _redBaseOut.setElement(GInd::getSite(sitexyzt(ix, iy, it, z)), v[z]);
        }
    }

}


template<class floatT>
__global__ void setValues(MemoryAccessor _redBaseOut, const size_t size, const size_t lx, const size_t ly, const size_t lz) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, iz;
    int  tmp;

    divmod(site, lx*ly, iz, tmp);
    divmod(tmp,  lx   , iy, ix );

    for(int z =0; z < lz ; z++) {
        _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*z), 0.1*(z+1)+0.1*(iy)+0.1*(ix));
    }

}


template<class floatT>
__global__ void moveValues(
    LatticeContainerAccessor _redBaseOut,
    LatticeContainerAccessor _redBaseIn,
    const size_t size,
    const size_t lx, const size_t ly, const size_t lz,
    const size_t lx2, const size_t ly2, const size_t lz2,
    const size_t xtopo, const size_t ytopo, const size_t ztopo
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, iz;
    int  tmp;

    divmod(site, lx2*ly2, iz, tmp);
    divmod(tmp,  lx2    , iy, ix );

    size_t pos , pos2;
    pos  = ix+lx2*(iy+ly2*iz);
    pos2 = (ix-lx*xtopo)+lx*((iy-ly*ytopo)+ly*(iz-lz*ztopo));

    // printf("%d %d %d %d %d %d %d \n", (int)site, (int)ix , (int)iy , (int)iz, (int)(lz*(ztopo+1)), (int)(lz*ztopo), (int)(( ix >= lx*xtopo && ix < lx*(xtopo+1) && iy >= ly*ytopo && iy < ly*(ytopo+1) && iz >= lz*ztopo && iz < lz*(ztopo+1))));

    if( ix >= lx*xtopo && ix < lx*(xtopo+1) && iy >= ly*ytopo && iy < ly*(ytopo+1) && iz >= lz*ztopo && iz < lz*(ztopo+1)) {
        printf("i  %d %d %d %d %d %f %f \n", (int)ix, (int)iy, (int)iz, (int)pos, (int)pos2, _redBaseIn.getElement<COMPLEX(floatT)>(pos2).cREAL , _redBaseIn.getElement<COMPLEX(floatT)>(pos2).cIMAG );
        _redBaseOut.setValue<COMPLEX(floatT)>(pos, _redBaseIn.getElement<COMPLEX(floatT)>(pos2));
    } else {
        printf("ii  %d %d %d %d \n", (int)ix, (int)iy, (int)iz, (int)pos);
        _redBaseOut.setValue<COMPLEX(floatT)>(pos, 0.0);
    }

}


template<class floatT>
__global__ void copyMatrix4x4SymComponentToContainerLocal(
    LatticeContainerAccessor redBaseAccessor,
    LatticeContainerAccessor matrixAccessor,
    const size_t size,
    int index,
    int lx, int ly, int lz, int lt
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All> GInd;

    int ix, iy, iz, it;
    // it = tt;

    int tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp , lx      , iy, ix);

    Matrix4x4SymComplex<floatT> tmpMat44 = matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));
    redBaseAccessor.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmpMat44(index));

}

template<class floatT>
__global__ void copyTensor4x4Symx4x4SymComplexComponentToContainerLocal(
    LatticeContainerAccessor redBaseAccessor,
    LatticeContainerAccessor tensorAccessor,
    const size_t size,
    int firstIndexPair, int secondIndexPair,
    int lx, int ly, int lz, int lt
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All> GInd;

    int ix, iy, iz, it;
    // it = tt;

    int tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp , lx      , iy, ix);

    Tensor4x4Symx4x4SymComplex<floatT> tmpMat44 = tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));
    redBaseAccessor.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmpMat44(firstIndexPair, secondIndexPair));

}


template<class floatT>
__global__ void copyContainerToMatrix4x4Local(
    LatticeContainerAccessor redBaseAccessor,
    LatticeContainerAccessor matrixAccessor,
    const size_t size,
    int index,
    int lx, int ly, int lz, int lt,
    const int xtopo, const int ytopo, const int ztopo, const int ttopo
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All> GInd;

    int ix, iy, iz, it;

    int tmp;

    divmod(site, GInd::getLatData().vol3, it, tmp);
    divmod(tmp , GInd::getLatData().vol2, iz, tmp);
    divmod(tmp , GInd::getLatData().vol1, iy, ix);

    // expanded by ttopo
    COMPLEX(floatT) val = redBaseAccessor.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*(it+ttopo*GInd::getLatData().lt))));

    Matrix4x4SymComplex<floatT> tmpMat44 = matrixAccessor.getElement<Matrix4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));

    tmpMat44(index, val);
    matrixAccessor.setElement(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)), tmpMat44);

}

template<class floatT>
__global__ void copyContainerToTensor4x4Symx4x4SymComplexLocal(
    LatticeContainerAccessor redBaseAccessor,
    LatticeContainerAccessor tensorAccessor,
    const size_t size,
    int firstIndexPair, int secondIndexPair,
    int lx, int ly, int lz, int lt,
    const int xtopo, const int ytopo, const int ztopo, const int ttopo
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All> GInd;

    int ix, iy, iz, it;

    int tmp;

    divmod(site, GInd::getLatData().vol3, it, tmp);
    divmod(tmp , GInd::getLatData().vol2, iz, tmp);
    divmod(tmp , GInd::getLatData().vol1, iy, ix);

    // expanded by ttopo
    COMPLEX(floatT) val = redBaseAccessor.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*(it+ttopo*GInd::getLatData().lt))));

    Tensor4x4Symx4x4SymComplex<floatT> tmpTensor4444 = tensorAccessor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));

    tmpTensor4444(firstIndexPair, secondIndexPair, val);
    tensorAccessor.setElement(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)), tmpTensor4444);

}
