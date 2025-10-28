#pragma once

#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "../base/math/matrix4x4SymComplex.h"

// hard coded
#define LZ 64

// functions definitions

// dslash wilson, no clover
template<typename floatT>
class FourierClass {

    // CommunicationBase &_commBase;
    MPI_Comm commX, commY, commZ, commT;
    size_t lx, ly, lz, lt; // local lattice sizes
    size_t lxL, lyL, lzL, ltL; // global lattice sizes
    size_t lsX, lsY, lsZ, lsT; // global lattice sizes as halved as integerly possible, used for the FFT algorithm
    int mycoords[4]; // communication information (current (?) coordinate?)
    int nodes[4]; // communication information (number of nodes in direction?)
    LatticeContainer<true, COMPLEX(floatT)> _redBaseDevice;
    LatticeContainer<false, COMPLEX(floatT)> _redBaseHost;

public:
    FourierClass(CommunicationBase &commBase) :
        _redBaseDevice(commBase, "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device", "Reduction_Base_Device"),
        _redBaseHost(commBase, "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host", "Reduction_Base_Host") {

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

        _redBaseDevice.adjustSize(lxL*lyL*lzL*ltL); // TODO: Not ltL?
        _redBaseHost.adjustSize(lxL*lyL*lzL*ltL);

    }

    template<size_t HaloDepth>
    void moveSpinor1212ToContainer(
        Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        int spincolor1, int spincolor2
    );

    // template<size_t HaloDepth> // didn't need this
    void moveEMTComponentToContainer(
        LatticeContainer<true, Matrix4x4SymComplex<floatT>> & emt,
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        int emt_component
    );

    void moveTensor4x4Symx4x4SymComplexComponentToContainer(
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> & tensor,
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        int firstIndexPair, int secondIndexPair
    );

    template<size_t HaloDepth, int dir>
    void moveContainerToSpinor1212Direction(
        Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        int spincolor1, int spincolor2
    );

    template<int dir>
    // template<size_t HaloDepth, int dir> // didn't need the HaloDepth
    void moveContainerToEMTDirection(
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        LatticeContainer<true, Matrix4x4SymComplex<floatT>> &emt,
        int emt_component
    );
    
    template<int dir>
    void moveContainerToTensor4x4Symx4x4SymComplexDirection(
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> &tensor,
        int firstIndexPair, int secondIndexPair
    );

    template<int dir>
    void performFourierTransformDirection(
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        LatticeContainer<false, COMPLEX(floatT)> & redBase2,
        int sign
    );

    template<size_t HaloDepth>
    void performFourier3DSpinor1212(
        Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
        Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,
        LatticeContainer<true, COMPLEX(floatT)> & redBase,
        LatticeContainer<false, COMPLEX(floatT)> & redBase2,
        int sign, int maxColorSpin
    );

    template<Summation summation>
    void performFourier3DEMT(
        LatticeContainer<true, Matrix4x4SymComplex<floatT>> & emt_in,
        LatticeContainer<true, Matrix4x4SymComplex<floatT>> & emt_out,
        int sign
    );

    template<Summation summation>
    void performFourier3DTensor4x4Symx4x4SymComplex(
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> & tensor_in,
        LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> & tensor_out,
        int sign
    );

};


template<class floatT, size_t HaloDepth>
void fourier3D(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,
    LatticeContainer<true, COMPLEX(floatT)> & redBaseDevice,
    LatticeContainer<false, COMPLEX(floatT)> & redBaseHost,
    CommunicationBase & commBase,
    int sign = 1,
    int maxColorSpin = 12
);

template<typename floatT, bool onDevice, size_t HaloDepthSpin>
void tr_spinorXspinor(
    Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn
);

template<typename floatT, bool onDevice, size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerM(
    int t,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
    LatticeContainer<true, COMPLEX(floatT)> & _redBase
);

template<typename floatT, bool onDevice, size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerMwave(
    int t,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
    const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 3, 1> & spinor_wave,
    LatticeContainer<true, COMPLEX(floatT)> & _redBase,
    int time, int col, int conjON
);

template<typename floatT, size_t HaloDepthSpin>
void loadWave(
    std::string fname,
    Spinorfield<floatT, true, All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    int time, int col,
    CommunicationBase & commBase
);

template<typename floatT, size_t HaloDepthSpin>
void loadWavePos(
    std::string fname,
    Spinorfield<floatT, true , All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    size_t posX, size_t posY, size_t posZ,
    int time, int col,
    CommunicationBase & commBase
);

template<typename floatT, size_t HaloDepth>
void makeWaveSource(
    Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
    const Spinorfield<floatT, true, All, HaloDepth, 3, 1> &spinor_wave,
    size_t time, size_t col, size_t post
);

template<typename floatT, size_t HaloDepthSpin>
void moveWave(
    Spinorfield<floatT, true, All, HaloDepthSpin, 3, 1> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 3, 1> & spinor_host,
    int posX, int posY, int posZ, int timeOut, int colOut, int timeIn, int colIn ,
    CommunicationBase & commBase
);

template<typename floatT, size_t HaloDepthSpin>
void gatherMomentum(
    COMPLEX(floatT) * CC,
    Spinorfield<floatT, true, All, HaloDepthSpin, 12, 12> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 12, 12> & spinor_host,
    int timeIn, int colIn, int savePos, int nMomentum,
    CommunicationBase & commBase
);

template<typename floatT, size_t HaloDepthSpin>
void gatherMomentumT(
    COMPLEX(floatT) * CC,
    Spinorfield<floatT, true, All, HaloDepthSpin, 12, 12> & spinor_device,
    Spinorfield<floatT, false, All, HaloDepthSpin, 12, 12> & spinor_host,
    int colIn, int savePos, int nP, int * pos,
    CommunicationBase & commBase
);

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

    // delete [] v;
    // delete [] v0;

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


template<class floatT, size_t HaloDepth>
__global__ void copySpinorToContainer(
    MemoryAccessor _redBase,
    Vect12ArrayAcc<floatT> _SpinorIn,
    const size_t size,
    int spincolor1, int spincolor2,
    int lx, int ly, int lz, int lt,
    const int xtopo, const int ytopo, const int ztopo
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All, HaloDepth> GInd;

    int ix, iy, iz, it;

    int tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp , lx      , iy, ix);

    if (
        ix >= xtopo*GInd::getLatData().lx && ix < (1+xtopo)*GInd::getLatData().lx &&
        iy >= ytopo*GInd::getLatData().ly && iy < (1+ytopo)*GInd::getLatData().ly &&
        iz >= ztopo*GInd::getLatData().lz && iz < (1+ztopo)*GInd::getLatData().lz
    ) {
        Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite(ix-xtopo*GInd::getLatData().lx, iy-ytopo*GInd::getLatData().ly, iz-ztopo*GInd::getLatData().lz, it), spincolor2));
        _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmp12.data[spincolor1]);
        // printf("%d %f %d %d %d \n", (int)(site), tmp12.data[spincolor1].cREAL, xtopo, ytopo, ztopo);
    } else {
        _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), 0.0);
    }

}


template<class floatT, size_t HaloDepth>
__global__ void copySpinorToContainerLocal(
    MemoryAccessor _redBase,
    Vect12ArrayAcc<floatT> _SpinorIn,
    const size_t size,
    int spincolor1, int spincolor2,
    int lx, int ly, int lz, int lt
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All, HaloDepth> GInd;

    int ix, iy, iz, it;
    // it = tt;

    int tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp , lx      , iy, ix);

    // 4x4 for me
    Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite(ix, iy, iz, it), spincolor2));
    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmp12.data[spincolor1]);

}

template<class floatT>
__global__ void copyEMTComponentToContainerLocal(
    LatticeContainerAccessor _redBase,
    LatticeContainerAccessor _emt,
    const size_t size,
    int emt_component,
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

    Matrix4x4SymComplex<floatT> tmpMat44 = _emt.getElement<Matrix4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));
    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmpMat44.elems[emt_component]);

}

template<class floatT>
__global__ void copyTensor4x4Symx4x4SymComplexComponentToContainerLocal(
    LatticeContainerAccessor _redBase,
    LatticeContainerAccessor _tensor,
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

    Tensor4x4Symx4x4SymComplex<floatT> tmpMat44 = _tensor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));
    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)), tmpMat44.elems[firstIndexPair][secondIndexPair]);

}

template<class floatT, size_t HaloDepth>
__global__ void copyContainerToSpinor(
    Vect12ArrayAcc<floatT> _SpinorOut,
    LatticeContainerAccessor _redBase,
    const size_t size,
    int spincolor1, int spincolor2,
    int lx, int ly, int lz, int lt,
    const int xtopo, const int ytopo, const int ztopo
) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All, HaloDepth> GInd;

    int ix, iy, iz, it;
    // it = tt;

    int tmp;

    divmod(site, GInd::getLatData().vol3, it, tmp);
    divmod(tmp , GInd::getLatData().vol2, iz, tmp);
    divmod(tmp , GInd::getLatData().vol1, iy, ix);

    COMPLEX(floatT) val = _redBase.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*it)));

    Vect12<floatT> tmp12 = _SpinorOut.getElement(GInd::getSiteStack(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2));

    tmp12.data[spincolor1] = val;
    _SpinorOut.setElement(GInd::getSiteStack(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2), tmp12);

}


template<class floatT>
__global__ void copyContainerToEMT(
    LatticeContainerAccessor _redBase,
    LatticeContainerAccessor _emt,
    const size_t size,
    int emt_component,
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
    COMPLEX(floatT) val = _redBase.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*(it+ttopo*GInd::getLatData().lt))));

    Matrix4x4SymComplex<floatT> tmpMat44 = _emt.getElement<Matrix4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));

    tmpMat44.elems[emt_component] = val;
    _emt.setElement(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)), tmpMat44);

}

template<class floatT>
__global__ void copyContainerToTensor4x4Symx4x4SymComplex(
    LatticeContainerAccessor _redBase,
    LatticeContainerAccessor _tensor,
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
    COMPLEX(floatT) val = _redBase.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*(it+ttopo*GInd::getLatData().lt))));

    Tensor4x4Symx4x4SymComplex<floatT> tmpTensor4444 = _tensor.getElement<Tensor4x4Symx4x4SymComplex<floatT>>(GInd::getSite(ix, iy, iz, it));

    tmpTensor4444.elems[firstIndexPair][secondIndexPair] = val;
    _tensor.setElement(GInd::getSite((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it)), tmpTensor4444);

}


template<class floatT, size_t HaloDepth, size_t NStacks>
struct SumXYZ_TrMdaggerM2 {
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;

    SpinorColorAcc<floatT> _spinorIn;
    SpinorColorAcc<floatT> _spinorInDagger;
    int _t;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrMdaggerM2(
        int t,
        const SpinorRHS_t &spinorInDagger,
        const SpinorRHS_t &spinorIn
    ) : _t(t), _spinorIn(spinorIn.getAccessor()), _spinorInDagger(spinorInDagger.getAccessor())
    { }

    // This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double) operator()(gSite site) {

        sitexyzt coords=site.coord;
        gSite siteT = GInd::getSite(coords.x, coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0, 0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT, stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(siteT, stack));
        }

        // printf("tr %d %d %d %f %f \n", coords.x, coords.y, coords.z, temp.cREAL, temp.cIMAG);

        return temp;
    }

};


template<class floatT, size_t HaloDepth, size_t NStacks, int conjON>
struct SumXYZ_TrMdaggerMwave{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;

    SpinorColorAcc<floatT> _spinorIn;
    SpinorColorAcc<floatT> _spinorInDagger;
    Vect3ArrayAcc<floatT> _spinor_wave;
    int _t, _time, _col;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrMdaggerMwave(
        int t,
        const SpinorRHS_t &spinorInDagger,
        const SpinorRHS_t &spinorIn,
        const Spinorfield<floatT, true, All, HaloDepth, 3, 1> &spinor_wave,
        int time, int col
    ) : _t(t), _spinorIn(spinorIn.getAccessor()), _spinorInDagger(spinorInDagger.getAccessor()), _spinor_wave(spinor_wave.getAccessor()), _time(time), _col(col) { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double) operator() (gSite site) {

        sitexyzt coords = site.coord;
        gSite siteT = GInd::getSite(coords.x, coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0, 0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT, stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(siteT, stack));
        }
        if (conjON == 2) {
            temp = COMPLEX(double)((temp *     ((_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col])).cREAL,
                                   (temp * conj((_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col])).cREAL);
        }
        else if (conjON == 1) {
            temp = temp*conj((_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col]);
        } else {
            temp = temp*((_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col]);  
        }

        temp = (_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col];

        // if (_time==5 && _col ==0 ) {
        //     COMPLEX(double) tmp2(0.0, 0.0);
        //     tmp2 = (_spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT, 0))).data[0];
        //     COMPLEX(double) tmp3(0.0, 0.0);
        //     tmp3 = (_spinorIn.template getElement<double>(GInd::getSiteStack(siteT, 0))).data[0];
        //     COMPLEX(double) tmp4(0.0, 0.0);
        //     tmp4 = (_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col];
        //     printf("psi %d %d %d %f %f %f %f %f %f \n", coords.x, coords.y, coords.z, tmp2.cREAL, tmp2.cIMAG, tmp3.cREAL, tmp3.cIMAG, tmp4.cREAL, tmp4.cIMAG);
        //     printf("psi %d %d %d %f %f \n", coords.x, coords.y, coords.z, tmp4.cREAL, tmp4.cIMAG);
        // }

        return temp;

    }

};


template<class floatT, size_t HaloDepth, size_t NStacks>
struct SpinorXdaggerwave{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;

    SpinorColorAcc<floatT> _spinorIn;
    Vect3ArrayAcc<floatT> _spinor_wave;
    int _time, _col;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SpinorXdaggerwave(const SpinorRHS_t &spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3, 1> &spinor_wave, int time, int col)
        : _spinorIn(spinorIn.getAccessor()), _spinor_wave(spinor_wave.getAccessor()), _time(time), _col(col) { }

    // This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator() (gSiteStack site) {

        sitexyzt coords = site.coord;

        return conj((_spinor_wave.template getElement<double>(GInd::getSite(coords.x, coords.y, coords.z, _time))).data[_col])*_spinorIn.template getElement<double>(site);

    }

};


template<class floatT, size_t HaloDepth, size_t NStacks>
struct Tr_SpinorXspinor{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;

    SpinorColorAcc<floatT> _spinorInDagger;
    SpinorColorAcc<floatT> _spinorIn;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    Tr_SpinorXspinor(SpinorRHS_t &spinorInDagger, const SpinorRHS_t &spinorIn) : _spinorInDagger(spinorInDagger.getAccessor()), _spinorIn(spinorIn.getAccessor()) { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator() (gSite site) {

        COMPLEX(double) temp(0.0, 0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(site, stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(site, stack));
        }
        Vect12<floatT> tmp(0.0);
        tmp.data[0] = temp;

        _spinorInDagger.setElement(site, tmp);

    }

};


template<class floatT, size_t HaloDepth>
struct MakeWaveSource12{

    // accessor to access the spinor field
    Vect12ArrayAcc<floatT> _spinorIn;
    Vect3ArrayAcc<floatT> _spinor_wave;

    size_t _time, _col, _post;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakeWaveSource12(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3, 1> &spinor_wave, size_t time, size_t col, size_t post) : _spinorIn(spinorIn.getAccessor()), _spinor_wave(spinor_wave.getAccessor()), _time(time), _col(col), _post(post) { }

    // This is the operator that is called inside the Kernel
    __device__ __host__ void operator() (gSite site) {

        for (size_t stack = 0; stack < 12; stack++) {
            Vect12<floatT> tmp(0.0);

            sitexyzt coords=site.coord;
            gSite siteT = GInd::getSite(coords.x, coords.y, coords.z, _time);
            if(coords[3] == _post ) {
                tmp.data[stack] = (_spinor_wave.template getElement<double>(siteT)).data[_col];
            }

            // if(_time == 0 && coords.y == 0 && coords.x == 0)
            // printf("i i %d %d %d %f %f \n", (int)coords.x, (int)coords.y, (int)coords.z, tmp.data[stack].cREAL , tmp.data[stack].cIMAG);
          
            const gSiteStack writeSite = GInd::getSiteStack(site, stack);
            _spinorIn.setElement(writeSite, tmp);

        }

    }

};
