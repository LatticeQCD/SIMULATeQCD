//
// Created by Lukas Mazur on 28.11.18.
//

#pragma once
#include "../../base/latticeContainer.h"
#include "../../base/math/simpleArray.h"
#include "../../gauge/gaugefield.h"
#include "fieldStrengthTensor.h"
#include "polyakovLoop.h"
#include "blocking.h"
#include "../../base/math/matrix4x4.h"
#include "../../base/math/complex.h"

/// this is the standard implementation for the EMT that will be used in the multi-level algorithm. It could take finite momentum(only z direction). and for the calculation of traceful part.
// a "displacement" must be provided to eliminate the round-off error. this "shift" must be very close to its true value and could be set to the "trace anomoly" obtained from a test run.
template<class floatT, bool onDevice, size_t HaloDepth>
class EnergyMomentumTensor {
protected:
    LatticeContainer<onDevice, Matrix4x4Sym<floatT> > _redBaseU;
    LatticeContainer<onDevice, Matrix4x4Sym<floatT> > _redBaseEMTUTimeSlices;
    LatticeContainer<onDevice, floatT > _redBaseEMTETimeSlices;
    LatticeContainer<onDevice, floatT> _redBaseE;
    LatticeContainer<onDevice, COMPLEX(floatT)> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;

private:
    bool recompute;
    typedef GIndexer<All, HaloDepth> GInd;

public:
    EnergyMomentumTensor(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield)
            : _redBaseU(gaugefield.getComm()), _redBaseEMTUTimeSlices(gaugefield.getComm()), _redBaseEMTETimeSlices(gaugefield.getComm()), _redBaseE(gaugefield.getComm()), _redBase(gaugefield.getComm()),
              _gauge(gaugefield),
              recompute(true) {
        _redBaseU.adjustSize(GInd::getLatData().vol3);
        _redBaseEMTUTimeSlices.adjustSize(GInd::getLatData().vol4);
        _redBaseEMTETimeSlices.adjustSize(GInd::getLatData().vol4);
        _redBaseE.adjustSize(GInd::getLatData().vol3);
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    ~EnergyMomentumTensor() {}

    void emTimeSlices(std::vector<floatT> &resultE, std::vector<Matrix4x4Sym<floatT>> &resultU,
                      MemoryAccessor &sub_E_gpu, MemoryAccessor &sub_U_gpu, int tau, int pz, int real_imag);
    COMPLEX(floatT) getFtau2PlusMinusFsigma2();

    //same as emTimeSlices but doesn't save data in memory for later use
    void EMTUTimeSlices(std::vector<Matrix4x4Sym<floatT>> &result);
    void EMTETimeSlices(std::vector<floatT> &result);
};


template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct emTensorElementsE {
    SU3Accessor<floatT> gAcc;
    MemoryAccessor sub_E_gpu;
    FieldStrengthTensor<floatT, HaloDepth, onDevice,comp> FT;
    typedef GIndexer<All, HaloDepth> GInd;

    emTensorElementsE(SU3Accessor<floatT> gAcc, MemoryAccessor sub_E_gpu) : gAcc(gAcc), sub_E_gpu(sub_E_gpu), FT(gAcc) {}

    __device__ __host__ inline void operator()(gSite site) {

        int nx = (int)GInd::getLatData().lx;
        int ny = (int)GInd::getLatData().ly;
        sitexyzt coord = site.coord;
        size_t Id = coord[0] + coord[1]*nx + coord[2]*ny*nx;

        SU3<floatT> FS01, FS02, FS03, FS12, FS13, FS23;
        floatT FtauSquare = 0;
        floatT FsigmaSquare = 0;
        FS01 = FT(site, 0, 1);
        FsigmaSquare += tr_d(FS01 * FS01);
        FS02 = FT(site, 0, 2);
        FsigmaSquare += tr_d(FS02 * FS02);
        FS12 = FT(site, 1, 2);
        FsigmaSquare += tr_d(FS12 * FS12);
        FS03 = FT(site, 0, 3);
        FtauSquare += tr_d(FS03 * FS03);
        FS13 = FT(site, 1, 3);
        FtauSquare += tr_d(FS13 * FS13);
        FS23 = FT(site, 2, 3);
        FtauSquare += tr_d(FS23 * FS23);

        sub_E_gpu.setValue<floatT>(Id, FtauSquare+FsigmaSquare);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct Ftau2PlusMinusFsigma2Elements {
    SU3Accessor<floatT> gAcc;
    FieldStrengthTensor<floatT, HaloDepth, onDevice,comp> FT;

    Ftau2PlusMinusFsigma2Elements(Gaugefield<floatT, onDevice, HaloDepth> &gauge) : gAcc(gauge.getAccessor()), FT(gauge.getAccessor()) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        SU3<floatT> FS01, FS02, FS03, FS12, FS13, FS23;
        floatT FtauSquare = 0;
        floatT FsigmaSquare = 0;
        FS01 = FT(site, 0, 1);
        FsigmaSquare += tr_d(FS01 * FS01);
        FS02 = FT(site, 0, 2);
        FsigmaSquare += tr_d(FS02 * FS02);
        FS12 = FT(site, 1, 2);
        FsigmaSquare += tr_d(FS12 * FS12);
        FS03 = FT(site, 0, 3);
        FtauSquare += tr_d(FS03 * FS03);
        FS13 = FT(site, 1, 3);
        FtauSquare += tr_d(FS13 * FS13);
        FS23 = FT(site, 2, 3);
        FtauSquare += tr_d(FS23 * FS23);

        return COMPLEX(floatT)(FtauSquare+FsigmaSquare, FtauSquare-FsigmaSquare);
    }
};


template<class floatT, size_t HaloDepth, bool onDevice>
struct energyMomentumTensorEKernel {
    SU3Accessor<floatT> gAcc;
    MemoryAccessor sub_E_gpu;
    emTensorElementsE<floatT, HaloDepth, onDevice, R18> EMTE;
    int tau;
    int pz;
    int real_imag;
    floatT displacement;
    energyMomentumTensorEKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, MemoryAccessor sub_E_gpu, int tau, int pz,
                                int real_imag, floatT displacement = 0) : gAcc(gauge.getAccessor()), sub_E_gpu(sub_E_gpu), EMTE(gAcc, sub_E_gpu),
                                tau(tau), pz(pz), real_imag(real_imag), displacement(displacement) {}

    __device__ __host__ inline floatT operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        sitexyzt coord = site.coord;
        int nx = (int)GInd::getLatData().lx;
        int ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().globLZ;
        size_t Id = coord[0] + coord[1]*nx + coord[2]*ny*nx;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],tau);
        sitexyzt newLocalCoord = newSite.coord;
        sitexyzt newGlobalCoord = GInd::getLatData().globalPos(newLocalCoord);

        floatT Phase;
        if ( real_imag == 0 )
            Phase = cos(2.*M_PI*pz*newGlobalCoord[2]/Nz);
        else
            Phase = sin(2.*M_PI*pz*newGlobalCoord[2]/Nz);

        if ( pz == 0 && real_imag == 0) {
            EMTE(newSite);
        }
        floatT temp, result;
        sub_E_gpu.getValue<floatT>(Id, temp);
        if ( pz==0 && real_imag == 0) {
            result = temp*Phase-displacement;
        } else {
            result = temp*Phase;
        }
        return result;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct emTensorElementsU {
    SU3Accessor<floatT> gAcc;
    MemoryAccessor sub_E_gpu;
    MemoryAccessor sub_U_gpu;
    FieldStrengthTensor<floatT, HaloDepth, onDevice,comp> FT;
    typedef GIndexer<All, HaloDepth> GInd;

    emTensorElementsU(SU3Accessor<floatT> gAcc, MemoryAccessor sub_E_gpu, MemoryAccessor sub_U_gpu) : gAcc(gAcc), sub_E_gpu(sub_E_gpu), sub_U_gpu(sub_U_gpu), FT(gAcc) {}

    __device__ __host__ inline void operator()(gSite site) {

        Matrix4x4Sym<floatT> emTensor;

        int nx = (int)GInd::getLatData().lx;
        int ny = (int)GInd::getLatData().ly;
        sitexyzt coord = site.coord;
        size_t Id = coord[0] + coord[1]*nx + coord[2]*ny*nx;


        SimpleArray<SU3<floatT>,16> FS(su3_zero<floatT>());
        for (int mu=0;mu<3;mu++) {
            for(int nu=mu+1;nu<4;nu++) {
                FS[mu*4+nu] = FT(site, mu, nu);
            }
        }

        floatT factor = -1.;
        for (int mu=1;mu<4;mu++) {
            for(int nu=0;nu<mu;nu++) {
                FS[mu*4+nu] = factor*FS[nu*4+mu];
            }
        }

        for (int mu=0;mu<4;mu++) {
            for(int nu=mu;nu<4;nu++) {

                SU3<floatT> FS1, FS2, FS3;
                FS1 = su3_zero<floatT>();
                FS2 = su3_zero<floatT>();
                FS3 = su3_zero<floatT>();
                floatT temp, result = 0;
                for (int sigma = 0; sigma < 4; sigma++) {
                    FS1 += FS[mu*4+sigma] * FS[nu*4+sigma];
                }
                result = 2 * tr_d(FS1);

                if (mu == nu) {
                    sub_E_gpu.getValue<floatT>(Id, temp);
                    result -= temp;
                }
                emTensor(mu, nu, result);
            }
        }
        sub_U_gpu.setValue<Matrix4x4Sym<floatT>>(Id, emTensor);
    }
};

template<class floatT, size_t HaloDepth, bool onDevice>
struct energyMomentumTensorUKernel {
    SU3Accessor<floatT> gAcc;
    MemoryAccessor sub_E_gpu;
    MemoryAccessor sub_U_gpu;
    emTensorElementsU<floatT, HaloDepth, onDevice, R18> EMTU;
    int tau;
    int pz;
    int real_imag;

    energyMomentumTensorUKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, MemoryAccessor sub_E_gpu, MemoryAccessor sub_U_gpu, int tau, int pz, int real_imag) :
    gAcc(gauge.getAccessor()), sub_E_gpu(sub_E_gpu), sub_U_gpu(sub_U_gpu), EMTU(gAcc,sub_E_gpu,sub_U_gpu), tau(tau), pz(pz), real_imag(real_imag) {}

    typedef GIndexer<All, HaloDepth> GInd;

    __device__ __host__ inline Matrix4x4Sym<floatT> operator()(gSite site) {
        Matrix4x4Sym<floatT> result;

        typedef GIndexer<All,HaloDepth> GInd;
        sitexyzt coord = site.coord;
        int nx = (int)GInd::getLatData().lx;
        int ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().globLZ;
        size_t Id = coord[0] + coord[1]*nx + coord[2]*ny*nx;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],tau);
        sitexyzt newLocalCoord = newSite.coord;
        sitexyzt newGlobalCoord = GInd::getLatData().globalPos(newLocalCoord);

        floatT Phase;
        if ( real_imag == 0 )
            Phase = cos(2.*M_PI*pz*newGlobalCoord[2]/Nz);
        else
            Phase = sin(2.*M_PI*pz*newGlobalCoord[2]/Nz);

        if ( pz == 0 && real_imag == 0) {
            EMTU(newSite);
        }

        sub_U_gpu.getValue<Matrix4x4Sym<floatT>>(Id, result);
        result *= Phase;
        return result;
    }

};


template<class floatT, bool onDevice, size_t HaloDepth>
void EnergyMomentumTensor<floatT, onDevice, HaloDepth>::emTimeSlices(std::vector<floatT> &resultE, std::vector<Matrix4x4Sym<floatT>> &resultU,
                      MemoryAccessor &sub_E_gpu, MemoryAccessor &sub_U_gpu, int tau, int pz, int real_imag) {

    typedef GIndexer<All, HaloDepth> GInd;
    const int Nt = (int)GInd::getLatData().globLT;
    const size_t elems = GInd::getLatData().vol3;
    const size_t spatialvol = GInd::getLatData().globvol3;
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }

    _redBaseE.template iterateOverSpatialBulk<All, HaloDepth>(energyMomentumTensorEKernel<floatT, HaloDepth, onDevice>(_gauge, sub_E_gpu, tau, pz, real_imag));
    floatT resultE_tmp(0);
    _redBaseE.reduce(resultE_tmp, elems);
    resultE_tmp /= spatialvol;
    resultE[pz*Nt+tau] = resultE_tmp;

    _redBaseU.template iterateOverSpatialBulk<All, HaloDepth>(energyMomentumTensorUKernel<floatT, HaloDepth, onDevice>(_gauge, sub_E_gpu, sub_U_gpu, tau, pz, real_imag));
    Matrix4x4Sym<floatT> resultU_tmp(0);
    _redBaseU.reduce(resultU_tmp, elems);
    resultU_tmp /= spatialvol;
    resultU[pz*Nt+tau] = resultU_tmp;
}

template<class floatT, bool onDevice, size_t HaloDepth>
COMPLEX(floatT) EnergyMomentumTensor<floatT, onDevice, HaloDepth>::getFtau2PlusMinusFsigma2() {

    _redBase.template iterateOverBulk<All, HaloDepth>(
             Ftau2PlusMinusFsigma2Elements<floatT, onDevice, HaloDepth, R18>(_gauge));
    COMPLEX(floatT) result = 0;
    _redBase.reduce(result, GInd::getLatData().vol4);
    return result/GInd::getLatData().globvol4;
}

template<class floatT, bool onDevice, size_t HaloDepth>
void EnergyMomentumTensor<floatT, onDevice, HaloDepth>::EMTUTimeSlices(std::vector<Matrix4x4Sym<floatT>> &result) {

    _redBaseEMTUTimeSlices.template iterateOverTimeslices<All, HaloDepth>(EMTtraceless<floatT, onDevice, HaloDepth>(_gauge.getAccessor()));
    _redBaseEMTUTimeSlices.reduceTimeSlices(result);
}

template<class floatT, bool onDevice, size_t HaloDepth>
void EnergyMomentumTensor<floatT, onDevice, HaloDepth>::EMTETimeSlices(std::vector<floatT> &result) {

    _redBaseEMTETimeSlices.template iterateOverTimeslices<All, HaloDepth>(EMTtrace<floatT, onDevice, HaloDepth>(_gauge.getAccessor()));
    _redBaseEMTETimeSlices.reduceTimeSlices(result);
}
