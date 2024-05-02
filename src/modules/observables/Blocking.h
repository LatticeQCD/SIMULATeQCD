//
// Created by Hai-Tao Shu on 2021.05.11
//

#pragma once
#include "../../base/LatticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "FieldStrengthTensor.h"
#include "Topology.h"
#include "../../base/math/matrix4x4.h"
#include "../../base/math/gcomplex.h"
#include "../../base/math/simpleArray.h"

//this class is to measure correlators using "blocking" method. only zero momentum is allowed for now..
template<class floatT, bool onDevice, size_t HaloDepth, typename dataType, class observableType, class corrFunc>
class BlockingMethod {
protected:
    LatticeContainer<onDevice, dataType> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;
    CommunicationBase &_commBase;

private:
    typedef GIndexer<All, HaloDepth> GInd;

public:
    BlockingMethod(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield): _redBase(gaugefield.getComm()), _gauge(gaugefield), _commBase(gaugefield.getComm()) {
        _redBase.adjustSize(GInd::getLatData().vol3);
    }

    ~BlockingMethod() {}

    dataType updateBlock(std::vector<dataType> &DataBlockOrdered, size_t binsize);
    std::vector<floatT> getCorr(std::vector<dataType> &DataBlockOrdered, size_t binsize);

};


template<class floatT>
class CorrType {
public:
    __host__ __device__ floatT inline correlate(floatT A, floatT B) {
        return A*B;
    }
    __host__ __device__ floatT inline correlate(Matrix4x4Sym<floatT> A, Matrix4x4Sym<floatT> B) {
        //"reduce" shear correlators. not good enough when using blocking method
        //return (0.25*( A.elems[0] - A.elems[1])*( B.elems[0] - B.elems[1])
        //      + 0.25*( A.elems[0] - A.elems[2])*( B.elems[0] - B.elems[2])
        //      + 0.25*( A.elems[1] - A.elems[2])*( B.elems[1] - B.elems[2]))/3.0;
        floatT X, Y;
        X = (0.25*( A.elems[0] - A.elems[1])*( B.elems[0] - B.elems[1])
           + 0.25*( A.elems[0] - A.elems[2])*( B.elems[0] - B.elems[2])
           + 0.25*( A.elems[1] - A.elems[2])*( B.elems[1] - B.elems[2]))/3.0;
        Y = A.elems[4]*B.elems[4] + A.elems[5]*B.elems[5] + A.elems[7]*B.elems[7];
        return (X*4. + Y*2.)/10.;
    }
};



template<size_t HaloDepth>
struct ReadIndexSpatialBlock {

    size_t _binsize;
    size_t _t;

    ReadIndexSpatialBlock(size_t binsize, size_t t) : _binsize(binsize), _t(t) {}
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {

        typedef GIndexer<All, HaloDepth> GInd;
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        size_t numBlocksInX = GInd::getLatData().lx/_binsize;
        size_t numBlocksInY = GInd::getLatData().ly/_binsize;

        size_t rem, x, y, z; //the coordinate of small lattice
        divmod(i,numBlocksInX*numBlocksInY,z,rem);
        divmod(rem,numBlocksInX,y,x);


        gSite site = GInd::getSite(x*_binsize, y*_binsize, z*_binsize, _t);
        return site;
    }
};

struct PassReadIndex {
    inline __host__ __device__ size_t operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        return blockDim.x * blockIdx.x + threadIdx.x;
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct EMTtrace {
    gaugeAccessor<floatT> _gAcc;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;
    typedef GIndexer<All, HaloDepth> GInd;

    EMTtrace(gaugeAccessor<floatT> gAcc) : _gAcc(gAcc), FT(gAcc) {}

    __device__ __host__ inline floatT operator()(gSite site) {

        GSU3<floatT> FS01, FS02, FS03, FS12, FS13, FS23;

        floatT FsigmaSquare = 0;
        FS01 = FT(site, 0, 1);
        FsigmaSquare += tr_d(FS01 * FS01);
        FS02 = FT(site, 0, 2);
        FsigmaSquare += tr_d(FS02 * FS02);
        FS12 = FT(site, 1, 2);
        FsigmaSquare += tr_d(FS12 * FS12);
        floatT FtauSquare = 0;
        FS03 = FT(site, 0, 3);
        FtauSquare += tr_d(FS03 * FS03);
        FS13 = FT(site, 1, 3);
        FtauSquare += tr_d(FS13 * FS13);
        FS23 = FT(site, 2, 3);
        FtauSquare += tr_d(FS23 * FS23);
        return FtauSquare+FsigmaSquare;

    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct EMTtraceless {
    gaugeAccessor<floatT> _gAcc;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;
    typedef GIndexer<All, HaloDepth> GInd;

    EMTtraceless(gaugeAccessor<floatT> gAcc) : _gAcc(gAcc), FT(gAcc) {}

    __device__ __host__ inline Matrix4x4Sym<floatT> operator()(gSite site) {

        GSU3<floatT> FS01, FS02, FS03, FS12, FS13, FS23;

        floatT FsigmaSquare = 0;
        FS01 = FT(site, 0, 1);
        FsigmaSquare += tr_d(FS01 * FS01);
        FS02 = FT(site, 0, 2);
        FsigmaSquare += tr_d(FS02 * FS02);
        FS12 = FT(site, 1, 2);
        FsigmaSquare += tr_d(FS12 * FS12);
        floatT FtauSquare = 0;
        FS03 = FT(site, 0, 3);
        FtauSquare += tr_d(FS03 * FS03);
        FS13 = FT(site, 1, 3);
        FtauSquare += tr_d(FS13 * FS13);
        FS23 = FT(site, 2, 3);
        FtauSquare += tr_d(FS23 * FS23);

        Matrix4x4Sym<floatT> emTensor;
        SimpleArray<GSU3<floatT>,16> FS(gsu3_zero<floatT>());

        FS[1] = FS01;
        FS[2] = FS02;
        FS[3] = FS03;
        FS[6] = FS12;
        FS[7] = FS13;
        FS[11] = FS23;

        floatT factor = -1;
        for (size_t mu=1;mu<4;mu++) {
            for(size_t nu=0;nu<mu;nu++) {
                FS[mu*4+nu] = factor*FS[nu*4+mu];
            }
        }


        for (size_t mu=0;mu<4;mu++) {
            for(size_t nu=mu;nu<4;nu++) {

                GSU3<floatT> FS1, FS2, FS3;
                FS1 = gsu3_zero<floatT>();
                FS2 = gsu3_zero<floatT>();
                FS3 = gsu3_zero<floatT>();
                floatT result = 0;
                for (size_t sigma = 0; sigma < 4; sigma++) {
                    FS1 += FS[mu*4+sigma] * FS[nu*4+sigma];
                }
                result = 2 * tr_d(FS1);

                if (mu == nu) {
                    result -= FtauSquare+FsigmaSquare;
                }
                emTensor(mu, nu, result);
            }
        }
        return emTensor;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, typename dataType, class observableType>
struct BlockingKernel {
    MemoryAccessor _Data;
    typedef GIndexer<All, HaloDepth> GInd;
    size_t _tau;
    observableType _Ob;

    BlockingKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, MemoryAccessor Data, size_t tau) : _Data(Data), _tau(tau), _Ob(gauge.getAccessor()) {}
    __device__ __host__ inline dataType operator()(gSite site) {

        sitexyzt coord = site.coord;
        site = GInd::getSite(coord[0],coord[1],coord[2],_tau);

        size_t nx = (size_t)GInd::getLatData().lx;
        size_t ny = (size_t)GInd::getLatData().ly;

        size_t Id = coord[0] + coord[1]*nx + coord[2]*ny*nx;
        dataType data = _Ob(site);
        _Data.setValue<dataType>(Id, data);
        return data;
    }
};


