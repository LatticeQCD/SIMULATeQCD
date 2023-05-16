//
// Created by Hai-Tao Shu on 2021.05.11
//

#include "Blocking.h"


template<class floatT, bool onDevice, size_t HaloDepth, typename dataType>
struct sumBlock {
    MemoryAccessor _Data;
    MemoryAccessor _DataBlock;
    size_t _binsize;
    sumBlock(MemoryAccessor Data, MemoryAccessor DataBlock, size_t binsize) :
               _Data(Data), _DataBlock(DataBlock), _binsize(binsize) {}

    __device__ __host__ inline void operator()(gSite site) {

        typedef GIndexer<All, HaloDepth> GInd;


        size_t Nx = (size_t)GInd::getLatData().globLX;
        size_t nx = (size_t)GInd::getLatData().lx;
        size_t ny = (size_t)GInd::getLatData().ly;
        size_t nz = (size_t)GInd::getLatData().lz;
        sitexyzt coord = site.coord;
        size_t global_nBlockxyz = Nx/_binsize;

        dataType sumData = 0;
        for (size_t i=0;i<_binsize;i++) {
            for (size_t j=0;j<_binsize;j++) {
                for (size_t k=0;k<_binsize;k++) {
                    size_t Id1 = (coord[0]+i) + (coord[1]+j)*nx + (coord[2]+k)*ny*nx;
                    dataType Data_tmp;
                    _Data.getValue<dataType>(Id1, Data_tmp);
                    sumData += Data_tmp;
                }
            }
        }
        sumData /= _binsize*_binsize*_binsize;

        sitexyzt globalCoord = GInd::getLatData().globalPos(coord);

        size_t Id2 = size_t(globalCoord[0]/_binsize) + size_t(globalCoord[1]/_binsize)*global_nBlockxyz+size_t(globalCoord[2]/_binsize)*global_nBlockxyz*global_nBlockxyz;
        _DataBlock.setValue<dataType>(Id2, sumData);
    }
};



template<class floatT, bool onDevice, size_t HaloDepth, typename dataType, class observableType, class corrFunc>
dataType BlockingMethod<floatT, onDevice, HaloDepth, dataType, observableType, corrFunc>::updateBlock(std::vector<dataType> &DataBlockOrdered, size_t binsize) {

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    const size_t Nx = (size_t)GInd::getLatData().globLX;
    const size_t local_vol3 = GInd::getLatData().vol3;
    const size_t global_vol3 = GInd::getLatData().globvol3;

    const size_t global_nbins = global_vol3/binsize/binsize/binsize; //global nbins in all directions
    const size_t local_nbins = local_vol3/binsize/binsize/binsize;

    size_t global_binxyz = Nx/binsize;//global nbins in each direction

    typedef gMemoryPtr<true> MemTypeGPU;

    MemTypeGPU mem57 = MemoryManagement::getMemAt<true>("DataBlock");
    mem57->template adjustSize<dataType>(global_nbins);

    typedef gMemoryPtr<false> MemTypeCPU;

    MemTypeCPU mem59 = MemoryManagement::getMemAt<false>("DataBlockCPU");
    mem59->template adjustSize<dataType>(global_nbins);


    std::fill(DataBlockOrdered.begin(), DataBlockOrdered.end(), 0);

    MemTypeGPU mem55 = MemoryManagement::getMemAt<true>("Data");
    mem55->template adjustSize<dataType>(local_vol3);


    dataType DataMean = 0.;

    for (size_t t=0; t<Nt; t++) {

        mem55->memset(0);
        MemoryAccessor Data (mem55->getPointer());

        mem57->memset(0);
        MemoryAccessor DataBlock (mem57->getPointer());

        _redBase.template iterateOverSpatialBulk<All, HaloDepth>(BlockingKernel<floatT,onDevice,HaloDepth,dataType,observableType>(_gauge, Data, t));
        dataType data = 0.;
        _redBase.reduce(data, local_vol3);
        DataMean += data;

        ReadIndexSpatialBlock<HaloDepth> calcReadIndexSpatialBlock(binsize, t);
        iterateFunctorNoReturn<onDevice>(sumBlock<floatT,onDevice,HaloDepth,dataType>(Data, DataBlock, binsize), calcReadIndexSpatialBlock, local_nbins);

        mem59->template copyFrom<true>(mem57, mem57->getSize());
        MemoryAccessor DataBlockCPU (mem59->getPointer());


        std::vector<dataType> vec_DataBlock_temp(global_nbins, 0);


        for (size_t k=0;k<global_binxyz;k++)
        {
            for (size_t j=0;j<global_binxyz;j++)
            {
                for (size_t i=0;i<global_binxyz;i++)
                {
                    size_t index = i+j*global_binxyz+k*global_binxyz*global_binxyz;
                    DataBlockCPU.getValue<dataType>(index, vec_DataBlock_temp[index]);
                }
            }
        }

        _commBase.reduce(&vec_DataBlock_temp[0], vec_DataBlock_temp.size());
        for (size_t k=0;k<global_binxyz;k++)
        {
            for (size_t j=0;j<global_binxyz;j++)
            {
                for (size_t i=0;i<global_binxyz;i++)
                {
                    size_t index = i+j*global_binxyz+k*global_binxyz*global_binxyz;
                    DataBlockOrdered[index+t*global_nbins] = vec_DataBlock_temp[index];
                }
            }
        }
    }
    DataMean /= 1.*global_vol3*Nt;
    return DataMean;
}



template<class floatT, typename dataType, class corrFunc>
struct contractionKernel {

    MemoryAccessor _DataBlockOrderedGPU;
    MemoryAccessor _CorrGPU;
    size_t _dt;
    size_t _global_binxyz;
    size_t _Nt;
    size_t _Ns;

    contractionKernel(MemoryAccessor DataBlockOrderedGPU, MemoryAccessor CorrGPU, size_t dt, size_t global_binxyz, size_t Nt, size_t Ns) :
            _DataBlockOrderedGPU(DataBlockOrderedGPU), _CorrGPU(CorrGPU), _dt(dt), _global_binxyz(global_binxyz), _Nt(Nt), _Ns(Ns) {}

    __device__ __host__ void operator()(size_t dindex) {

        size_t vol1 = _global_binxyz;
        size_t vol2 = _global_binxyz*vol1;
        size_t vol3 = _global_binxyz*vol2;
        size_t rem;
        size_t dx, dy, dz;

        corrFunc c;
        divmod(dindex,vol2,dz,rem);
        divmod(rem, vol1, dy, dx);

        floatT Corr = 0;
        dataType A, B;
        for (size_t t1=0; t1<_Nt; t1++) {
            for (size_t z1=0; z1<_global_binxyz; z1++) {
                for (size_t y1=0; y1<_global_binxyz; y1++) {
                    for (size_t x1=0; x1<_global_binxyz; x1++) {
                        size_t id1 = x1 + y1*vol1 + z1*vol2 + t1*vol3;
                        _DataBlockOrderedGPU.getValue<dataType>(id1, A);
                        size_t t2 = (t1+_dt)%_Nt;
                        size_t z2 = (z1+dz)%_global_binxyz;
                        size_t y2 = (y1+dy)%_global_binxyz;
                        size_t x2 = (x1+dx)%_global_binxyz;
                        size_t id2 = x2 + y2*vol1 + z2*vol2 + t2*vol3;
                        _DataBlockOrderedGPU.getValue<dataType>(id2, B);
                        Corr += c.correlate(A, B);
                    }
                }
            }
        }
        size_t binsize = _Ns/_global_binxyz;
        Corr *= 1./_Nt*binsize*binsize*binsize/_global_binxyz/_global_binxyz/_global_binxyz;

        _CorrGPU.setValue<floatT>(dindex, Corr);
    }
};



template<class floatT, bool onDevice, size_t HaloDepth, typename dataType, class observableType, class corrFunc>
std::vector<floatT> BlockingMethod<floatT, onDevice, HaloDepth, dataType, observableType, corrFunc>::getCorr(std::vector<dataType> &DataBlockOrdered, size_t binsize) {

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    const size_t Nx = (size_t)GInd::getLatData().globLX;

    size_t vol1 = Nx/binsize;
    size_t vol2 = vol1*vol1;
    size_t vol3 = vol1*vol2;

    typedef gMemoryPtr<true> MemTypeGPU;
    typedef gMemoryPtr<false> MemTypeCPU;


    MemTypeCPU mem61 = MemoryManagement::getMemAt<false>("DataBlockOrderedCPU");
    mem61->template adjustSize<dataType>(vol3*Nt);
    MemoryAccessor DataBlockOrderedCPU (mem61->getPointer());

    for (size_t t=0;t<Nt;t++)
    {
        for (size_t k=0;k<vol1;k++)
        {
            for (size_t j=0;j<vol1;j++)
            {
                for (size_t i=0;i<vol1;i++)
                {
                    size_t index = i+j*vol1+k*vol1*vol1+t*vol3;
                    DataBlockOrderedCPU.setValue(index, DataBlockOrdered[index]);
                }
            }
        }
    }


    MemTypeGPU mem63 = MemoryManagement::getMemAt<true>("DataBlockOrderedGPU");
    mem63->template adjustSize<dataType>(vol3*Nt);
    mem63->template copyFrom<false>(mem61, mem61->getSize());
    MemoryAccessor DataBlockOrderedGPU (mem63->getPointer());

    PassReadIndex passReadIndex;

    MemTypeGPU mem65 = MemoryManagement::getMemAt<true>("CorrGPU");
    mem65->template adjustSize<floatT>(vol3);

    MemTypeCPU mem66 = MemoryManagement::getMemAt<false>("CorrCPU");
    mem66->template adjustSize<floatT>(vol3);

    size_t RsqSize = 3*(vol1/2+1)*(vol1/2+1);
    std::vector<floatT> Corr(RsqSize*(Nt/2+1), 0);

    for (size_t dt=0;dt<Nt/2+1;dt++)
    {
        mem65->memset(0);
        MemoryAccessor CorrGPU (mem65->getPointer());

        iterateFunctorNoReturn<onDevice>(contractionKernel<floatT,dataType,corrFunc>(DataBlockOrderedGPU, CorrGPU, dt, vol1, Nt, Nx), passReadIndex, vol3);

        mem66->template copyFrom<true>(mem65, mem65->getSize());
        MemoryAccessor CorrCPU (mem66->getPointer());

        for (size_t index=0;index<vol3;index++)
        {
            size_t dx, dy, dz, rem;
            divmod(index,vol2,dz,rem);
            divmod(rem,vol1,dy,dx);
            dx = dx > vol1/2 ? vol1-dx : dx;
            dy = dy > vol1/2 ? vol1-dy : dy;
            dz = dz > vol1/2 ? vol1-dz : dz;
            size_t rsq = dx*dx + dy*dy + dz*dz;

            floatT temp;
            CorrCPU.getValue<floatT>(index, temp);

            Corr[rsq+dt*RsqSize] += temp;
        }
    }
    return Corr;
}


#define CLASS_INIT3(floatT, HALO) \
template class BlockingMethod<floatT,true,HALO,floatT,topChargeDens_imp<floatT,HALO,true>, CorrType<floatT>>; \
template class BlockingMethod<floatT,true,HALO,floatT,EMTtrace<floatT,true,HALO>, CorrType<floatT>>; \
template class BlockingMethod<floatT,true,HALO,Matrix4x4Sym<floatT>,EMTtraceless<floatT,true,HALO>, CorrType<floatT>>;
INIT_PH(CLASS_INIT3)
