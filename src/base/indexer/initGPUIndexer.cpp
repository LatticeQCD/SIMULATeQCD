/* 
 * initGPUIndexer.cpp                                                               
 * 
 * L. Mazur 
 * 
 */

#include "../../define.h"
#include "BulkIndexer.h"
#include "../indexer/HaloIndexer.h"


__device__ __constant__ struct LatticeData globLatDataGPU[MAXHALO+1];

void initGPUBulkIndexer(size_t lx, size_t ly, size_t lz, size_t lt, sitexyzt globCoord, sitexyzt globPos,unsigned int Nodes[4]){

    gpuError_t gpuErr;

    LatticeData latDat[MAXHALO+1];

    for (size_t i = 0; i <= MAXHALO; ++i) {
        latDat[i] = LatticeData(lx,ly,lz,lt,i,Nodes,
                globCoord.x,globCoord.y,globCoord.z,globCoord.t,
                globPos.x,globPos.y,globPos.z,globPos.t);
    }

    gpuErr = gpuMemcpyToSymbol(globLatDataGPU, &latDat, sizeof(LatticeData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
    if (gpuErr)
        GpuError("Failed to send LatticeData to device", gpuErr);

    gpuErr = gpuDeviceSynchronize();
    if (gpuErr)
        GpuError("initGPUBulkIndexer: gpuDeviceSynchronize failed", gpuErr);
}


__device__ __constant__ struct HaloData globHalDataGPU[MAXHALO+1];
__device__ __constant__ struct HaloData globHalDataGPUReduced[MAXHALO+1];

void initGPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt, unsigned int Nodes[4], unsigned int Halos[4]) {

    gpuError_t gpuErr;

    HaloData halDat[MAXHALO+1];
    HaloData halDatReduced[MAXHALO+1];
    for (size_t i = 0; i <= MAXHALO; ++i) {
        halDat[i] = HaloData(lx,ly,lz,lt,i,Nodes);
        halDatReduced[i] = HaloData(lx-2*Halos[0],ly-2*Halos[1],lz-2*Halos[2],lt-2*Halos[3],i,Nodes);
    }

    gpuErr = gpuMemcpyToSymbol(globHalDataGPU, &halDat, sizeof(HaloData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
    if (gpuErr)
        GpuError("Failed to send HaloData to device", gpuErr);
    gpuErr = gpuDeviceSynchronize();
    if (gpuErr)
        GpuError("initGPUHaloIndexer: gpuDeviceSynchronize failed (1)", gpuErr);

    gpuErr = gpuMemcpyToSymbol(globHalDataGPUReduced, &halDatReduced, sizeof(HaloData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
    if (gpuErr)
        GpuError("Failed to send HaloData to device", gpuErr);
    gpuErr = gpuDeviceSynchronize();
    if (gpuErr)
        GpuError("initGPUHaloIndexer: gpuDeviceSynchronize failed (2)", gpuErr);
}
