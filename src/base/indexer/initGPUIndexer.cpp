/*
 * initGPUIndexer.cpp
 *
 * L. Mazur
 *
 */

#include "../../define.h"
#include "bulkIndexer.h"
#include "../indexer/haloIndexer.h"


// We record the set of constant memory symbols to static arrays. We are avoiding using objects with non-trivial Ctors
// to not risk they to be constructed after the static object that uses them. There will be an instance of the static
// objects per compilation unit, so the ordering is hard to control.
static const size_t globLatDataRegisterSize = 16;
static LatticeData* globLatDataGPURegister[globLatDataRegisterSize] = {0};
globLatDataRegistor::globLatDataRegistor(LatticeData &v) {
   for (size_t InstanceIdx = 0; InstanceIdx < globLatDataRegisterSize; ++InstanceIdx)
      if (LatticeData *& slot = globLatDataGPURegister[InstanceIdx]; !slot) {
         slot = &v;
         printf("Got instance of globLatData handler %p (index %ld)\n", &v, InstanceIdx);
         return;
      }

   assert(false && "Not enough slots to register globLatData slots.");
}

void initGPUBulkIndexer(size_t lx, size_t ly, size_t lz, size_t lt, sitexyzt globCoord, sitexyzt globPos,unsigned int Nodes[4]){

    gpuError_t gpuErr;

    LatticeData latDat[MAXHALO+1];

    for (size_t i = 0; i <= MAXHALO; ++i) {
        latDat[i] = LatticeData(lx,ly,lz,lt,i,Nodes,
                globCoord.x,globCoord.y,globCoord.z,globCoord.t,
                globPos.x,globPos.y,globPos.z,globPos.t);
    }

    for (size_t InstanceIdx = 0; InstanceIdx < globLatDataRegisterSize; ++InstanceIdx)
      if (auto* slot = globLatDataGPURegister[InstanceIdx]; slot ) {

        printf("Setting globLatData handler %p (index %ld)\n", slot, InstanceIdx);

        gpuErr = gpuMemcpyToSymbol(*slot, &latDat, sizeof(LatticeData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
        if (gpuErr)
            GpuError("Failed to send LatticeData to device", gpuErr);

        gpuErr = gpuDeviceSynchronize();
        if (gpuErr)
            GpuError("initGPUBulkIndexer: gpuDeviceSynchronize failed", gpuErr);
      } else
         break;

    printf("Done setting all instances of device LatticeData handlers!\n");
}


// We record the set of constant memory symbols to static arrays. We are avoiding using objects with non-trivial Ctors
// to not risk they to be constructed after the static object that uses them. There will be an instance of the static
// objects per compilation unit, so the ordering is hard to control.
static const size_t globHalDataRegisterSize = 16;
static HaloData* globHalDataGPURegister[globHalDataRegisterSize] = {0};
static HaloData* globHalDataGPUReducedRegister[globHalDataRegisterSize] = {0};
globHalDataRegistor::globHalDataRegistor(HaloData &v1, HaloData &v2) {
   for (size_t InstanceIdx = 0; InstanceIdx < globHalDataRegisterSize; ++InstanceIdx)
      if (HaloData *&slot1 = globHalDataGPURegister[InstanceIdx], 
                   *&slot2 = globHalDataGPUReducedRegister[InstanceIdx]; !slot1 && !slot2) {
         slot1 = &v1;
         slot2 = &v2;
         printf("Got instances of globHalData handler %p, %p (index %ld)\n", &v1, &v2, InstanceIdx);
         return;
      }

   assert(false && "Not enough slots to register globHalData slots.");
}

void initGPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt, unsigned int Nodes[4], unsigned int Halos[4]) {

    gpuError_t gpuErr;

    HaloData halDat[MAXHALO+1];
    HaloData halDatReduced[MAXHALO+1];
    for (size_t i = 0; i <= MAXHALO; ++i) {
        halDat[i] = HaloData(lx,ly,lz,lt,i,Nodes);
        halDatReduced[i] = HaloData(lx-2*Halos[0],ly-2*Halos[1],lz-2*Halos[2],lt-2*Halos[3],i,Nodes);
    }
    for (size_t InstanceIdx = 0; InstanceIdx < globHalDataRegisterSize; ++InstanceIdx)
      if (auto *slot1 = globHalDataGPURegister[InstanceIdx],
               *slot2 = globHalDataGPUReducedRegister[InstanceIdx]; slot1 && slot2 ) {

        printf("Setting globHalData handler %p, %p (index %ld)\n", slot1, slot2, InstanceIdx);

        gpuErr = gpuMemcpyToSymbol(*slot1, &halDat, sizeof(HaloData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
        if (gpuErr)
            GpuError("Failed to send HaloData to device", gpuErr);
        gpuErr = gpuDeviceSynchronize();
        if (gpuErr)
            GpuError("initGPUHaloIndexer: gpuDeviceSynchronize failed (1)", gpuErr);

        gpuErr = gpuMemcpyToSymbol(*slot2, &halDatReduced, sizeof(HaloData[MAXHALO+1]), 0, gpuMemcpyHostToDevice);
        if (gpuErr)
            GpuError("Failed to send HaloData to device", gpuErr);
        gpuErr = gpuDeviceSynchronize();
        if (gpuErr)
            GpuError("initGPUHaloIndexer: gpuDeviceSynchronize failed (2)", gpuErr);    
      } else
         break;

    printf("Done setting all instances of device HaloData handlers!\n");
   
}
