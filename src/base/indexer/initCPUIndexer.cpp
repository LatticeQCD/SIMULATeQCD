/*
 * initCPUIndexer.cpp
 *
 * L. Mazur
 *
 */

#include "../../define.h"
#include "bulkIndexer.h"
#include "../indexer/haloIndexer.h"

void initIndexer(const size_t HaloDepth, const LatticeParameters &param, CommunicationBase &comm) {
    bool forceAllHalos = comm.forceHalos();
    LatticeDimensions _globalLattice(param.latDim);
    LatticeDimensions _localLattice(_globalLattice / comm.nodes());

    rootLogger.info("Initialize Indexer with HaloDepth = " ,  HaloDepth ,  CoutColors::yellow
                      ,  (forceAllHalos ? " and force Halos in all directions" : "") ,  CoutColors::reset);

    if (_localLattice * comm.nodes() != _globalLattice) {
        throw std::runtime_error(stdLogger.fatal("Lattice ", _globalLattice, " not divisible into Nodes ", comm.nodes()));
    }
    if ((_localLattice[0] < 1) || (_localLattice[1] < 1) || (_localLattice[2] < 1) || (_localLattice[3] < 1)) {
        throw std::runtime_error(stdLogger.fatal("lattice : size of lattice must be > 0 !"));
    }
    if ((_localLattice[0] % 2) || (_localLattice[1] % 2) || (_localLattice[2] % 2) || (_localLattice[3] % 2)) {
        throw std::runtime_error(stdLogger.fatal("lx=", _localLattice[0], " ly=", _localLattice[1],
                       " lz=", _localLattice[2], " lt=", _localLattice[3],
                       ". lattice : size of lattice must be even!"));
    }

    for(int i = 0; i < 4; i++) {
        if (((_localLattice[i] <= (int)(2 * HaloDepth)) && comm.nodes()[i] > 1) || ((_localLattice[i] <= (int)(2 * HaloDepth)) && forceAllHalos )) {
            throw std::runtime_error(stdLogger.fatal("One dimension smaller or equal HaloSize"));
        }
    }
    unsigned int Nodes[4] = {forceAllHalos ? 2 : (unsigned int)param.nodeDim()[0],
                             forceAllHalos ? 2 : (unsigned int)param.nodeDim()[1],
                             forceAllHalos ? 2 : (unsigned int)param.nodeDim()[2],
                             forceAllHalos ? 2 : (unsigned int)param.nodeDim()[3]};

    unsigned int Halos[4] = {Nodes[0] > 1 ? (unsigned int)HaloDepth : 0,
                             Nodes[1] > 1 ? (unsigned int)HaloDepth : 0,
                             Nodes[2] > 1 ? (unsigned int)HaloDepth : 0,
                             Nodes[3] > 1 ? (unsigned int)HaloDepth : 0};

    sitexyzt gCoord(_globalLattice[0], _globalLattice[1], _globalLattice[2], _globalLattice[3]);

    LatticeDimensions globalpos = comm.mycoords() * _localLattice;
    sitexyzt gPos(globalpos[0], globalpos[1], globalpos[2], globalpos[3]);

#ifndef CPUONLY
    initGPUBulkIndexer((size_t)_localLattice[0], (size_t)_localLattice[1],(size_t)_localLattice[2], (size_t)_localLattice[3], gCoord, gPos,Nodes);
    initGPUHaloIndexer((size_t)_localLattice[0], (size_t)_localLattice[1],(size_t)_localLattice[2], (size_t)_localLattice[3], Nodes,Halos);
#endif
    initCPUBulkIndexer((size_t)_localLattice[0], (size_t)_localLattice[1],(size_t)_localLattice[2], (size_t)_localLattice[3], gCoord, gPos,Nodes);
    initCPUHaloIndexer((size_t)_localLattice[0], (size_t)_localLattice[1],(size_t)_localLattice[2], (size_t)_localLattice[3], Nodes,Halos);

    stdLogger.debug("Local size without Halos: " ,  globLatDataCPU[HaloDepth].lx ,  " "
                      ,  globLatDataCPU[HaloDepth].ly ,  " " ,  globLatDataCPU[HaloDepth].lz ,  " "
                      ,  globLatDataCPU[HaloDepth].lt);
    stdLogger.debug("Local size with Halos: " ,  globLatDataCPU[HaloDepth].lxFull ,  " "
                      ,  globLatDataCPU[HaloDepth].lyFull ,  " " ,  globLatDataCPU[HaloDepth].lzFull ,  " "
                      ,  globLatDataCPU[HaloDepth].ltFull);
}


const struct LatticeData globLatDataCPU[MAXHALO+1];

template <size_t HaloDepth>
void initCPUBulk(size_t lx, size_t ly, size_t lz, size_t lt, sitexyzt globCoord, sitexyzt globPos, unsigned int Nodes[4]){

    globLatDataCPU[HaloDepth] = LatticeData(lx,ly,lz,lt,HaloDepth, Nodes,
            globCoord.x,globCoord.y,globCoord.z,globCoord.t,
            globPos.x,globPos.y,globPos.z,globPos.t);

}


void initCPUBulkIndexer(size_t lx, size_t ly, size_t lz, size_t lt,sitexyzt globCoord, sitexyzt globPos, unsigned int Nodes[4]) {

#ifdef HALODEPTH_0
    initCPUBulk<0>(lx,ly,lz,lt,globCoord,globPos,Nodes);
#endif
#ifdef HALODEPTH_1
    initCPUBulk<1>(lx,ly,lz,lt,globCoord,globPos,Nodes);
#endif
#ifdef HALODEPTH_2
    initCPUBulk<2>(lx,ly,lz,lt,globCoord,globPos,Nodes);
#endif
#ifdef HALODEPTH_3
    initCPUBulk<3>(lx,ly,lz,lt,globCoord,globPos,Nodes);
#endif
#ifdef HALODEPTH_4
    initCPUBulk<4>(lx,ly,lz,lt,globCoord,globPos,Nodes);
#endif
}

struct HaloData globHalDataCPU[MAXHALO+1];
struct HaloData globHalDataCPUReduced[MAXHALO+1];

void initCPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt, unsigned int Nodes[4], unsigned int Halos[4]) {
    for (size_t i = 0; i <= MAXHALO; ++i) {
        globHalDataCPU[i] = HaloData(lx,ly,lz,lt,i,Nodes);
        globHalDataCPUReduced[i] = HaloData(lx-2*Halos[0],ly-2*Halos[1],lz-2*Halos[2],lt-2*Halos[3],i,Nodes);

    }
}
