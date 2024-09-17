#pragma once
#include <iostream>
#include <cstdlib>
#include <mpi.h>


#include <cstring>
#include <complex>
#include "../../define.h"
#include "../latticeDimension.h"
#include "../latticeParameters.h"

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include "../math/complex.h"
#include "../math/su3.h"
#include "../IO/misc.h"
#include "haloOffsetInfo.h"
#include "../math/matrix4x4.h"

template<class floatT>
class SU3;

enum IO_Mode {
    READ, WRITE, READWRITE
};

class CommunicationBaseSycl {
private:
    bool _forceHalos;
    int world_size;

    LatticeDimensions dims;
    
    MPI_Comm cart_comm;
    MPI_Comm node_comm;
    MPI_Info mpi_info;
    
    ProcessInfo myInfo;
    NeighborInfo neighbor_info;

    void _MPI_fail(int ret, const std::string& func);
    
    bool _initialized = false;

    MPI_File fh;
    MPI_Datatype basetype;
    MPI_Datatype fvtype;
    
    void initNodeComm();

public:
    CommunicationBaseSycl(int *argc, char***argv, bool forceHalos=false);

    void init(const LatticeDimensions &Dim, const LatticeDimensions &Topo = LatticeDimensions());
    
    NeighborInfo &getNeighborInfo() { return neighbor_info; }

    const LatticeDimensions &mycoords() { return myInfo.coord; }
    const LatticeDimensions &nodes();

    bool IamRoot();

    int MyRank() { return myInfo.world_rank; }

    bool forceHalos() const { return _forceHalos; }
    void forceHalos(bool forceHalos) { _forceHalos = forceHalos; }

    int getNumberProcesses() { return world_size; }

    int getRank(LatticeDimensions);

    MPI_Comm getCart_comm() const { return cart_comm; }

    
}