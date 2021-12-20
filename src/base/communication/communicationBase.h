//
// Created by Lukas Mazur on 11.10.17.
//

#ifndef COMMUNICATOR_COMMUNICATIONBASE_H
#define COMMUNICATOR_COMMUNICATIONBASE_H

#include <iostream>
#include <cstdlib>
#include <mpi.h>

#ifdef OPEN_MPI

#include <mpi-ext.h> /* Needed for CUDA-aware check */

#endif

#define COMBASE_DEBUG 0

#include <cstring>
#include <complex>
#include "../../define.h"
#include "../LatticeDimension.h"
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include "../math/gcomplex.h"
#include "../math/gsu3.h"
#include "../IO/misc.h"
#include "haloOffsetInfo.h"
#include "../math/matrix4x4.h"

template<class floatT>
class GSU3;

/*
	     T Z Y X 	   Lies on another node
0	     0 0 0 0       0  (everything local)
====================================================
1   	 0 0 0 1	   1  (X, offset 2 hplanes LY*LZ*LT )
2   	 0 0 1 0       2  (Y, offset 2 hplanes LX*LZ*LT )
3   	 0 1 0 0       4  (Z, offset 2 hplanes LX*LY*LT )
4   	 1 0 0 0       8  (T, offset 2 hplanes LX*LY*LZ )
====================================================
5   	 0 0 1 1       3  (XY, offset 4 planes LZ*LT )
6   	 0 1 0 1       5  (XZ, offset 4 planes LY*LT )
7   	 1 0 0 1       9  (XT, offset 4 planes LY*LZ )

8   	 0 1 1 0       6  (YZ, offset 4 planes LX*LT )
9   	 1 0 1 0       10 (YT, offset 4 planes LX*LZ )

10  	 1 1 0 0       12 (ZT, offset 4 planes LX*LY )
====================================================
11  	 0 1 1 1        7 (XYZ,  offset 8 length LT )
12  	 1 0 1 1       11 (XYT,  offset 8 length LZ )
13  	 1 1 0 1       13 (XZT,  offset 8 length LY )
14  	 1 1 1 0       14 (YZT,  offset 8 length LX )
====================================================
15  	 1 1 1 1       15 (XYZT, offset 16 lattice corners)



        T Z Y X
0       0 0 0 0
1       0 0 0 1
2       0 0 1 0
3       0 0 1 1
4       0 1 0 0
5       0 1 0 1
6       0 1 1 0
7       0 1 1 1

8       1 0 0 0
9       1 0 0 1
10      1 0 1 0
11      1 0 1 1
12      1 1 0 0
13      1 1 0 1
14      1 1 1 0
15      1 1 1 1

1st Block
left:          right:
0 0 0 0 (0)    0 0 0 1 (1)

2nd Block:
left:          right:
0 0 0 0 (0)    0 0 1 1 (3)
0 0 0 1 (1)    0 0 1 0 (4)

3rd Block
left:          right:
0 0 0 0 (0)    0 1 1 1 (7)
0 0 0 1 (1)    0 1 1 0 (6)
0 0 1 0 (2)    0 1 0 1 (5)
0 0 1 1 (3)    0 1 0 0 (4)

4th Block
left:          right:
0 0 0 0 (0)    1 1 1 1 (15)
0 0 0 1 (1)    1 1 1 0 (14)
0 0 1 0 (2)    1 1 0 1 (13)
0 0 1 1 (3)    1 1 0 0 (12)
0 1 0 0 (4)    1 0 1 1 (11)
0 1 0 1 (5)    1 0 1 0 (10)
0 1 1 0 (6)    1 0 0 1 (9)
0 1 1 1 (7)    1 0 0 0 (8)
*/


enum IO_Mode {
    READ, WRITE, READWRITE
};

class CommunicationBase {
private:
    int world_size; /// Total number of processes

    LatticeDimensions dims;
    MPI_Comm cart_comm;
    MPI_Comm node_comm;
    MPI_Info mpi_info;
    ProcessInfo myInfo;

    NeighborInfo neighbor_info;

    void _MPI_fail(int ret, const std::string& func);


    MPI_File fh;
    MPI_Datatype basetype;
    MPI_Datatype fvtype;

    void initNodeComm();

public:
    CommunicationBase(int *argc, char ***argv);

    void init(const LatticeDimensions &Dim, const LatticeDimensions &Topo = LatticeDimensions());

    ~CommunicationBase();


    bool gpuAwareMPIAvail() const {
#ifdef USE_CUDA_AWARE_MPI
        return true;
#else
        return false;
#endif
    }

    bool useGpuP2P() {
#ifdef USE_CUDA_P2P
        return true;
#else
        return false;
#endif
    }

    NeighborInfo &getNeighborInfo() { return neighbor_info; }

    const LatticeDimensions &mycoords() { return myInfo.coord; } /// Cartesian coordinates of this process
    const LatticeDimensions &nodes();// { return dims; }            /// Number of nodes in Cartesian grid
    bool IamRoot() const RET0_IF_SCALAR;// { return (myInfo.world_rank == 0); }

    int MyRank() { return myInfo.world_rank; }

    int getNumberProcesses() { return world_size; }

    /// Return rank of process with given coordinates
    int getRank(LatticeDimensions) const RET0_IF_SCALAR;

    MPI_Comm getCart_comm() const { return cart_comm; }

    /// Return if only a single process is running
    bool single() const { return (world_size == 1); }

    /// Send values from root node to all others.
    void root2all(int &) const EMPTY_IF_SCALAR;

    void root2all(int64_t &) const EMPTY_IF_SCALAR;

    void root2all(bool &) const EMPTY_IF_SCALAR;

    void root2all(float &) const EMPTY_IF_SCALAR;

    void root2all(double &) const EMPTY_IF_SCALAR;

    void root2all(GCOMPLEX(float) &) const EMPTY_IF_SCALAR;

    void root2all(GCOMPLEX(double) &) const EMPTY_IF_SCALAR;

    void root2all(Matrix4x4Sym<float> &) const EMPTY_IF_SCALAR;

    void root2all(Matrix4x4Sym<double> &) const EMPTY_IF_SCALAR;

    void root2all(GSU3<float> &) const EMPTY_IF_SCALAR;

    void root2all(GSU3<double> &) const EMPTY_IF_SCALAR;

    void root2all(std::vector<int> &) const EMPTY_IF_SCALAR;

    void root2all(std::vector<float> &) const EMPTY_IF_SCALAR;

    void root2all(std::vector<double> &) const EMPTY_IF_SCALAR;

    void root2all(std::vector<std::complex<float> > &) const EMPTY_IF_SCALAR;

    void root2all(std::vector<std::complex<double> > &) const EMPTY_IF_SCALAR;

    void root2all(std::string &) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) a  value
    int reduce(int a) const RETa_IF_SCALAR;

    uint32_t reduce(uint32_t a) const RETa_IF_SCALAR;

    float reduce(float a) const RETa_IF_SCALAR;

    double reduce(double a) const RETa_IF_SCALAR;

    float reduceMax(float a) const RETa_IF_SCALAR;

    double reduceMax(double a) const RETa_IF_SCALAR;

    /// Reduce (summing up) an array of complex values
    std::complex<float> reduce(std::complex<float> a) const RETa_IF_SCALAR;

    std::complex<double> reduce(std::complex<double> a) const RETa_IF_SCALAR;

    /// Reduce (summing up) a complex  value
    GCOMPLEX(float) reduce(GCOMPLEX(float) a) const RETa_IF_SCALAR;

    GCOMPLEX(double) reduce(GCOMPLEX(double) a) const RETa_IF_SCALAR;

    /// Reduce (summing up) an array of double values, replacing values
    void reduce(float *, int) const EMPTY_IF_SCALAR;

    void reduce(uint32_t *, int) const EMPTY_IF_SCALAR;

    void reduce(double *, int) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) an array of values, replacing values
    void reduce(GCOMPLEX(float) *, int) const EMPTY_IF_SCALAR;

    void reduce(GCOMPLEX(double) *, int) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) an array of matrix values, replacing values
    void reduce(Matrix4x4Sym<float> *, int) const EMPTY_IF_SCALAR;

    void reduce(Matrix4x4Sym<double> *, int) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) an array of GSU3(double) values, replacing values
    void reduce(GSU3<float> *, int) const EMPTY_IF_SCALAR;

    void reduce(GSU3<double> *, int) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) an array of complex values, replacing values
    void reduce(std::complex<float> *, int) const EMPTY_IF_SCALAR;

    void reduce(std::complex<double> *, int) const EMPTY_IF_SCALAR;

    /// Reduce (summing up) a Matrix4x4Sym value
    Matrix4x4Sym<float> reduce(Matrix4x4Sym<float> a) const RETa_IF_SCALAR;

    Matrix4x4Sym<double> reduce(Matrix4x4Sym<double> a) const RETa_IF_SCALAR;

    /// Reduce (summing up) a GSU3 value
    GSU3<float> reduce(GSU3<float> a) const RETa_IF_SCALAR;

    GSU3<double> reduce(GSU3<double> a) const RETa_IF_SCALAR;

    /// Average values from all processes
    float globalAverage(float a) const RETa_IF_SCALAR;

    double globalAverage(double a) const RETa_IF_SCALAR;

    /// Average values from all processes
    std::complex<float> globalAverage(std::complex<float> a) const RETa_IF_SCALAR;

    std::complex<double> globalAverage(std::complex<double> a) const RETa_IF_SCALAR;

    /// Find the minimal value from all processes
    float globalMinimum(float a) const RETa_IF_SCALAR;

    double globalMinimum(double a) const RETa_IF_SCALAR;

    /// Find the maximum value from all processes
    float globalMaximum(float a) const RETa_IF_SCALAR;

    double globalMaximum(double a) const RETa_IF_SCALAR;

    /// Check indices
    void failIdx(int mu, int plus) {
        if ((mu < 0) || (mu > 4) || (plus < 0) || (plus > 1)) {
            throw std::runtime_error(stdLogger.fatal("Fail IDX/mu/plus in CommunicationBase"));
        }
    }

    /// Set a global barrier (program flow continues only if all processes have reached barrier)
    void globalBarrier() const EMPTY_IF_SCALAR;

    void nodeBarrier() const EMPTY_IF_SCALAR;


    template<bool onDevice>
    int updateSegment(HaloSegment hseg, size_t direction, int leftRight, HaloOffsetInfo<onDevice> &HalInfo);

    template<bool onDevice>
    void updateAll(HaloOffsetInfo<onDevice> &HalInfo,
                   unsigned int haltype = COMM_BOTH | AllTypes);

    void initIOBinary(std::string fileName, size_t filesize, size_t bytesPerSite, size_t displacement,
                      LatticeDimensions globalLattice, LatticeDimensions localLattice, IO_Mode mode);

    void writeBinary(void *buffer, size_t elemCount);

    void readBinary(void *buffer, size_t elemCount);

    void closeIOBinary();

    static std::string gpuAwareMPICheck();

    std::string getLocalInfoString() const;

};

#endif //COMMUNICATOR_COMMUNICATIONBASE_H
