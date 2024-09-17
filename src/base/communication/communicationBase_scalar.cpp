/*
 * communicationBase_scalar.cpp
 *
 * Scalar implementation with no communication.
 *
 */

#include "../../define.h"

#include "../latticeParameters.h"
// #ifndef COMPILE_WITH_MPI

// using namespace BiLat;
Logger rootLogger(OFF);
Logger stdLogger;


//! if compiling scalar, this sets matching default values
CommunicationBase::CommunicationBase( int *argc, char ***argv, bool forceHalos) : _forceHalos(forceHalos) {
    // myrank = 0;
    // num_proc_world = 1;
    // _nodes = LatticeDimensions(1,1,1,1);
    rootLogger.setVerbosity(stdLogger.getVerbosity());
}

// CommunicationBase::~CommunicationBase(){}


void init(const LatticeDimensions &Dim, const LatticeDimensions &Topo = LatticeDimensions()) {}
// #endif
