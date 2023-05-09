/*
 * communicationBase_scalar.cpp
 *
 * Scalar implementation with no communication.
 *
 */

#include "../../define.h"
#ifndef COMPILE_WITH_MPI

using namespace BiLat;
Logger rootLogger(OFF);
Logger stdLogger;


//! if compiling scalar, this sets matching default values
CommunicationBase::CommunicationBase( int *argc, char ***argv) {
    myrank = 0;
    num_proc_world = 1;
    _nodes = LatticeDimensions(1,1,1,1);
    rootLogger.setVerbosity(stdLogger.getVerbosity());
}

void CommunicationBase::init(const LatticeParameters &lp ) {
    // check if more than one node is requested
    if (LatticeDimensions(lp.nodeDim) != nodes()) {
        throw std::runtime_error(stdLogger.fatal("Running scalar code, but more than one node set."));
    }

};

#endif
