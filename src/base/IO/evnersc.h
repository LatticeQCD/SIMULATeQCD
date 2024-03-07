/*
 * nersc.h
 *
 */

#pragma once

#include "parameterManagement.h"
#include "../../spinor/new_eigenpairs.h"
#include <iostream>

template<size_t HaloDepth>
class evNerscFormat {
private:
    const CommunicationBase &comm;

    typedef GIndexer<All,HaloDepth> GInd;
    int rows;
    int float_size;
    bool switch_endian;
    uint32_t stored_checksum, computed_checksum;
    int su3_size;
    size_t index; //position in buffer
    static const bool sep_lines = false; // make the buffer smaller and read each xline separately
                                         // (slow on large lattices, but needs less memory)
    std::vector<char> buf;

public:

    evNerscFormat(CommunicationBase &comm) : comm(comm) {
        rows = 0;
        float_size = 0;
        su3_size = 0;
        switch_endian = false;
        stored_checksum = 0;
        computed_checksum = 0;
        index = 0;
    }

    // size_t bytes_per_site() const {
    //     return 4 * su3_size;
    // }
};



