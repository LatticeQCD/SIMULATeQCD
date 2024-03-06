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

    //compute checksum of 'bytes' bytes at beginning of buffer
    uint32_t checksum(size_t bytes) {
        uint32_t result = 0;
        uint32_t *dat = (uint32_t *) &buf[0];
        for (size_t i = 0; i < bytes / 4; i++)
            result += dat[i];
        return result;
    }

    template<class floatT>
    uint32_t checksum(GSU3<floatT> U) {
        if (float_size == 4)
            to_buf((float *) &buf[0], U);
        else if (float_size == 8)
            to_buf((double *) &buf[0], U);
        return checksum(su3_size);
    }

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
};



