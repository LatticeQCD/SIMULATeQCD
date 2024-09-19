/*
 * nersc.h
 *
 */

#pragma once

#include "parameterManagement.h"
#include <iostream>

class evNerscHeader : virtual private ParameterList {
private:
    const CommunicationBase &comm;
    int header_size;

    bool read(std::istream &in, double * lambda) {
        if (in.fail()) {
            rootLogger.error("Could not open file.");
            return false;
        } else {
            in.read((char*)lambda, sizeof(double));
            header_size = in.tellg();
            return true;
        }
    }

    evNerscHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;
    }


    template <size_t HaloDepth>
    friend class evNerscFormat;

public:
    size_t size() const {
        return header_size;
    }

    // called from all nodes, but only root node has already opened file
    bool read(std::istream &in, double &content) {
        // double content;
        bool success = true;
        if (comm.IamRoot())
            success = read(in, &content);

        if (!comm.single()) {
            comm.root2all(success);
            if (success) {
                comm.root2all(header_size);
                comm.root2all(content);
            }
        }
        if (!success) {
            return false;
        } else {
            return true;
        }
    }
};

template<size_t HaloDepth>
class evNerscFormat {
private:

    const CommunicationBase &comm;
    evNerscHeader header;
    typedef GIndexer<All,HaloDepth> GInd;
    int rows;
    int float_size;
    bool switch_endian;
    uint32_t stored_checksum, computed_checksum;
    int local_size;
    size_t index; //position in buffer
    // static const bool sep_lines = false; // make the buffer smaller and read each xline separately
    //                                      // (slow on large lattices, but needs less memory)
    std::vector<char> buf;

    template<class floatT>
    Vect3<floatT> from_buf(floatT *buf)  {
        int i = 0;
        Vect3<floatT> U;
        for (int k = 0; k < 3; k++) {
            floatT re = buf[i++];
            floatT im = buf[i++];
            U(k) = COMPLEX(floatT)(re, im);
        }
        return U;
    }

    void byte_swap() {
        const long count = buf.size() / float_size;
        for (long i = 0; i < count; i++)
            Byte_swap(&buf[i * float_size], float_size);
    }

    //compute checksum of 'bytes' bytes at beginning of buffer
    uint32_t checksum(size_t bytes) {
        uint32_t result = 0;
        uint32_t *dat = (uint32_t *) &buf[0];
        for (size_t i = 0; i < bytes / 4; i++)
            result += dat[i];
        return result;
    }

public:

    evNerscFormat(CommunicationBase &comm) : comm(comm), header(comm) {
        rows = 0;
        float_size = sizeof(float_t);
        local_size = 8 * float_size;
        switch_endian = false;
        stored_checksum = 0;
        computed_checksum = 0;
        index = 0;
    }

    bool read_header(std::istream &in, double &content) {
        if (!header.read(in, content)){
            rootLogger.error("header.read() failed!");
            return false;
        } else {

            buf.resize(GInd::getLatData().vol4 * local_size);
            index = buf.size();

            return true;
        }
    }

    size_t header_size() {
        return header.size();
    }

    char *buf_ptr() {
        return &buf[0];
    }

    size_t buf_size() const {
        return buf.size();
    }

    size_t bytes_per_site() const {
        return local_size;
    }

    bool end_of_buffer() const {
        return index >= buf.size();
    }

    void process_read_data() {
        if (switch_endian)
            byte_swap();
        computed_checksum += checksum(buf.size());
        index = 0;
    }
    template<class floatT>
    Vect3<floatT> get() {
        char *start = &buf[index];
        Vect3<floatT> ret = from_buf<floatT>((floatT *) start);
        index += local_size;
        return ret;
    }

    bool checksums_match() {
        uint32_t checksum = comm.reduce(computed_checksum);
        if (stored_checksum != checksum) {
            rootLogger.error("Checksum mismatch! "
                               ,  std::hex ,  stored_checksum ,  " != "
                               ,  std::hex ,  checksum);
            return false;
        }
        return true;
    }
};



