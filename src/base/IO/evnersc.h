/*
 * nersc.h
 *
 */

#pragma once

#include "parameterManagement.h"
#include "../../spinor/new_eigenpairs.h"
#include <iostream>

class evNerscHeader : virtual private ParameterList {
private:
    const CommunicationBase &comm;
    int header_size;

    // Parameter<std::string> dattype;
    // Parameter<int> dim[4];
    Parameter<std::string> checksum;
    Parameter<std::string> floatingpoint;
    // Parameter<double> linktrace;
    // Parameter<double> plaq;


    bool read(std::istream &in, std::string &content) {
        if (in.fail()) {
            rootLogger.error("Could not open file.");
            return false;
        }
        std::string line;

        getline(in, line);
        if (line != "BEGIN_HEADER") {
            rootLogger.error("BEGIN_HEADER not found!");
            return false;
        }

        while (!in.fail()) {
            getline(in, line);
            if (line == "END_HEADER")
                break;
            content.append(line + '\n');
        }
        if (in.fail()) {
            rootLogger.error("END_HEADER not found!");
            return false;
        }
        header_size = in.tellg();
        return true;
    }

    evNerscHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;

        // add(dattype, "DATATYPE");
        // add(dim[0], "DIMENSION_1");
        // add(dim[1], "DIMENSION_2");
        // add(dim[2], "DIMENSION_3");
        // add(dim[3], "DIMENSION_4");
        add(checksum, "CHECKSUM");
        // add(linktrace, "LINK_TRACE");
        // add(plaq, "PLAQUETTE");
        addDefault(floatingpoint, "FLOATING_POINT", std::string("IEEE32BIG"));
    }


    template <size_t HaloDepth>
    friend class evNerscFormat;

public:
    size_t size() const {
        return header_size;
    }

    // called from all nodes, but only root node has already opened file
    bool read(std::istream &in) {
        std::string content;
        bool success = true;
        if (comm.IamRoot())
            success = read(in, content);

        if (!comm.single()) {
            comm.root2all(success);
            if (success) {
                comm.root2all(header_size);
                comm.root2all(content);
            }
        }
        if (!success)
            return false;
        std::istringstream str(content);
        return readstream(str, "NERSC", true);
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
    int su3_size;
    size_t index; //position in buffer
    // static const bool sep_lines = false; // make the buffer smaller and read each xline separately
    //                                      // (slow on large lattices, but needs less memory)
    std::vector<char> buf;

    template<class f2>
    gVect3<f2> from_buf(f2 *buf) const {
        int i = 0;
        gVect3<f2> U;
        for (int k = 0; k < 3; k++) {
            f2 re = buf[i++];
            f2 im = buf[i++];
            U(k) = GCOMPLEX(f2)(re, im);

        }
        // if (rows == 2 || sizeof(f1) != sizeof(f2))
        //     U.su3unitarize();
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
        float_size = 0;
        su3_size = 0;
        switch_endian = false;
        stored_checksum = 0;
        computed_checksum = 0;
        index = 0;
    }

    bool read_header(std::istream &in) {
        rootLogger.push_verbosity(OFF);
        if (!header.read(in)){
            rootLogger.error("header.read() failed!");
            return false;
        }
        rootLogger.pop_verbosity();

        bool error = false;

        Endianness disken = ENDIAN_AUTO;
        if (header.floatingpoint() == "IEEE32BIG" || header.floatingpoint() == "IEEE32") {
            float_size = 4;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE64BIG") {
            float_size = 8;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE32LITTLE") {
            float_size = 4;
            disken = ENDIAN_LITTLE;
        } else if (header.floatingpoint() == "IEEE64LITTLE") {
            float_size = 8;
            disken = ENDIAN_LITTLE;
        } else {
            rootLogger.error("Unrecognized FLOATING_POINT " ,  header.floatingpoint());
            error = true;
        }
        switch_endian = switch_endianness(disken);

        std::stringstream s(header.checksum());
        s >> std::hex >> stored_checksum;
        if (s.fail()) {
            rootLogger.error("Could not interpret checksum " ,
                               header.checksum() ,  "as hexadecimal number.");
            error = true;
        }

        su3_size = 2 * 3 * float_size;
        buf.resize(GInd::getLatData().vol4 * su3_size);
        index = buf.size();

        return !error;
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
        return su3_size;
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
    gVect3<floatT> get() {
        char *start = &buf[index];
        gVect3<floatT> ret;
        if (float_size == 4)
            ret = from_buf<float>((float *) start);
        else if (float_size == 8)
            ret = from_buf<double>((double *) start);
        index += su3_size;
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



