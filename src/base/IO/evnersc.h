/*
 * evnersc.h
 *
 */

#pragma once

#include "parameterManagement.h"
#include "../indexer/bulkIndexer.h"
#include <iostream>

class EigenHeader : private ParameterList {
private:
    const CommunicationBase &comm;
    size_t header_size;

    Parameter<std::string> dattype;
    Parameter<int> dim[4];
    Parameter<int> spinor_count;
    Parameter<std::string> floatingpoint;


    bool read(std::istream &in, std::string &content) {
        std::string line;
        if (!getline(in, line)) {
            rootLogger.error("Failed to read BEGIN_HEADER line.");
            return false;
        }
        if (line != "BEGIN_HEADER") {
            rootLogger.error("BEGIN_HEADER not found!");
            return false;
        }

        while (getline(in, line)) {
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

    EigenHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;

        add(dattype, "DATATYPE");
        add(dim[0], "DIMENSION_1");
        add(dim[1], "DIMENSION_2");
        add(dim[2], "DIMENSION_3");
        add(dim[3], "DIMENSION_4");
        add(spinor_count, "VECTOR_LENGTH");
        addDefault(floatingpoint, "FLOATING_POINT", std::string("IEEE32BIG"));
    }


    template <size_t HaloDepth>
    friend class EigenFormat;

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

    bool write(std::ostream &out) {
        bool success = true;
        if (comm.IamRoot()) {
            out.precision(10);
            out << "BEGIN_HEADER" << std::endl
                << (*this)
                << "END_HEADER" << std::endl;
            header_size = out.tellp();
            success = !out.fail();
        }
        if (!comm.single()) {
            comm.root2all(header_size);
            comm.root2all(success);
        }
        return success;
    }
};

template<size_t HaloDepth>
class EigenFormat {
private:

    const CommunicationBase &comm;
    EigenHeader header;
    typedef GIndexer<All,HaloDepth> GInd;
    int rows;
    size_t float_size;
    bool switch_endian;
    size_t spinor_size;
    size_t index = std::numeric_limits<size_t>::max(); //position in buffer
    static const bool sep_lines = false; // make the buffer smaller and read each xline separately
                                         // (slow on large lattices, but needs less memory)
    std::vector<char> buf;

    template<class floatT>
    Vect3<floatT> from_buf_vector(floatT *buf) const {
        size_t i = 0;
        Vect3<floatT> U;
        for (size_t k = 0; k < 3; k++) {
            floatT re = buf[i++];
            floatT im = buf[i++];
            U(k) = COMPLEX(floatT)(re, im);
        }
        return U;
    }

    template<class floatT>
    void to_buf_vector(floatT *buf, const Vect3<floatT> &U) const {
        size_t i = 0;
        COMPLEX(floatT) v0 = U.getElement0();
        buf[i++] = v0.cREAL;
        buf[i++] = v0.cIMAG;

        COMPLEX(floatT) v1 = U.getElement1();
        buf[i++] = v1.cREAL;
        buf[i++] = v1.cIMAG;

        COMPLEX(floatT) v2 = U.getElement2();
        buf[i++] = v2.cREAL;
        buf[i++] = v2.cIMAG;
    }

    template<class floatT>
    floatT from_buf_scalar(floatT *buf) const {
        return buf[0];
    }

    template<class floatT>
    void to_buf_scalar(floatT *buf, floatT value) const {
        buf[0] = value;
    }

    void byte_swap() {
        const long count = buf.size() / float_size;
        for (long i = 0; i < count; i++)
            Byte_swap(&buf[i * float_size], float_size);
    }

public:

    EigenFormat(const CommunicationBase &comm)
            : comm(comm), header(comm) {
        rows = 0;
        float_size = 0;
        spinor_size = 0;
        switch_endian = false;
        index = 0;
    }

    bool read_header(std::istream &in) {
        if (!header.read(in)){
            rootLogger.error("header.read() failed!");
            return false;
        }

        bool error = false;
        for (size_t mu = 0; mu < 4; mu++) {
            if (header.dim[mu]() != GInd::getLatData().globalLattice()[mu]) {
                rootLogger.error( "Stored extension N_", mu," = ",header.dim[mu](),
                                  " not equal to expected extension N_", mu," = ",GInd::getLatData().globalLattice()[mu] );
                error = true;
            }
        }

        if (header.dattype() != "EIGEN") {
            rootLogger.error("DATATYPE = " ,  header.dattype() ,  "not recognized.");
            error = true;
        }

        Endianness disken = ENDIAN_AUTO;
        if (header.floatingpoint() == "IEEE32BIG" || header.floatingpoint() == "IEEE32") {
            float_size = 4;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE64BIG") {
            float_size = 8;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE32LITTLE" || header.floatingpoint() == "IEEE32BE") {
            float_size = 4;
            disken = ENDIAN_LITTLE;
        } else if (header.floatingpoint() == "IEEE64LITTLE" || header.floatingpoint() == "IEEE64LE") {
            float_size = 8;
            disken = ENDIAN_LITTLE;
        } else {
            rootLogger.error("Unrecognized FLOATING_POINT " ,  header.floatingpoint());
            error = true;
        }
        switch_endian = switch_endianness(disken);

        spinor_size = 6 * float_size;
        buf.resize(GInd::getLatData().vol4 * spinor_size + 4 * float_size);
        index = buf.size();

        return !error;
    }

    template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
    bool write_header(int diskprec, int spinor_count, Endianness en, std::ostream &out) {

        if (diskprec == 1 || (diskprec == 0 && sizeof(floatT) == sizeof(float)))
            float_size = 4;
        else if (diskprec == 2 || (diskprec == 0 && sizeof(floatT) == sizeof(double)))
            float_size = 8;
        else {
            rootLogger.error("diskprec should be 0, 1 or 2.");
            return false;
        }

        spinor_size = 6 * float_size;
        buf.resize(GInd::getLatData().vol4 * spinor_size + 4 * float_size);

        if (en == ENDIAN_AUTO)
            en = get_endianness(false); //use system endianness
        switch_endian = switch_endianness(en);

        for (size_t mu = 0; mu < 4; mu++)
            header.dim[mu].set(GInd::getLatData().globalLattice()[mu]);

        header.dattype.set("EIGEN");

        std::string fp;
        if (float_size == 4)
            fp = "IEEE32";
        else if (float_size == 8)
            fp = "IEEE64";
        else {
            rootLogger.error("NERSC format must store single or double precision.");
            return false;
        }
        if (en == ENDIAN_LITTLE)
            fp += "LITTLE";
        else
            fp += "BIG";
        header.floatingpoint.set(fp);

        header.spinor_count.set(spinor_count);

        return header.write(out);
    }

    size_t header_size() {
        return header.size();
    }

    size_t spinor_count() {
        return header.spinor_count();
    }

    char *buf_ptr() {
        return &buf[0];
    }

    size_t buf_size() const {
        return buf.size();
    }

    size_t bytes_per_site() const {
        return spinor_size;
    }

    bool end_of_buffer() const {
        return index >= buf.size();
    }

    void process_read_data() {
        if (switch_endian)
            byte_swap();
        index = 0;
    }

    void process_write_data() {
        if (switch_endian)
            byte_swap();
        index = 0;
    }

    template<class floatT>
    Vect3<floatT> get_vector() {
        if (index + spinor_size > buf.size()) {
            throw std::out_of_range("Buffer overrun in get_vector()");
        }
        char *start = &buf[index];
        Vect3<floatT> ret = from_buf_vector<floatT>((floatT *) start);
        index += spinor_size;
        return ret;
    }

    template<class floatT>
    void put_vector(Vect3<floatT> vec) {
        if (index + spinor_size > buf.size()) {
            throw std::out_of_range("Buffer overrun in put_vector()");
        }
        char *start = &buf[index];
        to_buf_vector((floatT *) start, vec);
        index += spinor_size;
    }

    template<class floatT>
    floatT get_scalar() {
        if (index + sizeof(floatT) > buf.size()) {
            throw std::out_of_range("Buffer overrun in get_scalar()");
        }
        floatT ret = from_buf_scalar<floatT>((floatT *) &buf[index]);
        index += sizeof(floatT);
        return ret;
    }

    template<class floatT>
    void put_scalar(floatT value) {
        if (index + sizeof(floatT) > buf.size()) {
            throw std::out_of_range("Buffer overrun in put_scalar()");
        }
        to_buf_scalar<floatT>((floatT *) &buf[index], value);
        index += sizeof(floatT);
    }

    double get_double() {
        if (index + sizeof(double) > buf.size()) {
            throw std::out_of_range("Buffer overrun in get_scalar()");
        }
        double ret = from_buf_scalar<double>((double *) &buf[index]);
        index += sizeof(double);
        return ret;
    }

    void put_double(double value) {
        if (index + sizeof(double) > buf.size()) {
            throw std::out_of_range("Buffer overrun in put_scalar()");
        }
        to_buf_scalar<double>((double *) &buf[index], value);
        index += sizeof(double);
    }
};



