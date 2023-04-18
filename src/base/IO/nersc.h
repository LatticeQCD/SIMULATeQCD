/*
 * nersc.h
 *
 */

#ifndef INC_NERSC_H
#define INC_NERSC_H

#include "parameterManagement.h"
#include "misc.h"
#include "../../gauge/gaugefield.h"
#include "../../gauge/GaugeAction.h"
#include "../LatticeContainer.h"
#include <iostream>

class NerscHeader : virtual private ParameterList {
private:
    const CommunicationBase &comm;
    int header_size;

    Parameter<std::string> dattype;
    Parameter<int> dim[4];
    Parameter<std::string> checksum;
    Parameter<std::string> floatingpoint;
    Parameter<double> linktrace;
    Parameter<double> plaq;


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

    NerscHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;

        add(dattype, "DATATYPE");
        add(dim[0], "DIMENSION_1");
        add(dim[1], "DIMENSION_2");
        add(dim[2], "DIMENSION_3");
        add(dim[3], "DIMENSION_4");
        add(checksum, "CHECKSUM");
        add(linktrace, "LINK_TRACE");
        add(plaq, "PLAQUETTE");
        addDefault(floatingpoint, "FLOATING_POINT", std::string("IEEE32BIG"));
    }


    template <size_t HaloDepth>
    friend class NerscFormat;

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
class NerscFormat {
private:

    const CommunicationBase &comm;
    NerscHeader header;
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

    template<class f1, class f2>
    GSU3<f2> from_buf(f1 *buf) const {
        int i = 0;
        GSU3<f2> U;
        for (int j = 0; j < rows; j++)
            for (int k = 0; k < 3; k++) {
                f2 re = buf[i++];
                f2 im = buf[i++];
                U(j, k) = GCOMPLEX(f2)(re, im);

            }
        if (rows == 2 || sizeof(f1) != sizeof(f2))
            U.su3unitarize();
        return U;
    }

    template<class f1, class f2>
    void to_buf(f1 *buf, const GSU3<f2> &U) const {
        int i = 0;
        for (int j = 0; j < rows; j++)
            for (int k = 0; k < 3; k++) {
                buf[i++] = U(j, k).cREAL;
                buf[i++] = U(j, k).cIMAG;
            }
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

    template<class floatT>
    uint32_t checksum(GSU3<floatT> U) {
        if (float_size == 4)
            to_buf((float *) &buf[0], U);
        else if (float_size == 8)
            to_buf((double *) &buf[0], U);
        return checksum(su3_size);
    }

public:

    NerscFormat(CommunicationBase &comm)
            :comm(comm), header(comm) {
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
        for (int i = 0; i < 4; i++)
            if (header.dim[i]() != GInd::getLatData().globalLattice()[i]) {
                rootLogger.error( "Stored extension N_", i," = ",header.dim[i](),
                                  " not equal to expected extension N_", i," = ",GInd::getLatData().globalLattice()[i] );
                error = true;
            }

        if (header.dattype() == "4D_SU3_GAUGE")
            rows = 2;
        else if (header.dattype() == "4D_SU3_GAUGE_3x3")
            rows = 3;
        else {
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

        su3_size = 2 * 3 * rows * float_size;
        buf.resize((sep_lines ? GInd::getLatData().lx : GInd::getLatData().vol4) * 4 * su3_size);
        index = buf.size();

        return !error;
    }

    template<class floatT,bool onDevice, CompressionType comp>
    bool write_header(Gaugefield<floatT, onDevice, HaloDepth,comp> &gf, gaugeAccessor<floatT,comp> gaugeAccessor, int _rows,
                      int diskprec, Endianness en, std::ostream &out) {
        rows = _rows;
        if (diskprec == 1 || (diskprec == 0 && sizeof(floatT) == sizeof(float)))
            float_size = 4;
        else if (diskprec == 2 || (diskprec == 0 && sizeof(floatT) == sizeof(double)))
            float_size = 8;
        else {
            rootLogger.error("diskprec should be 0, 1 or 2.");
            return false;
        }
        su3_size = 2 * 3 * rows * float_size;
        buf.resize((sep_lines ? GInd::getLatData().lx : GInd::getLatData().vol4) * 4 * su3_size);

        if (en == ENDIAN_AUTO)
            en = get_endianness(false); //use system endianness
        switch_endian = switch_endianness(en);

        for (int mu = 0; mu < 4; mu++)
            header.dim[mu].set(GInd::getLatData().globalLattice()[mu]);

        if (rows == 2)
            header.dattype.set("4D_SU3_GAUGE");
        else if (rows == 3)
            header.dattype.set("4D_SU3_GAUGE_3x3");
        else {
            rootLogger.error("NERSC format must store 2 or 3 rows.");
            return false;
        }

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

        double linktrace = 0;
        stored_checksum = 0;
        for (size_t t = 0; t < GInd::getLatData().lt; t++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
                for (size_t y = 0; y < GInd::getLatData().ly; y++)
                    for (size_t x = 0; x < GInd::getLatData().lx; x++)
                        for (int mu = 0; mu < 4; mu++) {
                            gSite site = GInd::getSite(x, y, z, t);
                            GSU3<floatT> temp = gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
                            linktrace += tr_d(temp);
                            stored_checksum += checksum(temp);
                        }

        header.linktrace.set(comm.reduce(linktrace) / (3 * 4 * GInd::getLatData().globalLattice().mult()));

        GaugeAction<floatT,onDevice,HaloDepth,comp> enDensity(gf);
        header.plaq.set(enDensity.plaquette());

        stored_checksum = comm.reduce(stored_checksum);

        std::stringstream s;
        s << std::hex << stored_checksum;
        header.checksum.set(s.str());
        rootLogger.info("Calculated checksum: " ,  s.str());
        return header.write(out);
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
        return 4 * su3_size;
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

    void process_write_data() {
        if (switch_endian)
            byte_swap();
        index = 0;
    }

    template<class floatT>
    GSU3<floatT> get() {
        char *start = &buf[index];
        GSU3<floatT> ret;
        if (float_size == 4)
            ret = from_buf<float,float>((float *) start);
        else if (float_size == 8)
            ret = from_buf<double,double>((double *) start);
        index += su3_size;
        return ret;
    }

    template<class floatT>
    void put(GSU3<floatT> U) {
        char *start = &buf[index];
        if (float_size == 4)
            to_buf((float *) start, U);
        else if (float_size == 8)
            to_buf((double *) start, U);
        index += su3_size;
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

#endif
