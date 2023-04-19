/* 
 * ildg.h                                                               
 * 
 * R. Larsen, S. Ali, D. Clarke 
 *
 * This class is the reader class for ILDG configurations. ILDG configurations are packed as LIME files. The best
 * way to write this class would be to only allow reading configurations that are exactly in ILDG format, as
 * specified by their documentation. Unfortunately it seems that some codes, while correctly packing their
 * configuration as a LIME file, did not adhere to the ILDG specification. This class can read such configurations
 * anyway. Hence this class should rather be thought of as an "ILDG-type" configuration reader. On the bright side, 
 * we have written it so that the writing is done to correct ILDG specification.
 *
 * To help with understanding LIME format, note the following organizational hierarchy:
 *     message > record > word
 * Message begin (MB) and message end (ME) flags have to be set according to the following rules:
 *
 * (M1)  MB=1 for the first record of the LIME file.
 * (M2)  For any two consecutive records with ME flag me1 and MB flag mb2, respectively, the relation me1=mb2 must hold.
 * (M3)  ME=1 for the last record of the LIME file.
 *
 * Special thanks to H. Simma for the explanation. Since we are not using the message organizational level, we
 * will always set these flags to 1.
 * 
 */

#pragma once
#include "parameterManagement.h"
#include "misc.h"
#include "../../gauge/gaugefield.h"
#include "../../gauge/GaugeAction.h"
#include "../LatticeContainer.h"
#include "../math/gaugeAccessor.h"
#include "checksum.h"
#include <iostream>
#include <time.h>

#define ILDG_MAGIC_NUMBER 1164413355

template<class floatT>
floatT returnEndian(floatT input,bool change) {
    if(change) {
        if(sizeof(input) == 8) {
            return __builtin_bswap64(input);
        } else if(sizeof(input) == 4) {
            return __builtin_bswap32(input);
        }
    } 
    return input;
}

/// INTENT: IN--infoStr, tag; OUT--value
void extractFromTag(std::string infoStr, std::string tag, std::string &value) {
    std::string xmlTagOpen  = "<"+tag+">";
    std::string xmlTagClose = "</"+tag+">";
    /// This if statement checks whether we are at the end of infoStr.
    if(infoStr.find(xmlTagOpen) != std::string::npos) {
        int pos    = infoStr.find(xmlTagOpen);
        int posEnd = infoStr.find(xmlTagClose);
        int strLen = xmlTagOpen.length();
        value      = infoStr.substr(pos+strLen,posEnd-strLen-pos);
    }
}
void extractFromTag(std::string infoStr, std::string tag, int &value) {
    std::string xmlTagOpen  = "<"+tag+">";
    std::string xmlTagClose = "</"+tag+">";
    if(infoStr.find(xmlTagOpen) != std::string::npos) {
        int pos    = infoStr.find(xmlTagOpen);
        int posEnd = infoStr.find(xmlTagClose);
        int strLen = xmlTagOpen.length();
        value      = stoi(infoStr.substr(pos+strLen,posEnd-strLen-pos));
    }
}

class IldgHeader : virtual private ParameterList {
private:
    const CommunicationBase &comm;
    int header_size;

    Parameter<std::string> checksuma, checksumb, floatingpoint; 
    Parameter<int>         dim[4];
    Parameter<double>      linktrace, plaq;

    bool read_info(std::istream &in) {
        int32_t magic_number;
        int64_t data_length;
        int dimx=-1; 
        int dimy=-1;
        int dimz=-1; 
        int dimt=-1;
        int dataPos=-1;
        std::string precision, suma, sumb, lstr;
        bool Endian=false;

        in.read(reinterpret_cast<char *>(&magic_number),sizeof(magic_number));
        if(magic_number != ILDG_MAGIC_NUMBER) {
            if(__builtin_bswap32(magic_number) == ILDG_MAGIC_NUMBER) {
                Endian = true;
                rootLogger.info("testing magic number");
            } else {
                rootLogger.info("could not read magic number");
                return false;
            }
        }

        in.clear();
        in.seekg(0);

        while(in.read(reinterpret_cast<char *>(&magic_number),sizeof(magic_number))){

            if(returnEndian(magic_number,Endian) == ILDG_MAGIC_NUMBER) {
                in.ignore(4);
                in.read(reinterpret_cast<char *>(&data_length),sizeof(data_length));

                if(returnEndian(data_length,Endian) > 100000) {
                    in.ignore(8*16);
                    dataPos = in.tellg();
                    in.ignore(returnEndian(data_length,Endian));
                } else {
                    int bytes = ceil(returnEndian(data_length,Endian)/8.0)*8;

                    in.ignore(8*16);
                    char info[bytes];
                    in.read(info,sizeof(info));
                    std::string info_str(info, bytes);

                    extractFromTag(info_str,"precision",precision);
                    extractFromTag(info_str,"lx"       ,dimx);
                    extractFromTag(info_str,"ly"       ,dimy);
                    extractFromTag(info_str,"lz"       ,dimz);
                    extractFromTag(info_str,"lt"       ,dimt);
                    extractFromTag(info_str,"suma"     ,suma);
                    extractFromTag(info_str,"sumb"     ,sumb);

                    /// This is how QUDA stores their dimension information
                    if(info_str.find("<dims>") != std::string::npos) {
                        int pos = info_str.find("<dims>");
                        int posEnd = info_str.find("</dims>");
                        std::string part;
                        std::stringstream content(info_str.substr(pos+6,posEnd-pos-6));
                        std::getline(content,part,' ');
                        dimx = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimy = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimz = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimt = atoi(part.c_str());
                    }
                }
            }
        }

        dim[0].set(dimx);
        dim[1].set(dimy);
        dim[2].set(dimz);
        dim[3].set(dimt);
        checksuma.set(suma);
        checksumb.set(sumb);
        floatingpoint.set(precision);
        header_size=dataPos;

        in.clear();
        in.seekg(0);

        return true;
    }

    IldgHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;
        add(dim[0], "DIMENSION_1");
        add(dim[1], "DIMENSION_2");
        add(dim[2], "DIMENSION_3");
        add(dim[3], "DIMENSION_4");
        add(checksuma, "CHECKSUMA");
        add(checksumb, "CHECKSUMB");
        add(linktrace, "LINK_TRACE");
        add(plaq, "PLAQUETTE");
        addDefault(floatingpoint, "FLOATING_POINT", std::string("32"));
    }

    template <size_t HaloDepth>
    friend class IldgFormat;

public:
    size_t size() const {
        return header_size;
    }

    // called from all nodes, but only root node has already opened file
    bool read(std::istream &in) {
        std::string content;
        bool success = true;
        if (this->comm.IamRoot()) {
            success = read_info(in);
        }

        if (!comm.single()) {
            comm.root2all(success);
            if (success) {
                int dim0, dim1, dim2, dim3;
                std::string checka, checkb, precision;
                checka = checksuma();
                checkb = checksumb();
                precision = floatingpoint();
                dim0 = dim[0]();
                dim1 = dim[1]();
                dim2 = dim[2]();
                dim3 = dim[3]();
                comm.root2all(header_size);
                comm.root2all(dim0);
                comm.root2all(dim1);
                comm.root2all(dim2);
                comm.root2all(dim3);
                comm.root2all(checka);
                comm.root2all(checkb);
        		comm.root2all(precision);
                dim[0].set(dim0);
                dim[1].set(dim1);
                dim[2].set(dim2);
                dim[3].set(dim3);
                checksuma.set(checka);
                checksumb.set(checkb);
                floatingpoint.set(precision);
            }
        }
        if (!success)
            return false;
        return true;
    }

    bool write(std::ostream &out) {
        bool success = true;
        if (comm.IamRoot()) {
            out.precision(10);
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
class IldgFormat {
private:

    const CommunicationBase &comm;
    IldgHeader header;
    typedef GIndexer<All,HaloDepth> GInd;
    int float_size;
    bool switch_endian;
    uint32_t stored_checksum_nersc, computed_checksum_nersc;
    int su3_size;
    size_t index; //position in buffer
    static const bool sep_lines = false; // make the buffer smaller and read each xline separately
                                         // (slow on large lattices, but needs less memory)
    std::vector<char> buf;

    template<class f1, class f2>
    GSU3<f2> from_buf(f1 *buf) const {
        int i = 0;
        GSU3<f2> U;
        for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
            f2 re = buf[i++];
            f2 im = buf[i++];
            U(j, k) = GCOMPLEX(f2)(re, im);
        }
        return U;
    }

    template<class f1, class f2>
    void to_buf(f1 *buf, const GSU3<f2> &U) const {
        int i = 0;
        for (int j = 0; j < 3; j++)
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

    IldgFormat(CommunicationBase &comm) : comm(comm), header(comm) {
        float_size = 0;
        su3_size = 0;
        switch_endian = false;
        stored_checksum_nersc = 0;
        computed_checksum_nersc = 0;
        index = 0;
    }

    bool read_header(std::istream &in) {
        rootLogger.push_verbosity(OFF);
        if (!header.read(in)) {
            rootLogger.error("header.read() failed!");
            return false;
        }
        rootLogger.pop_verbosity();

        bool error = false;
        for (int i = 0; i < 4; i++) {
            if (header.dim[i]() != GInd::getLatData().globalLattice()[i]) {
                rootLogger.error("Stored dimension ", i, " not equal to current lattice size.");
                error = true;
            }
        }

        // ILDG binaries are always saved in BIG endian
        Endianness disken = ENDIAN_BIG;
        ///                          ILDG                              QUDA
        if (header.floatingpoint() == "32" || header.floatingpoint() == "F") {
            float_size = 4;
        } else if (header.floatingpoint() == "64" || header.floatingpoint() == "D") {
            float_size = 8;
        } else {
            rootLogger.error("Unrecognized FLOATING_POINT ", header.floatingpoint());
            error = true;
        }
        switch_endian = switch_endianness(disken);

        rootLogger.info(header.checksuma);
        rootLogger.info(header.checksumb);
        rootLogger.info(header.floatingpoint);
        rootLogger.info(header.dim[0]);
        rootLogger.info(header.dim[1]);
        rootLogger.info(header.dim[2]);
        rootLogger.info(header.dim[3]);

        su3_size = 2 * 3 * 3 * float_size;
        buf.resize((sep_lines ? GInd::getLatData().lx : GInd::getLatData().vol4) * 4 * su3_size);
        index = buf.size();

        return !error;
    }


    void lime_record(std::ostream &out, bool switch_endian, std::string header_ildg, std::string data) {

        // The first 32+16=48 bits (0-47) of the header word are the magic number and the version number. 
        const int32_t magic_number   = ILDG_MAGIC_NUMBER;
        const int16_t version_number = 1;

        // Bit 48 is the flag for whether the message is beginning. Bit 49 is the flag for whether the message is ending.
        // Binary is stored from left to right, even though the least-significant digit is on the right. This storage
        // convention, which puts the most significant digit at the smallest memory address, is big endian, which is
        // the ILDG standard. The remaining 6 bits are reserved bits.
        const int8_t begin_end = 0b11000000;

        const int8_t zero_8bit = 0;

        int64_t data_length,data_length_swap;
        int data_mod, null_padding;

        if (data=="") {
            data_length=GInd::getLatData().globvol4*bytes_per_site(); // lattice volume x #(links) x link_entries(re+im) x precision
        } else {
            data_length=data.length();
        }
        data_length_swap=data_length;

        if (switch_endian){
            Byte_swap(magic_number);
            Byte_swap(version_number);
            Byte_swap(zero_8bit);
            Byte_swap(begin_end);
            Byte_swap(data_length_swap);
        }

        out.write((char *) &magic_number, sizeof(magic_number));
        out.write((char *) &version_number, sizeof(version_number));
        out.write((char *) &begin_end, sizeof(begin_end));
        out.write((char *) &zero_8bit, sizeof(zero_8bit));
        out.write((char *) &data_length_swap, sizeof(data_length_swap));

        out.write(header_ildg.c_str(), header_ildg.length());
        for(int i = header_ildg.length(); i <128; i++) {
            out.write((char *) &zero_8bit, sizeof(zero_8bit));
        }

        if (data!="") {
            out.write(data.c_str(), data_length);
            data_mod=data.length()%8;
            if (data_mod==0) null_padding=0;
            else null_padding = 8-data_mod;
            for(int i =0; i < null_padding; i++) {
                out.write((char *) &zero_8bit, sizeof(zero_8bit));
            }
        }
    }

    template<class floatT,bool onDevice, CompressionType comp>
    bool write_header(int diskprec, Checksum computed_checksum_crc32, std::ostream &out, bool head, LatticeParameters param) {

        /// In general, we will add some newlines to make the header more readable to the eye.
        std::string xmlProlog = "\n<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        std::string fp, data;

        if ( diskprec == 1 || (diskprec == 0 && sizeof(floatT) == sizeof(float)) ) {
            float_size = 4;
        } else if ( diskprec == 2 || (diskprec == 0 && sizeof(floatT) == sizeof(double)) ) {
            float_size = 8;
        } else {
            rootLogger.error("diskprec should be 0, 1 or 2.");
            return false;
        }

        su3_size = 2 * 3 * 3 * float_size;
        buf.resize((sep_lines ? GInd::getLatData().lx : GInd::getLatData().vol4) * 4 * su3_size);

        for (int mu = 0; mu < 4; mu++)
            header.dim[mu].set(GInd::getLatData().globalLattice()[mu]);

        Endianness en = ENDIAN_BIG;
        switch_endian = switch_endianness(en);

        if (float_size == 4) {
            fp = "32";
        } else if (float_size == 8) {
            fp = "64";
        } else {
            rootLogger.error("ILDG format must store single or double precision.");
            return false;
        }

        if (comm.IamRoot()) {
            if (head) {

                data = xmlProlog + "<ildgFormat xmlns=\"http://www.lqcd.org/ildg\"\n" 
                                 + "            xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n" 
                                 + "            xsi:schemaLocation=\"http://www.lqcd.org/ildg/filefmt.xsd\">\n"
                                 + "  <version>1.0</version>\n"
                                 + "  <field>su3gauge</field>\n"
                                 + "  <precision>"+fp+"</precision>\n"
                                 + "  <lx>"+std::to_string(GInd::getLatData().globLX)+"</lx>"
                                 + "  <ly>"+std::to_string(GInd::getLatData().globLY)+"</ly>"
                                 + "  <lz>"+std::to_string(GInd::getLatData().globLZ)+"</lz>"
                                 + "  <lt>"+std::to_string(GInd::getLatData().globLT)+"</lt>\n"
                                 + "</ildgFormat>\n";
                lime_record(out, switch_endian, "ildg-format", data);

                data = "";
                lime_record(out, switch_endian, "ildg-binary-data", data);

            } else {

                data = "mc://ldg/"+param.ILDGcollaboration()+"/"+param.ILDGprojectName()+"/"+param.ensembleExt()+"/ildg"+param.fileExt();
                rootLogger.info(data);
                lime_record(out, switch_endian, "ildg-data-lfn", data);

                std::stringstream crc32a, crc32b;
                crc32a << std::hex << computed_checksum_crc32.checksuma;
                crc32b << std::hex << computed_checksum_crc32.checksumb;
                rootLogger.info("checksuma (ildg): ", crc32a.str());
                rootLogger.info("checksumb (ildg): ", crc32b.str());

                data = xmlProlog + "<scidacChecksum><version>1.0</version><suma>"+crc32a.str()+"</suma><sumb>"+crc32b.str()+"</sumb></scidacChecksum>";
                lime_record(out, switch_endian, "scidac-checksum", data);
            }
        }
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

    bool checksums_match(Checksum sum) {
        std::stringstream suma, sumb;
        suma << std::hex << sum.checksuma;
        sumb << std::hex << sum.checksumb;
        if ( header.checksuma() == suma.str() && header.checksumb() == sumb.str() ) {
            rootLogger.info("Checksums match successfully!");
            return true;
        } else if ( header.checksuma().empty() && header.checksumb().empty() ) {
            // Technically a checksum is not part of the ILDG specification. Therefore our code must be able to read
            // ILDG configurations that have no checksum metadata.
            rootLogger.warn("Couldn't find checksums for read-in ILDG binary.");
            return true;
        } else {
            rootLogger.info("Checksuma: ",std::hex, header.checksuma()," (Stored) != ", sum.checksuma," (Computed)");
            rootLogger.info("Checksumb: ",std::hex, header.checksumb()," (Stored) != ", sum.checksumb," (Computed)");
            throw std::runtime_error(stdLogger.fatal("Checksum mismatch!"));
            return false;
        }
    }

    void byte_swap_sitedata(char *sitedata, int n) {
        for (size_t bs = 0; bs < 72; bs++)
            Byte_swap(sitedata + bs * n, n);
    }

    int precision_read() {
        int precision;
        if (header.floatingpoint() == "32" || header.floatingpoint() == "F") {
            precision = 1;
        } else if (header.floatingpoint() == "64" || header.floatingpoint() == "D") {
            precision = 2;
        } else {
            throw std::runtime_error(stdLogger.fatal("FLOATING_POINT = ", header.floatingpoint(), "not recognized."));
        }
        return precision;
    }
};
