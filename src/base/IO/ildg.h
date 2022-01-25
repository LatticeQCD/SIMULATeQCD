//
// Created by Rasmus Larsen and sajid Ali on 12/13/21.
//

#ifndef SIMULATEQCD_ILDG_H
#define SIMULATEQCD_ILDG_H

#include "parameterManagement.h"
#include "misc.h"
#include "../../gauge/gaugefield.h"
#include "../../gauge/GaugeAction.h"
#include "../LatticeContainer.h"
#include "../math/gaugeAccessor.h"
#include "checksum.h"
#include <iostream>
#include <time.h>

template<class floatT>
floatT returnEndian(floatT input,bool change){
    if(change){
        if(sizeof(input) == 8){
            return __builtin_bswap64(input);
        }
        else if(sizeof(input) == 4){
            return __builtin_bswap32(input);
        }
    }
    else{
        return input;
    }
}

//ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
//ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

class IldgHeader : virtual private ParameterList {
private:
    const CommunicationBase &comm;
    int header_size;

    Parameter<std::string> dattype;
    Parameter<int> dim[4];
    Parameter<std::string> checksuma;
    Parameter<std::string> checksumb;
    Parameter<std::string> floatingpoint;
//////////////////////////////
    Parameter<double> linktrace;
    Parameter<double> plaq;
///////////////////////////////

    bool read_info(std::istream &in) {
        int32_t magic_number;
        int64_t data_length;
        int dimx, dimy, dimz, dimt;
        int dataPos;
        std::string precision, dataType, suma, sumb, typesize, datacount;
        bool Endian=false;

        in.read(reinterpret_cast<char *>(&magic_number),sizeof(magic_number));
        if(magic_number != 1164413355){
            if(__builtin_bswap32(magic_number) == 1164413355){
                Endian = true;
            }
            else{
                rootLogger.info("could not read magic number");
                return false;
            }
        }
        rootLogger.info("magic_number = ",magic_number);

        in.clear();
        in.seekg(0);

        while(in.read(reinterpret_cast<char *>(&magic_number),sizeof(magic_number))){

            if(returnEndian(magic_number,Endian) == 1164413355){
                in.ignore(4);
                in.read(reinterpret_cast<char *>(&data_length),sizeof(data_length));

                if(returnEndian(data_length,Endian) > 100000){
                    in.ignore(8*16);
                    dataPos = in.tellg();
                    in.ignore(returnEndian(data_length,Endian));
                }
                else{
                    int bytes = ceil(returnEndian(data_length,Endian)/8.0)*8;

                    in.ignore(8*16);
                    char info[bytes];
                    in.read(info,sizeof(info));
                    std::string myString(info, bytes);

                    if(myString.find("<dims>") != std::string::npos){
                        int pos = myString.find("<dims>");
                        int posEnd = myString.find("</dims>");
                        std::string part;
                        std::stringstream content(myString.substr(pos+6,posEnd-pos-6));
                        std::getline(content,part,' ');
                        dimx = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimy = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimz = atoi(part.c_str());
                        std::getline(content,part,' ');
                        dimt = atoi(part.c_str());
                    }
                    if(myString.find("<datatype>") != std::string::npos){
                        int pos = myString.find("<datatype>");
                        int posEnd = myString.find("</datatype>");
                        dataType = myString.substr(pos+10,posEnd-10-pos);
                    }

                    if(myString.find("<precision>") != std::string::npos){
                        int pos = myString.find("<precision>");
                        int posEnd = myString.find("</precision>");
                        precision = myString.substr(pos+11,posEnd-11-pos);
                    }

                    if(myString.find("<suma>") != std::string::npos){
                        int pos = myString.find("<suma>");
                        int posEnd = myString.find("</suma>");
                        suma = myString.substr(pos+6,posEnd-6-pos);
                    }

                    if(myString.find("<sumb>") != std::string::npos){
                        int pos = myString.find("<sumb>");
                        int posEnd = myString.find("</sumb>");
                        sumb = myString.substr(pos+6,posEnd-6-pos);
                    }

                    if(myString.find("<typesize>") != std::string::npos){
                        int pos = myString.find("<typesize>");
                        int posEnd = myString.find("</typesize>");
                        typesize = myString.substr(pos+10,posEnd-10-pos);
                    }

                    if(myString.find("<datacount>") != std::string::npos){
                        int pos = myString.find("<datacount>");
                        int posEnd = myString.find("</datacount>");
                        datacount = myString.substr(pos+11,posEnd-11-pos);
                    }
                }
            }
        }

        dattype.set(typesize);
        dim[0].set(dimx);
        dim[1].set(dimy);
        dim[2].set(dimz);
        dim[3].set(dimt);
        checksuma.set(suma);
        checksumb.set(sumb);
        floatingpoint.set(precision);
        header_size=dataPos;
        //std::cout<<"header_size "<<header_size<<std::endl;

        in.clear();
        in.seekg(0);

        return true;

    }
    IldgHeader(const CommunicationBase &_comm) : comm(_comm) {
        header_size = 0;
        add(dattype, "DATATYPE");
        add(dim[0], "DIMENSION_1");
        add(dim[1], "DIMENSION_2");
        add(dim[2], "DIMENSION_3");
        add(dim[3], "DIMENSION_4");
        add(checksuma, "CHECKSUMA");
        add(checksumb, "CHECKSUMB");
        add(linktrace, "LINK_TRACE");
        add(plaq, "PLAQUETTE");
        addDefault(floatingpoint, "FLOATING_POINT", std::string("IEEE32BIG"));
        //addDefault(floatingpoint, "FLOATING_POINT", std::string("IEEE32LITTLE"));
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
        if (this->comm.IamRoot()){
            success = read_info(in);
        }

        if (!comm.single()) {
            comm.root2all(success);
            if (success) {
                int dim0, dim1, dim2, dim3;
                std::string checka, checkb, dtype, precision;
                checka = checksuma();
                checkb = checksumb();
                dtype = dattype();
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
                comm.root2all(dtype);
        		comm.root2all(precision);
                dim[0].set(dim0);
                dim[1].set(dim1);
                dim[2].set(dim2);
                dim[3].set(dim3);
                checksuma.set(checka);
                checksumb.set(checkb);
                dattype.set(dtype);
                floatingpoint.set(precision);
            }
        }
        if (!success)
            return false;
        return true;
    }
////////////////////////////
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
//ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
//ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
template<size_t HaloDepth>
class IldgFormat {
private:

    const CommunicationBase &comm;
    IldgHeader header;
    typedef GIndexer<All,HaloDepth> GInd;
    int rows;
    int float_size;
    bool switch_endian;
    uint32_t stored_checksum_nersc, computed_checksum_nersc;
    int su3_size;
    size_t index; //position in buffer
    static const bool sep_lines = false; // make the buffer smaller and
    // read each xline separately
    // (slow on large lattices, but
    // needs less memory)
    std::vector<char> buf;

    //void from_buf(f1 *buf, GSU3<f2> &U) const {
    template<class f1, class f2>
    GSU3<f2> from_buf(f1 *buf) const {
        int i = 0;
        GSU3<f2> U;
        for (int j = 0; j < rows; j++)
            for (int k = 0; k < 3; k++) {
                f2 re = buf[i++];
                f2 im = buf[i++];
                //U.set(j, k, complex<f2>(re, im));
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

    IldgFormat(CommunicationBase &comm)
            :comm(comm), header(comm) {
        rows = 3;
        float_size = 0;
        su3_size = 0;
        switch_endian = false;
        stored_checksum_nersc = 0;
        computed_checksum_nersc = 0;
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
                rootLogger.error("Stored dimension ", i, " not equal to current lattice size.");
                error = true;
            }
        //su3_size=2*3*rows*float_size, float_size=4 or 8.
        if (header.dattype() == "48"){
            float_size=4;
            rows=2;
        } else if (header.dattype() == "96"){
            float_size=8;
            rows=2;
        } else if (header.dattype() == "72"){
            float_size=4;
            rows=3;
        } else if (header.dattype() == "144"){
            float_size=8;
            rows=3;
        } else {
            rootLogger.error("DATATYPE = ", header.dattype(), "not recognized.");
            error = true;
        }

        Endianness disken = ENDIAN_AUTO;
        if (header.floatingpoint() == "IEEE32BIG" || header.floatingpoint() == "IEEE32") {
            //float_size = 4;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE64BIG") {
            //float_size = 8;
            disken = ENDIAN_BIG;
        } else if (header.floatingpoint() == "IEEE32LITTLE") {
            //float_size = 4;
            disken = ENDIAN_LITTLE;
        } else if (header.floatingpoint() == "IEEE64LITTLE") {
            //float_size = 8;
            disken = ENDIAN_LITTLE;
        } /*else {
            rootLogger.error("Unrecognized FLOATING_POINT ", header.floatingpoint());
            error = true;
        }*/
        switch_endian = switch_endianness(disken);

        rootLogger.info(header.checksuma);
        rootLogger.info(header.checksumb);
        rootLogger.info(header.dattype);
        rootLogger.info(header.floatingpoint);
        rootLogger.info(header.dim[0]);
        rootLogger.info(header.dim[1]);
        rootLogger.info(header.dim[2]);
        rootLogger.info(header.dim[3]);

        /*s >> std::hex >> stored_checksum_nersc;
        if (s.fail()) {
            rootLogger.error("Could not interpret checksum ", header.checksum(), "as hexadecimal number.");
            error = true;
        }*/

        su3_size = 2 * 3 * rows * float_size;
        buf.resize((sep_lines ? GInd::getLatData().lx : GInd::getLatData().vol4) * 4 * su3_size);
        index = buf.size();

        return !error;
    }

    //##########################################################################################

    void lime_record(std::ostream &out, bool switch_endian, std::string header_ildg, std::string data, int su3_size){
        int32_t magic_number = 1164413355;
        int16_t version_number = 1;
        int8_t zero_8bit=0;
        int8_t begin_end_reserved_8bit=0b10000000;

        int64_t data_length,data_length_swap;
        int data_mod, null_padding;

        if (data==""){

            data_length=GInd::getLatData().globvol4*bytes_per_site();//lattice volume x #(links) x link_enteries(re+im) x precision
        }
        else{
            data_length=data.length();
        }
        data_length_swap=data_length;

        if (switch_endian){
            Byte_swap(magic_number);
            Byte_swap(version_number);
            Byte_swap(zero_8bit);
            Byte_swap(begin_end_reserved_8bit);
            Byte_swap(data_length_swap);
        }

        out.write((char *) &magic_number, sizeof(magic_number));
        out.write((char *) &version_number, sizeof(version_number));
        out.write((char *) &begin_end_reserved_8bit, sizeof(begin_end_reserved_8bit));
        out.write((char *) &zero_8bit, sizeof(zero_8bit));
        out.write((char *) &data_length_swap, sizeof(data_length_swap));

        out.write(header_ildg.c_str(), header_ildg.length());
        for(int i =header_ildg.length();i <128; i++){
            out.write((char *) &zero_8bit, sizeof(zero_8bit));
        }

        if (data!=""){
            out.write(data.c_str(), data_length);
            data_mod=data.length()%8;
            if (data_mod==0) null_padding=0;
            else null_padding = 8-data_mod;
            for(int i =0;i < null_padding; i++){
                out.write((char *) &zero_8bit, sizeof(zero_8bit));
            }
        }
    }

    template<class floatT,bool onDevice, CompressionType comp>
    bool write_header(Gaugefield<floatT, onDevice, HaloDepth,comp> &gf, gaugeAccessor<floatT,comp> gaugeAccessor, int _rows,
                      int diskprec, Endianness en, Checksum computed_checksum_crc32, std::ostream &out, bool head) {
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

        for (int mu = 0; mu < 4; mu++)
            header.dim[mu].set(GInd::getLatData().globalLattice()[mu]);

        if (rows == 2)
            header.dattype.set("48");
        else if (rows == 3)
            if (float_size == 4)
                header.dattype.set("72");
            else if (float_size == 8)
                header.dattype.set("144");
            else {
                rootLogger.error("ILDG format must have a single or double precision.");
                return false;

            }
        else {
            rootLogger.error("ILDG format must store 2 or 3 rows.");
            return false;
        }

        if (en == ENDIAN_AUTO) en = get_endianness(false);
        switch_endian = switch_endianness(en);

        std::string fp;
        if (float_size == 4)
            //fp = "F";
            fp = "IEEE32BIG";
        else if (float_size == 8)
            //fp = "D";
            fp = "IEEE64BIG";
        else {
            rootLogger.error("ILDG format must store single or double precision.");
            return false;
        }
        if (comm.IamRoot()) {

            std::string header_ildg, data, dt;
            time_t current_time = time(0);
            struct tm *tm = localtime(&current_time);
            dt = asctime(tm);

            if (head) {

                // first lime record (header)
                header_ildg = "scidac-private-file-xml";
                data = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><scidacFile><version>1.1</version><spacetime>4</spacetime><dims>"
                       + std::to_string(GInd::getLatData().globLX)
                       + " " + std::to_string(GInd::getLatData().globLY)
                       + " " + std::to_string(GInd::getLatData().globLZ)
                       + " " + std::to_string(GInd::getLatData().globLT)
                       + " </dims><volfmt>0</volfmt></scidacFile>";
                lime_record(out, switch_endian, header_ildg, data, su3_size);

                // second lime record (header)
                header_ildg = "scidac-file-xml";
                data = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><title>ILDG archival gauge configuration</title>";
                lime_record(out, switch_endian, header_ildg, data, su3_size);

                // third lime record (header)
                header_ildg = "scidac-private-record-xml";
                data = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><scidacRecord><version>1.1</version><date>" +
                       dt + "</date><recordtype>0</recordtype><datatype>";
                data += "USQCD_F3_ColorMatrix</datatype><precision>" + fp +
                        "</precision><colors>3</colors><typesize>"
                        + header.dattype() + "</typesize><datacount>4</datacount></scidacRecord>";
                lime_record(out, switch_endian, header_ildg, data, su3_size);

                // fourth lime record (header)
                header_ildg = "scidac-record-xml";
                data = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><title>Dummy QCDML</title>";
                lime_record(out, switch_endian, header_ildg, data, su3_size);

                // fifth lime record (binary data)
                header_ildg = "scidac-binary-data";
                data = "";
                lime_record(out, switch_endian, header_ildg, data, su3_size);
            } else {
                std::stringstream crc32a, crc32b;
                crc32a<<std::hex<<computed_checksum_crc32.checksuma;
                crc32b<<std::hex<<computed_checksum_crc32.checksumb;
                rootLogger.info("checksuma (ildg): ", crc32a.str());
                rootLogger.info("checksumb (ildg): ", crc32b.str());

                // sixth lime record (tail)
                header_ildg = "scidac-checksum";
                data = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><scidacChecksum><version>1.0</version><suma>"+crc32a.str()+"</suma><sumb>"+crc32b.str()+"</sumb></scidacChecksum>";
                lime_record(out, switch_endian, header_ildg, data, su3_size);
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

    //void get(GSU3<floatT> &U) {
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
        std::stringstream suma,sumb;
        suma<<std::hex<<sum.checksuma;
        sumb<<std::hex<<sum.checksumb;
        if (header.checksuma() == suma.str() && header.checksumb() == sumb.str()) {
            rootLogger.info("Checksuma: ",std::hex, header.checksuma()," (Stored) = ", sum.checksuma," (Computed)");
            rootLogger.info("Checksumb: ",std::hex, header.checksumb()," (Stored) = ", sum.checksumb," (Computed)");
            rootLogger.info(CoutColors::green, "Checksums match successfully!", CoutColors::reset);
            return true;
        } else{
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
        if (header.dattype() == "48" || header.dattype() == "72"){
            precision=1;
        } else if (header.dattype() == "96" || header.dattype() == "144"){
            precision=2;
        } else {
            rootLogger.error("DATATYPE = ", header.dattype(), "not recognized.");
        }
        return precision;
    }
};

#endif //SIMULATEQCD_ILDG_H
