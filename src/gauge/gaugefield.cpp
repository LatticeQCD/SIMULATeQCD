/* 
 * gaugefield.cpp                                                               
 * 
 * L. Mazur 
 * 
 */

#include "gaugefield.h"
#include "../base/IO/nersc.h"
#include "../base/IO/milc.h"
#include "../base/IO/ildg.h"
#include <fstream>

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::writeconf_nersc(const std::string &fname, int rows,
                                                                             int diskprec, Endianness en) {
    if(onDevice) {
        rootLogger.info("writeconf_nersc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host((int)GInd::getLatData().vol4Full*4);
        lattice_host.copyFrom(_lattice);
        writeconf_nersc_host(lattice_host.getAccessor(),fname,rows,diskprec,en);
    } else {
        writeconf_nersc_host(getAccessor(),fname,rows,diskprec,en);
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::writeconf_nersc_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                        const std::string &fname, int rows,
                                                                        int diskprec, Endianness en)
{
    typedef GIndexer<All,HaloDepth> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    NerscFormat<HaloDepth> nersc(this->getComm());

    std::ofstream out;
    if (this->getComm().IamRoot())
        out.open(fname.c_str());
    if (!nersc.template write_header<floatT, onDevice, comp>(*this, gaugeAccessor, rows, diskprec, en, out)) {
        rootLogger.error("Unable to write NERSC file: " ,  fname);
        return;
    }

    rootLogger.info("Writing NERSC file format: " ,  fname);

    const size_t filesize = nersc.header_size() + GInd::getLatData().globalLattice().mult() * nersc.bytes_per_site();

    this->getComm().initIOBinary(fname, filesize, nersc.bytes_per_site(), nersc.header_size(), global, local, WRITE);
    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                    for (size_t mu = 0; mu < 4; mu++) {
                        gSite site = GInd::getSite(x, y, z, t);
                        GSU3<floatT> tmp = gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
                        nersc.put(tmp);
                    }
                    if (nersc.end_of_buffer()) {
                        nersc.process_write_data();
                        this->getComm().writeBinary(nersc.buf_ptr(), nersc.buf_size() / nersc.bytes_per_site());
                    }
                }
    this->getComm().closeIOBinary();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::writeconf_ildg(const std::string &fname, int rows,
                                                                   int diskprec, Endianness en) {
    if(onDevice) {
//        rootLogger.info("writeconf_ildg: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host((int)GInd::getLatData().vol4Full*4);
        lattice_host.copyFrom(_lattice);
        writeconf_ildg_host(lattice_host.getAccessor(),fname,rows,diskprec,en);
    } else {
        writeconf_ildg_host(getAccessor(),fname,rows,diskprec,en);
    }
}

template<class floatT,bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::writeconf_ildg_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                       const std::string &fname, int rows,
                                                                       int diskprec, Endianness en)
{
    Checksum crc32;
    InitializeChecksum(&crc32);
    //dynamically allocate memory
    gMemoryPtr<false> crc32_array = MemoryManagement::getMemAt<false>("crc32_array");
    crc32_array->template adjustSize<uint32_t>(GInd::getLatData().globvol4);
    crc32_array->memset(0);
    uint32_t *crc32_array_ptr=crc32_array->template getPointer<uint32_t>();

    typedef GIndexer<All,HaloDepth> GInd;
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    IldgFormat<HaloDepth> ildg(this->getComm());

    std::ofstream out;
    if (this->getComm().IamRoot())
        out.open(fname.c_str(),std::ios::out | std::ios::binary);
    //changed open to binary
    if (!ildg.template write_header<floatT, onDevice, comp>(rows, diskprec, en, crc32, out,true)) {
        rootLogger.error("Unable to write ILDG file: ", fname);
        return;
    }

    rootLogger.info("Writing ILDG file format: ", fname);
    const size_t filesize = ildg.header_size() + GInd::getLatData().globalLattice().mult() * ildg.bytes_per_site();
    this->getComm().initIOBinary(fname, filesize, ildg.bytes_per_site(), ildg.header_size(), global, local, WRITE);
    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                    gSite site = GInd::getSite(x, y, z, t);
                    LatticeDimensions localCoord(site.coord.x, site.coord.y, site.coord.z, site.coord.t);
                    size_t index = GInd::localCoordToGlobalIndex(localCoord);

                    if(diskprec == 1 || (diskprec == 0 && sizeof(floatT) == sizeof(float))){
                        std::vector <GSU3<float>> sitedata;
                        for (size_t mu = 0; mu < 4; mu++) {
                            GSU3 <float> tmp = gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
                            sitedata.push_back(tmp);
                            ildg.put(tmp);
                        }
                        ildg.byte_swap_sitedata((char *) (&sitedata[0]), sizeof(float));
                        crc32_array_ptr[index] = checksum_crc32_sitedata((char *) (&sitedata[0]), ildg.bytes_per_site());
                    } else if (diskprec == 2 || (diskprec == 0 && sizeof(floatT) == sizeof(double))) {
                        std::vector <GSU3<double>> sitedata;
                        for (size_t mu = 0; mu < 4; mu++) {
                            GSU3 <double> tmp = gaugeAccessor.getLink(GInd::getSiteMu(site, mu));
                            sitedata.push_back(tmp);
                            ildg.put(tmp);
                        }
                        ildg.byte_swap_sitedata((char *) (&sitedata[0]), sizeof(double));
                        crc32_array_ptr[index] = checksum_crc32_sitedata((char *) (&sitedata[0]), ildg.bytes_per_site());
                    } else {
                        rootLogger.error("diskprec should be 0, 1 or 2.");
                    }

                    if (ildg.end_of_buffer()) {
                        ildg.process_write_data();
                        this->getComm().writeBinary(ildg.buf_ptr(), ildg.buf_size() / ildg.bytes_per_site());
                    }
                }

    this->getComm().reduce(crc32_array_ptr, GInd::getLatData().globvol4);
    if(this->getComm().IamRoot()) {
        checksum_crc32_combine(&crc32, GInd::getLatData().globvol4, crc32_array_ptr);
    }

    this->getComm().closeIOBinary();

    if (this->getComm().IamRoot())
        out.open(fname.c_str(),std::ios::app | std::ios::binary);
    if (!ildg.template write_header<floatT, onDevice, comp>(rows, diskprec, en, crc32, out,false)) {
        rootLogger.error("Unable to write ILDG file: ", fname);
        return;
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::readconf_nersc(const std::string &fname) {

    if(onDevice) {
        rootLogger.info("readconf_nersc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host(GInd::getLatData().vol4Full*4);
        readconf_nersc_host(lattice_host.getAccessor(),fname);
        _lattice.copyFrom(lattice_host);
    } else {
        readconf_nersc_host(getAccessor(),fname);
    }
    this->su3latunitarize();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::readconf_ildg(const std::string &fname) {

    if(onDevice) {

        rootLogger.info("readconf_ildg: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host(GInd::getLatData().vol4Full*4);
        readconf_ildg_host(lattice_host.getAccessor(),fname);
        _lattice.copyFrom(lattice_host);
    } else {
        readconf_ildg_host(getAccessor(),fname);
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::readconf_milc(const std::string &fname) {

    if(onDevice) {
        rootLogger.info("readconf_milc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host(GInd::getLatData().vol4Full*4);
        readconf_milc_host(lattice_host.getAccessor(),fname);
        _lattice.copyFrom(lattice_host);
    } else {
        readconf_milc_host(getAccessor(),fname);
    }
    this->su3latunitarize();
}


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::readconf_nersc_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                        const std::string &fname)
{
    NerscFormat<HaloDepth> nersc(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;
    if (this->getComm().IamRoot()) {
        in.open(fname.c_str());
    }
    if (!nersc.read_header(in)) {
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    this->getComm().initIOBinary(fname, 0, nersc.bytes_per_site(), nersc.header_size(), global, local, READ);

    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                    if (nersc.end_of_buffer()) {
                        this->getComm().readBinary(nersc.buf_ptr(), nersc.buf_size() / nersc.bytes_per_site());
                        nersc.process_read_data();
                    }
                    for (int mu = 0; mu < 4; mu++) {
                        GSU3<floatT> ret = nersc.template get<floatT>();
                        gSite site = GInd::getSite(x, y, z, t);
                        gaugeAccessor.setLink(GInd::getSiteMu(site, mu), ret);
                    }
                }

    this->getComm().closeIOBinary();

    if (!nersc.checksums_match()) {
        throw std::runtime_error(stdLogger.fatal("Error checksum!"));
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::readconf_ildg_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                       const std::string &fname)
{
    IldgFormat<HaloDepth> ildg(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;
    if (this->getComm().IamRoot())
        in.open(fname.c_str(), std::ios::in | std::ios::binary);
    if (!ildg.read_header(in)){
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    Checksum read_crc32;
    InitializeChecksum(&read_crc32);
    gMemoryPtr<false> crc32_array_read = MemoryManagement::getMemAt<false>("crc32_array_read");
    crc32_array_read->template adjustSize<uint32_t>(GInd::getLatData().globvol4);
    crc32_array_read->memset(0);
    uint32_t *crc32_array_read_ptr=crc32_array_read->template getPointer<uint32_t>();
    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    this->getComm().initIOBinary(fname, 0, ildg.bytes_per_site(), ildg.header_size(), global, local, READ);
    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                    gSite site = GInd::getSite(x, y, z, t);
                    LatticeDimensions localCoord(site.coord.x, site.coord.y, site.coord.z, site.coord.t);
                    size_t index = GInd::localCoordToGlobalIndex(localCoord);

                    if (ildg.end_of_buffer()) {
                        this->getComm().readBinary(ildg.buf_ptr(), ildg.buf_size() / ildg.bytes_per_site());
                        ildg.process_read_data();
                    }

                    if (ildg.precision_read() == 1 || (ildg.precision_read() == 0 && sizeof(floatT) == sizeof(float))) {
                        std::vector <GSU3<float>> sitedata;
                        for (int mu = 0; mu < 4; mu++) {
                            GSU3<float> ret = ildg.template get<float>();
                            sitedata.push_back(ret);
                            gaugeAccessor.setLink(GInd::getSiteMu(site, mu), ret);
                        }
                        ildg.byte_swap_sitedata((char *) (&sitedata[0]), sizeof(float));
                        crc32_array_read_ptr[index] = checksum_crc32_sitedata((char *) (&sitedata[0]),
                                                                              ildg.bytes_per_site());;
                    } else if (ildg.precision_read() == 2 || (ildg.precision_read() == 0 && sizeof(floatT) == sizeof(double))) {
                        std::vector <GSU3<double>> sitedata;
                        for (int mu = 0; mu < 4; mu++) {
                            GSU3<double> ret = ildg.template get<double>();
                            sitedata.push_back(ret);
                            gaugeAccessor.setLink(GInd::getSiteMu(site, mu), ret);
                        }
                        ildg.byte_swap_sitedata((char *) (&sitedata[0]), sizeof(double));
                        crc32_array_read_ptr[index] = checksum_crc32_sitedata((char *) (&sitedata[0]),
                                                                              ildg.bytes_per_site());
                    } else {
                        rootLogger.error("diskprec should be 0, 1 or 2.");
                    }
                }
    this->getComm().reduce(crc32_array_read_ptr, GInd::getLatData().globvol4);
    if(this->getComm().IamRoot()) {
        checksum_crc32_combine(&read_crc32, GInd::getLatData().globvol4, crc32_array_read_ptr);
        ildg.checksums_match(read_crc32);
    }
    this->getComm().closeIOBinary();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::readconf_milc_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                        const std::string &fname)
{
    MilcFormat<HaloDepth> milc(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    std::ifstream in;
    if (this->getComm().IamRoot())
        in.open(fname.c_str());
    if (!milc.read_header()){
        throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    this->getComm().initIOBinary(fname, 0, milc.bytes_per_site(), milc.header_size(), global, local, READ);

    floatT traceSum = 0.0;

    typedef GIndexer<All, HaloDepth> GInd;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {
                    if (milc.end_of_buffer()) {
                        this->getComm().readBinary(milc.buf_ptr(), milc.buf_size() / milc.bytes_per_site());
                        milc.template get_endian<floatT>();
                        milc.process_read_data();
                    }
                    for (int mu = 0; mu < 4; mu++) {
                        GSU3<floatT> ret = milc.template get<floatT>();
                        gSite site = GInd::getSite(x, y, z, t);
                        gaugeAccessor.setLink(GInd::getSiteMu(site, mu), ret);
         
                        traceSum += tr_d(ret);
                    }
                }


    this->getComm().closeIOBinary();

    traceSum = ((this->getComm()).reduce(traceSum)) / (3  * 4 * GInd::getLatData().globalLattice().mult());

    rootLogger.info("Trace sum is = " ,  traceSum);

    if (!milc.checksums_match()) {
        throw std::runtime_error(stdLogger.fatal("Error checksum!"));
    }
}



#define _GLATTICE_CLASS_INIT(floatT, onDevice, HaloDepth,COMP) \
template class Gaugefield<floatT,onDevice,HaloDepth, COMP>; \

#define INIT(floatT,HALO,COMP) \
_GLATTICE_CLASS_INIT(floatT, 0,HALO,COMP)  \
_GLATTICE_CLASS_INIT(floatT, 1,HALO,COMP)

INIT_PHC(INIT)

