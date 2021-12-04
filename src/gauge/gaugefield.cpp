//
// Created by Lukas Mazur on 13.09.18.
//
#include "gaugefield.h"
#include "../base/IO/nersc.h"
#include "../base/IO/milc.h"
#include <fstream>

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::writeconf_nersc(const std::string &fname, int rows,
                                                                             int diskprec, Endianness en) {
    if(onDevice){

        rootLogger.info("writeconf_nersc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host((int)GInd::getLatData().vol4Full*4);
        lattice_host.copyFrom(_lattice);
        writeconf_nersc_host(lattice_host.getAccessor(),fname,rows,diskprec,en);
    }
    else{
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

    {
        std::ofstream out;
        if (this->getComm().IamRoot())
            out.open(fname.c_str());
        if (!nersc.template write_header<floatT, onDevice, comp>(*this, gaugeAccessor, rows, diskprec, en, out)) {
            rootLogger.error("Unable to write NERSC file: " ,  fname);
            return;
        }
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
void Gaugefield<floatT, onDevice, HaloDepth,comp>::readconf_nersc(const std::string &fname) {

    if(onDevice){

        rootLogger.info("readconf_nersc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host(GInd::getLatData().vol4Full*4);
        readconf_nersc_host(lattice_host.getAccessor(),fname);
        _lattice.copyFrom(lattice_host);
    }
    else{
        readconf_nersc_host(getAccessor(),fname);
    }

    this->su3latunitarize();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::readconf_milc(const std::string &fname) {

    if(onDevice){

        rootLogger.info("readconf_milc: Create temporary GSU3Array!");
        GSU3array<floatT, false, comp>  lattice_host(GInd::getLatData().vol4Full*4);
        readconf_milc_host(lattice_host.getAccessor(),fname);
        _lattice.copyFrom(lattice_host);
    }
    else{
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

    {
        std::ifstream in;
        if (this->getComm().IamRoot())
            in.open(fname.c_str());
        if (!nersc.read_header(in)){
            throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str());
        }
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

    if (!nersc.checksums_match()){
        throw std::runtime_error(stdLogger.fatal("Error checksum!");
    }

}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::readconf_milc_host(gaugeAccessor<floatT,comp> gaugeAccessor,
                                                                        const std::string &fname)
{
    MilcFormat<HaloDepth> milc(this->getComm());
    typedef GIndexer<All,HaloDepth> GInd;

    {
        std::ifstream in;
        if (this->getComm().IamRoot())
            in.open(fname.c_str());
        if (!milc.read_header()){
            throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str());
        }
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


    if (!milc.checksums_match()){
//        throw std::runtime_error(stdLogger.fatal("Error checksum!");
    }

}



#define _GLATTICE_CLASS_INIT(floatT, onDevice, HaloDepth,COMP) \
template class Gaugefield<floatT,onDevice,HaloDepth, COMP>; \

#define INIT(floatT,HALO,COMP) \
_GLATTICE_CLASS_INIT(floatT, 0,HALO,COMP)  \
_GLATTICE_CLASS_INIT(floatT, 1,HALO,COMP)

INIT_PHC(INIT)

