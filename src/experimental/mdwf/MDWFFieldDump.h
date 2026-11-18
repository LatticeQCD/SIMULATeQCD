/*
 * Raw, bit-exact dumps of gauge-like fields (U, P, forces: 4 x 3x3 complex per site) and even-site MDWF spinors
 * (Ls x 12 complex per even site) for the deflation study (DESIGN.md in SQCDruns/deflation, Step 0).
 *
 * Format: one file per rank, <base>.r<rank>, native byte order, written from a HOST copy of the field:
 *   header  MDWFFieldDumpHeader (fixed-size, all int64_t)
 *   body    per local bulk site: int32 global x y z t, then doubles_per_site doubles
 *           gauge:  mu = 0..3, rows i = 0..2, columns j = 0..2, (re, im)          (72 doubles)
 *           spinor: stack s = 0..Ls-1, component c = 0..11, (re, im)             (24 Ls doubles)
 * The reader opens every rank file of the writer and keeps the sites that are local to the reading rank, so the
 * reading program may use any node decomposition. Doubles are written and read unchanged (no conversion), so a
 * write/read round trip reproduces the device field bit for bit (mdwfDeflationStep0Test). Every rank writes
 * <file>.part and renames it when complete, so a reader never sees a half-written file under the final name.
 * The reader overwrites the bulk only: update the halo of the field afterwards.
 */

#pragma once

#include "MDWFSpinor.h"

#include "../../gauge/gaugefield.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MDWFFieldDumpHeader {
    char magic[8];          // "MDWFDMP1"
    int64_t kind;           // 0 = gauge-like (4 links per site), 1 = even-site spinor
    int64_t globL[4];
    int64_t localL[4];
    int64_t gPos[4];
    int64_t nranks;
    int64_t rank;
    int64_t stacks;         // Ls for spinors, 0 for gauge fields
    int64_t nsites;         // local bulk sites written
    int64_t doublesPerSite;
};

inline std::string mdwfFieldDumpRankFile(const std::string &base, int rank) {
    return base + ".r" + std::to_string(rank);
}

template<size_t HaloDepth>
inline MDWFFieldDumpHeader mdwfFieldDumpHeader(CommunicationBase &comm, int64_t kind, int64_t stacks, int64_t nsites,
                                               int64_t doublesPerSite) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    MDWFFieldDumpHeader h{};
    std::memcpy(h.magic, "MDWFDMP1", 8);
    h.kind = kind;
    h.globL[0] = lat.globLX; h.globL[1] = lat.globLY; h.globL[2] = lat.globLZ; h.globL[3] = lat.globLT;
    h.localL[0] = lat.lx; h.localL[1] = lat.ly; h.localL[2] = lat.lz; h.localL[3] = lat.lt;
    h.gPos[0] = lat.gPosX; h.gPos[1] = lat.gPosY; h.gPos[2] = lat.gPosZ; h.gPos[3] = lat.gPosT;
    h.nranks = comm.getNumberProcesses();
    h.rank = comm.MyRank();
    h.stacks = stacks;
    h.nsites = nsites;
    h.doublesPerSite = doublesPerSite;
    return h;
}

template<size_t HaloDepth>
inline void mdwfFieldDumpCheckHeader(const MDWFFieldDumpHeader &h, int64_t kind, int64_t stacks, int64_t doublesPerSite,
                                     const std::string &file) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const bool ok = std::memcmp(h.magic, "MDWFDMP1", 8) == 0 && h.kind == kind && h.stacks == stacks
                    && h.doublesPerSite == doublesPerSite && h.globL[0] == static_cast<int64_t>(lat.globLX)
                    && h.globL[1] == static_cast<int64_t>(lat.globLY) && h.globL[2] == static_cast<int64_t>(lat.globLZ)
                    && h.globL[3] == static_cast<int64_t>(lat.globLT);
    if (!ok) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump ", file, ": header mismatch (kind ", h.kind, " vs ",
                                                 kind, ", stacks ", h.stacks, " vs ", stacks, ", global lattice ",
                                                 h.globL[0], " ", h.globL[1], " ", h.globL[2], " ", h.globL[3], ")"));
    }
}

// Local coordinates of a global site on this rank, or false.
template<size_t HaloDepth>
inline bool mdwfFieldDumpLocal(const int32_t g[4], int &x, int &y, int &z, int &t) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const long gp[4] = {static_cast<long>(lat.gPosX), static_cast<long>(lat.gPosY), static_cast<long>(lat.gPosZ),
                        static_cast<long>(lat.gPosT)};
    const long ll[4] = {static_cast<long>(lat.lx), static_cast<long>(lat.ly), static_cast<long>(lat.lz),
                        static_cast<long>(lat.lt)};
    long l[4];
    for (int d = 0; d < 4; d++) {
        l[d] = static_cast<long>(g[d]) - gp[d];
        if (l[d] < 0 || l[d] >= ll[d]) {
            return false;
        }
    }
    x = static_cast<int>(l[0]); y = static_cast<int>(l[1]); z = static_cast<int>(l[2]); t = static_cast<int>(l[3]);
    return true;
}

// ---------------------------------------------------------------------------------------------------- gauge-like

template<size_t HaloDepth>
void mdwfDumpGaugeHost(CommunicationBase &comm, const Gaugefield<double, false, HaloDepth, R18> &host,
                       const std::string &base) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const std::string file = mdwfFieldDumpRankFile(base, comm.MyRank());
    std::ofstream out(file + ".part", std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump: cannot open ", file, ".part for writing"));
    }
    const MDWFFieldDumpHeader h = mdwfFieldDumpHeader<HaloDepth>(comm, 0, 0, lat.vol4, 72);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    const SU3Accessor<double, R18> acc = host.getAccessor();
    std::vector<char> record(16 + 72 * sizeof(double));
    for (size_t isite = 0; isite < lat.vol4; isite++) {
        const gSite site = GInd::getSite(isite);
        const int32_t g[4] = {static_cast<int32_t>(lat.gPosX + site.coord.x), static_cast<int32_t>(lat.gPosY + site.coord.y),
                              static_cast<int32_t>(lat.gPosZ + site.coord.z), static_cast<int32_t>(lat.gPosT + site.coord.t)};
        std::memcpy(record.data(), g, 16);
        double *d = reinterpret_cast<double *>(record.data() + 16);
        for (int mu = 0; mu < 4; mu++) {
            const SU3<double> link = acc.getLink(GInd::getSiteMu(site, mu));
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    const COMPLEX(double) e = link(i, j);
                    *d++ = e.cREAL;
                    *d++ = e.cIMAG;
                }
            }
        }
        out.write(record.data(), static_cast<std::streamsize>(record.size()));
    }
    out.close();
    if (!out || std::rename((file + ".part").c_str(), file.c_str()) != 0) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump: write error on ", file));
    }
}

template<size_t HaloDepth>
void mdwfReadGaugeHost(CommunicationBase &comm, Gaugefield<double, false, HaloDepth, R18> &host, const std::string &base) {
    typedef GIndexer<All, HaloDepth> GInd;
    (void)comm;
    SU3Accessor<double, R18> acc = host.getAccessor();
    int64_t nranks = -1;
    size_t found = 0;
    for (int r = 0; nranks < 0 || r < nranks; r++) {
        const std::string file = mdwfFieldDumpRankFile(base, r);
        std::ifstream in(file, std::ios::binary);
        if (!in) {
            throw std::runtime_error(stdLogger.fatal("MDWF field dump: cannot open ", file));
        }
        MDWFFieldDumpHeader h{};
        in.read(reinterpret_cast<char *>(&h), sizeof(h));
        if (!in) {
            throw std::runtime_error(stdLogger.fatal("MDWF field dump: no header in ", file));
        }
        mdwfFieldDumpCheckHeader<HaloDepth>(h, 0, 0, 72, file);
        nranks = h.nranks;
        std::vector<char> record(16 + 72 * sizeof(double));
        for (int64_t s = 0; s < h.nsites; s++) {
            in.read(record.data(), static_cast<std::streamsize>(record.size()));
            if (!in) {
                throw std::runtime_error(stdLogger.fatal("MDWF field dump: short read on ", file));
            }
            int32_t g[4];
            std::memcpy(g, record.data(), 16);
            int x, y, z, t;
            if (!mdwfFieldDumpLocal<HaloDepth>(g, x, y, z, t)) {
                continue;
            }
            const double *d = reinterpret_cast<const double *>(record.data() + 16);
            const gSite site = GInd::getSite(x, y, z, t);
            for (int mu = 0; mu < 4; mu++) {
                SU3<double> link;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        link(i, j) = COMPLEX(double)(d[0], d[1]);
                        d += 2;
                    }
                }
                acc.setLink(GInd::getSiteMu(site, mu), link);
            }
            found++;
        }
    }
    if (found != GInd::getLatData().vol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump ", base, ": found ", found, " of ",
                                                 GInd::getLatData().vol4, " local sites"));
    }
}

// Device convenience wrappers (host staging copy inside).
template<size_t HaloDepth>
void mdwfDumpGauge(CommunicationBase &comm, Gaugefield<double, true, HaloDepth, R18> &field, const std::string &base,
                   const std::string &stagingName) {
    Gaugefield<double, false, HaloDepth, R18> host(comm, stagingName);
    host = field;
    mdwfDumpGaugeHost<HaloDepth>(comm, host, base);
}

template<size_t HaloDepth>
void mdwfReadGauge(CommunicationBase &comm, Gaugefield<double, true, HaloDepth, R18> &field, const std::string &base,
                   const std::string &stagingName) {
    Gaugefield<double, false, HaloDepth, R18> host(comm, stagingName);
    host = field;   // halo contents stay whatever they were; the caller updates the halo
    mdwfReadGaugeHost<HaloDepth>(comm, host, base);
    field = host;
}

// ---------------------------------------------------------------------------------------------------- even spinor

template<size_t HaloDepth, size_t Ls>
void mdwfDumpEvenSpinorHost(CommunicationBase &comm, const MDWFSpinor<double, false, Even, HaloDepth, Ls> &host,
                            const std::string &base) {
    typedef GIndexer<Even, HaloDepth> GIndE;
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const size_t sizeh = GIndE::getLatData().sizeh;
    const std::string file = mdwfFieldDumpRankFile(base, comm.MyRank());
    std::ofstream out(file + ".part", std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump: cannot open ", file, ".part for writing"));
    }
    const int64_t perSite = 24 * static_cast<int64_t>(Ls);
    const MDWFFieldDumpHeader h = mdwfFieldDumpHeader<HaloDepth>(comm, 1, Ls, sizeh, perSite);
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    const Vect12ArrayAcc<double> acc = host.getAccessor();
    std::vector<char> record(16 + perSite * sizeof(double));
    for (size_t isite = 0; isite < sizeh; isite++) {
        const gSite site = GIndE::getSite(isite);
        const int32_t g[4] = {static_cast<int32_t>(lat.gPosX + site.coord.x), static_cast<int32_t>(lat.gPosY + site.coord.y),
                              static_cast<int32_t>(lat.gPosZ + site.coord.z), static_cast<int32_t>(lat.gPosT + site.coord.t)};
        std::memcpy(record.data(), g, 16);
        double *d = reinterpret_cast<double *>(record.data() + 16);
        for (size_t s = 0; s < Ls; s++) {
            const Vect12<double> v = acc.getElement(GIndE::getSiteStack(site, s));
            for (int c = 0; c < 12; c++) {
                *d++ = v.data[c].cREAL;
                *d++ = v.data[c].cIMAG;
            }
        }
        out.write(record.data(), static_cast<std::streamsize>(record.size()));
    }
    out.close();
    if (!out || std::rename((file + ".part").c_str(), file.c_str()) != 0) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump: write error on ", file));
    }
}

template<size_t HaloDepth, size_t Ls>
void mdwfReadEvenSpinorHost(CommunicationBase &comm, MDWFSpinor<double, false, Even, HaloDepth, Ls> &host,
                            const std::string &base) {
    typedef GIndexer<Even, HaloDepth> GIndE;
    (void)comm;
    Vect12ArrayAcc<double> acc = host.getAccessor();
    const int64_t perSite = 24 * static_cast<int64_t>(Ls);
    int64_t nranks = -1;
    size_t found = 0;
    for (int r = 0; nranks < 0 || r < nranks; r++) {
        const std::string file = mdwfFieldDumpRankFile(base, r);
        std::ifstream in(file, std::ios::binary);
        if (!in) {
            throw std::runtime_error(stdLogger.fatal("MDWF field dump: cannot open ", file));
        }
        MDWFFieldDumpHeader h{};
        in.read(reinterpret_cast<char *>(&h), sizeof(h));
        if (!in) {
            throw std::runtime_error(stdLogger.fatal("MDWF field dump: no header in ", file));
        }
        mdwfFieldDumpCheckHeader<HaloDepth>(h, 1, Ls, perSite, file);
        nranks = h.nranks;
        std::vector<char> record(16 + perSite * sizeof(double));
        for (int64_t s = 0; s < h.nsites; s++) {
            in.read(record.data(), static_cast<std::streamsize>(record.size()));
            if (!in) {
                throw std::runtime_error(stdLogger.fatal("MDWF field dump: short read on ", file));
            }
            int32_t g[4];
            std::memcpy(g, record.data(), 16);
            int x, y, z, t;
            if (!mdwfFieldDumpLocal<HaloDepth>(g, x, y, z, t)) {
                continue;
            }
            if (((x + y + z + t) & 1) != 0) {
                throw std::runtime_error(stdLogger.fatal("MDWF field dump ", file, ": odd site in an even-spinor dump"));
            }
            const double *d = reinterpret_cast<const double *>(record.data() + 16);
            const gSite site = GIndE::getSite(x, y, z, t);
            for (size_t st = 0; st < Ls; st++) {
                Vect12<double> v;
                for (int c = 0; c < 12; c++) {
                    v.data[c] = COMPLEX(double)(d[0], d[1]);
                    d += 2;
                }
                acc.setElement(GIndE::getSiteStack(site, st), v);
            }
            found++;
        }
    }
    if (found != GIndE::getLatData().sizeh) {
        throw std::runtime_error(stdLogger.fatal("MDWF field dump ", base, ": found ", found, " of ",
                                                 GIndE::getLatData().sizeh, " local even sites"));
    }
}

template<size_t HaloDepth, size_t Ls>
void mdwfDumpEvenSpinor(CommunicationBase &comm, MDWFSpinor<double, true, Even, HaloDepth, Ls> &field,
                        const std::string &base, const std::string &stagingName) {
    MDWFSpinor<double, false, Even, HaloDepth, Ls> host(comm, stagingName);
    host = field;
    mdwfDumpEvenSpinorHost<HaloDepth, Ls>(comm, host, base);
}

template<size_t HaloDepth, size_t Ls>
void mdwfReadEvenSpinor(CommunicationBase &comm, MDWFSpinor<double, true, Even, HaloDepth, Ls> &field,
                        const std::string &base, const std::string &stagingName) {
    MDWFSpinor<double, false, Even, HaloDepth, Ls> host(comm, stagingName);
    host = field;
    mdwfReadEvenSpinorHost<HaloDepth, Ls>(comm, host, base);
    field = host;
}
