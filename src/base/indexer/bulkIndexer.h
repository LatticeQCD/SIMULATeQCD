/*
 * bulkIndexer.h
 *
 * L. Mazur
 *
 * Header file with all indexing objects and methods.
 *
 *     Full:   Indexing including sites on the Halo.
 *     Global: Indexing over all sublattices combined.
 *
 * As a rule of thumb, you should use methods without Full or Global appended; these are special functions that are
 * usually needed only for the lowest level programming.
 *
 */

#ifndef INDEXERDEVICE
#define INDEXERDEVICE

#include  "../../define.h"
#include "../gutils.h"
#include <vector>
#include "../latticeDimension.h"
#include "../latticeParameters.h"

/// ---------------------------------------------------------------------------------------- SITE STRUCTURES AND METHODS

struct sitexyzt {
    int x;
    int y;
    int z;
    int t;
    __device__ __host__ sitexyzt(int x, int y, int z, int t) : x(x), y(y), z(z), t(t) {};
    __device__ __host__ inline int& operator[](const int i) {
        if(i == 0) return x;
        if(i == 1) return y;
        if(i == 2) return z;
        if(i == 3) return t;
        x=99999;
        return x;
    }
    friend std::ostream& operator<< (std::ostream& stream, const sitexyzt& out){
        stream << out.x << " " << out.y << " " << out.z << " " << out.t;
        return stream;
    }

};
struct gSite {
    // This holds the local lattice site. This means, there are no halos involved. It cares about even / odd. That means
    // if the LatticeLayout is even or odd, the maximal value is sizeh.
    size_t isite;

    // This will hold the full lattice site, including halos. It cares about even / odd.  If that LatticeLayout is even
    // or odd, the maximal value is sizehFull.
    size_t isiteFull;

    // This holds coordinates x, y, z, t. They are always the full, true coordinates.
    sitexyzt coord, coordFull;

    // These constructors should only be called from GIndexer.
    __device__  __host__ inline gSite() : isite(0), isiteFull(0), coord(0, 0, 0, 0), coordFull(0, 0, 0, 0) {}

    __device__  __host__ inline gSite(size_t isite, size_t isiteFull, sitexyzt coord, sitexyzt coordFull) :
            isite(isite), isiteFull(isiteFull), coord(coord), coordFull(coordFull) {};


    __host__ friend inline std::ostream &operator << (std::ostream &s, const gSite &site) {
        s << "gSite: coord: " << site.coord.x << " " << site.coord.y << " " << site.coord.z << " " << site.coord.t << " "
          << "coordFull: " << site.coordFull.x << " " << site.coordFull.y << " " << site.coordFull.z << " " << site.coordFull.t << " "
          << "isite: " << site.isite << " isiteFull: " << site.isiteFull;
        return s;

    }
    __host__ inline std::string getStr() {
        std::ostringstream s;
        s << "gSite: coord: " << coord.x << " " << coord.y << " " << coord.z << " " << coord.t << " "
          << "coordFull: " << coordFull.x << " " << coordFull.y << " " << coordFull.z << " " << coordFull.t << " "
          << "isite: " << isite << " isiteFull: " << isiteFull;
        return s.str();
    }
};
struct gSiteStack : public gSite {
    // Index of the stack
    size_t isiteStack;
    size_t isiteStackFull;
    size_t stack;

    __device__ __host__ gSiteStack() : gSite(), isiteStack(0), isiteStackFull(0), stack(0){}

    __device__ __host__ gSiteStack(size_t isite, size_t isiteFull, sitexyzt coord,
                                   sitexyzt coordFull, size_t isiteStack, size_t isiteStackFull, size_t stack) :
            gSite(isite, isiteFull, coord, coordFull), isiteStack(isiteStack), isiteStackFull(isiteStackFull), stack(stack){}

    __device__ __host__ gSiteStack(gSite site, size_t isiteStack, size_t isiteStackFull, size_t stack) :
            gSite(site), isiteStack(isiteStack), isiteStackFull(isiteStackFull), stack(stack){}

    gSiteStack(const gSite) = delete;

    __host__ friend inline std::ostream &operator << (std::ostream &s, const gSiteStack &site) {
        s << "gSiteStack: coord: " << site.coord.x << " " << site.coord.y << " " << site.coord.z << " " << site.coord.t << " "
          << " coordFull: " << site.coordFull.x << " " << site.coordFull.y << " " << site.coordFull.z << " " << site.coordFull.t << " "
          << " isite: " << site.isite << " isiteFull: " << site.isiteFull << " stack: " << site.stack
          << " isiteStack: " << site.isiteStack << " isiteStackFull: " << site.isiteStackFull;
        return s;
    }
};
struct gSiteMu : public gSite {
    // This holds a link index. This index does not care about even/odd, it is always the full lattice index
    size_t indexMuFull;
    // Link direction.
    uint8_t mu;

    __device__ __host__ gSiteMu() : gSite(), indexMuFull(0), mu(0){}

    __device__ __host__ gSiteMu(size_t isite, size_t isiteFull, sitexyzt coord,
                                sitexyzt coordFull, size_t indexMuFull, uint8_t mu) :
            gSite(isite, isiteFull, coord, coordFull), indexMuFull(indexMuFull), mu(mu){}

    __device__ __host__ gSiteMu(gSite site, size_t indexMuFull, uint8_t mu)
            : gSite(site), indexMuFull(indexMuFull), mu(mu) {}

    gSiteMu(const gSite) = delete;
    gSiteMu(const gSiteStack) = delete;

    __host__ friend inline std::ostream &operator << (std::ostream &s, const gSiteMu &site) {
        s << "gSite: coord: " << site.coord.x     << " " << site.coord.y     << " " << site.coord.z     << " " << site.coord.t     << " "
          << "coordFull: "    << site.coordFull.x << " " << site.coordFull.y << " " << site.coordFull.z << " " << site.coordFull.t << " "
          << "isite: "        << site.isite
          << "isiteFull: "    << site.isiteFull
          << "mu: "           << site.mu
          << "indexMu_Full: "  << site.indexMuFull;
        return s;
    }
};

//! you can use these print functions for debugging, but in production code they are unused:
__attribute__((unused)) void  __host__ __device__ inline printGSite(const gSite& site) {
    printf("Coord: %d %d %d %d, coordFull: %d %d %d %d, isite: %lu, isiteFull %lu\n",
           site.coord.x, site.coord.y, site.coord.z, site.coord.t,
           site.coordFull.x, site.coordFull.y, site.coordFull.z, site.coordFull.t,
           site.isite,
           site.isiteFull);
}
__attribute__((unused)) void __host__ __device__  inline printGSiteStack(const gSiteStack& site) {
    printf("Coord: %d %d %d %d, coordFull: %d %d %d %d, isite: %lu, isiteFull %lu, stack: %lu, isiteStack: %lu, isiteStackFull %lu\n",
           site.coord.x, site.coord.y, site.coord.z, site.coord.t,
           site.coordFull.x, site.coordFull.y, site.coordFull.z, site.coordFull.t,
           site.isite,
           site.isiteFull,
           site.stack,
           site.isiteStack,
           site.isiteStackFull);
}
__attribute__((unused)) void __host__ __device__ inline printGSiteStack(const gSiteMu& site){
    printf("Coord: %d %d %d %d, coordFull: %d %d %d %d, isite: %lu, isiteFull %lu, mu: %d, indexMu_Full: %lu\n",
           site.coord.x, site.coord.y, site.coord.z, site.coord.t,
           site.coordFull.x, site.coordFull.y, site.coordFull.z, site.coordFull.t,
           site.isite,
           site.isiteFull,
           site.mu,
           site.indexMuFull);
}

/// ------------------------------------------------------------------------------------------------------- LATTICE DATA

struct LatticeData {
    size_t HaloDepth[4];
    size_t lx, ly, lz, lt; //! bulk (sub)lattice extents
    size_t lxFull, lyFull, lzFull, ltFull; //! full (sub)lattice extents
    size_t vol1, vol2, vol3, vol4; //! products of bulk (sub)lattice extents
    size_t vol1Full, vol2Full, vol3Full, vol4Full; //! products of full (sub)lattice extents
    size_t vol3h, vol3hFull; //! "h" stand for "half"
    size_t sizeh, sizehFull; //! number of lattice sites divided by 2

    size_t globLX, globLY, globLZ, globLT; //! global lattice extents
    size_t globvol1, globvol2, globvol3, globvol4; //! products of global lattice extents

    //! Offset of the bulk sublattice from 0 0 0 0.
    //! For example, when splitting 2 1 1 1 on a 20 20 20 20 lattice,
    //! this will be 0 0 0 0 on rank 0, and 10 0 0 0 on rank 1
    size_t gPosX, gPosY, gPosZ, gPosT;

    LatticeData() {}

    __host__ __device__ LatticeData(size_t _lx, size_t _ly, size_t _lz, size_t _lt, size_t _HaloDepth, unsigned int _Nodes[4],
                                    size_t _globX, size_t _globY, size_t _globZ, size_t _globT,
                                    size_t _gPosX, size_t _gPosY, size_t _gPosZ, size_t _gPosT) :

            HaloDepth{_Nodes[0] != 1 ? _HaloDepth : 0,
                      _Nodes[1] != 1 ? _HaloDepth : 0,
                      _Nodes[2] != 1 ? _HaloDepth : 0,
                      _Nodes[3] != 1 ? _HaloDepth : 0},
            lx(_lx),
            ly(_ly),
            lz(_lz),
            lt(_lt),
            lxFull(_lx + 2 * HaloDepth[0]),
            lyFull(_ly + 2 * HaloDepth[1]),
            lzFull(_lz + 2 * HaloDepth[2]),
            ltFull(_lt + 2 * HaloDepth[3]),
            vol1(_lx),
            vol2(vol1 * _ly),
            vol3(vol2 * _lz),
            vol4(vol3 * _lt),

            vol1Full(lxFull),
            vol2Full(vol1Full * lyFull),
            vol3Full(vol2Full * lzFull),
            vol4Full(vol3Full * ltFull),
            vol3h(vol3 / 2),
            vol3hFull(vol3Full / 2),
            sizeh(vol4 / 2),
            sizehFull(vol4Full / 2),

            globLX(_globX),
            globLY(_globY),
            globLZ(_globZ),
            globLT(_globT),
            globvol1(_globX),
            globvol2(globvol1 * _globY),
            globvol3(globvol2 * _globZ),
            globvol4(globvol3 * _globT),
            gPosX(_gPosX),
            gPosY(_gPosY),
            gPosZ(_gPosZ),
            gPosT(_gPosT) {}

    __device__ __host__ sitexyzt globalPos(sitexyzt n) {

        sitexyzt coord = sitexyzt(gPosX + n.x,gPosY + n.y,gPosZ + n.z,gPosT + n.t);

        //! periodicity
        for(int i = 0; i<4; i++) {
            if (coord[i] < 0) coord[i] += globalLatticeXYZT()[i];
            if (coord[i] >= globalLatticeXYZT()[i]) coord[i] -= globalLatticeXYZT()[i];
        }
        return coord;
    }

    __device__ __host__ bool isLocal(sitexyzt globalsite){
        //! make sure globalsite is valid, i.e. not negative or greater than lattice extents!

        // consider lattice 20 20 20 20 with split 2 2 1 1
        // then local extents are 10 10 20 20
        // and gPos are:
        // rank 0:  0  0 0 0
        // rank 1: 10  0 0 0
        // rank 2:  0 10 0 0
        // rank 3: 10 10 0 0
        // lets say source is at 15 15 15 15
        // it needs to check that globalsite coords are greater or equal gPos, but it also needs to check that
        // gPos + local extent is greater or equal globalsite !
        // all ranks fulfill the first criterion in this example, but only one fulfills the second:
        // rank 3 should have the source, since 10 10 20 20 + 10 10 0 0 is 20 20 20 20 which is greater than 15 15 15 15
        if ( globalsite.x >= static_cast<int>(gPosX) and
             globalsite.y >= static_cast<int>(gPosY) and
             globalsite.z >= static_cast<int>(gPosZ) and
             globalsite.y >= static_cast<int>(gPosT) and
             globalsite.x < static_cast<int>(gPosX+lx) and
             globalsite.y < static_cast<int>(gPosY+ly) and
             globalsite.z < static_cast<int>(gPosZ+lz) and
             globalsite.t < static_cast<int>(gPosT+lt)) {
            return true;
        }
        return false;
    }

    __host__ LatticeDimensions globalPos(LatticeDimensions n) {

        LatticeDimensions coord = LatticeDimensions(gPosX,gPosY,gPosZ,gPosT) + n;

        for(int i = 0; i<4; i++) {
            if (coord[i] < 0) coord[i] += globalLattice()[i];

            if(coord[i]>= globalLattice()[i]) coord[i] -= globalLattice()[i];
        }
        return coord;
    }

    __host__ LatticeDimensions globalLattice() {
        return LatticeDimensions(globLX,globLY,globLZ,globLT);
    }

    __host__ LatticeDimensions localLattice() {
        return LatticeDimensions(lx,ly,lz,lt);
    }

    __device__ __host__ sitexyzt globalLatticeXYZT() {
        return sitexyzt(globLX,globLY,globLZ,globLT);
    }


};
extern __device__ __constant__ struct LatticeData globLatDataGPU[MAXHALO + 1];
const extern struct LatticeData globLatDataCPU[MAXHALO + 1];

/// --------------------------------------------------------------------------------------------- INDEXER INITIALIZATION

void initGPUBulkIndexer(size_t lx, size_t ly, size_t lz, size_t lt, sitexyzt globCoord, sitexyzt globPos, unsigned int Nodes[4]);
void initCPUBulkIndexer(size_t lx, size_t ly, size_t lz, size_t lt, sitexyzt globCoord, sitexyzt globPos, unsigned int Nodes[4]);
void initGPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt, unsigned int Nodes[4], unsigned int Halos[4]);
void initCPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt, unsigned int Nodes[4], unsigned int Halos[4]);

class CommunicationBase;
void initIndexer(const size_t HaloDepth, const LatticeParameters &param, CommunicationBase &comm);

/// ----------------------------------------------------------------------------------------------------------- GINDEXER

/// This is the indexing class. Indexing should be done only by using this class. It fully cares about even/odd lattice
/// layouts. For instance, GIndexer<Odd, 1>::site_up will return an even lattice object.
template<Layout LatLayout, size_t HaloDepth=MAXUSEDHALO>
class GIndexer {
public:
    __device__ __host__ GIndexer() = default;
    __device__ __host__ inline static LatticeData getLatData() {

#ifdef __GPU_ARCH__
        return globLatDataGPU[HaloDepth];
#else
        return globLatDataCPU[HaloDepth];
#endif
    }

    /// ---------------------------------------------------------------------------------------------------- getSite*
    /// BULK (NO HALOS)
    __device__ __host__ inline static gSite getSite(size_t isite) {
        sitexyzt coord(0, 0, 0, 0);
        sitexyzt coordFull(0, 0, 0, 0);
        size_t isiteFull = 0;
        if (LatLayout == All) {
            coord = indexToCoord(isite);
            coordFull = coordToFullCoord(coord);
            isiteFull = coordToIndex_Full(coordFull);
        } else if (LatLayout == Even) {
            coord = indexToCoord_eo(isite, 0);
            coordFull = coordToFullCoord(coord);
            isiteFull = coordToIndex_Full_eo(coordFull);
        } else if (LatLayout == Odd) {
            coord = indexToCoord_eo(isite, 1);
            coordFull = coordToFullCoord(coord);
            isiteFull = coordToIndex_Full_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSite getSite(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx) {
        return getSite(_blockDim.x * _blockIdx.x + _threadIdx.x);
    }
    #endif
    __device__ __host__ inline static gSite getSite(int x, int y, int z, int t) {
        sitexyzt coord = sitexyzt(x, y, z, t);
        sitexyzt coordFull = coordToFullCoord(coord);
        size_t isite = 0;
        size_t isiteFull = 0;

        if (LatLayout == All) {
            isite = coordToIndex_Bulk(coord);
            isiteFull = coordToIndex_Full(coordFull);
        } else {
            isite = coordToIndex_Bulk_eo(coord);
            isiteFull = coordToIndex_Full_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    __device__ __host__ inline static gSite getSite(sitexyzt coord) {
        return getSite(coord.x,coord.y,coord.z,coord.t);
    }

    /*! This creates a gSite object, assuming that you only want to index the spacelike volume. This would
 happen whenever you call a kernel running over spacelike indices only. All coordinates will be of
 the form (x, y, z, 0). The indices isite and isiteFull will by bounded by their respective 3-volumes.
 The indexing needs to change, because there are fewer sites than with the full bulk.*/
    __device__ __host__ inline static gSite getSiteSpatial(size_t isite) {
        sitexyzt coord(0, 0, 0, 0);
        sitexyzt coordFull(0, 0, 0, 0);
        size_t isiteFull = 0;
        if (LatLayout == All) {
            coord     = indexToCoord_Spatial(isite);
            coordFull = coordToFullCoord(coord);     // Running over spatial does not change this mapping.
            isiteFull = coordToIndex_SpatialFull(coordFull);
        } else if (LatLayout == Even) {
            coord     = indexToCoord_Spatial_eo(isite, 0);
            coordFull = coordToFullCoord(coord);
            isiteFull = coordToIndex_SpatialFull_eo(coordFull);
        } else if (LatLayout == Odd) {
            coord     = indexToCoord_Spatial_eo(isite, 1);
            coordFull = coordToFullCoord(coord);
            isiteFull = coordToIndex_SpatialFull_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSite getSiteSpatial(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx) {
        return getSiteSpatial(_blockDim.x * _blockIdx.x + _threadIdx.x);
    }
    #endif
    __device__ __host__ inline static gSite getSiteSpatial(int x, int y, int z, int t) {
        // There is probably a way to allow t>0. My worry right now is that there is that if you allow
        // t>0, there is no longer a one-to-one correspondence between isite and coord.
        sitexyzt coord = sitexyzt(x, y, z, t);
        sitexyzt coordFull = coordToFullCoord(coord);
        size_t isite = 0;
        size_t isiteFull = 0;
        if (LatLayout == All) {
            isite     = coordToIndex_SpatialBulk(coord);
            isiteFull = coordToIndex_SpatialFull(coordFull);
        } else {
            isite     = coordToIndex_SpatialBulk_eo(coord);
            isiteFull = coordToIndex_SpatialFull_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }

    /// FULL (WITH HALOS)
    __device__ __host__ inline static gSite getSiteFull(size_t isiteFull) {
        sitexyzt coord(0, 0, 0, 0);
        sitexyzt coordFull(0, 0, 0, 0);
        size_t isite = 0;
        if (LatLayout == All) {
            coordFull = indexToCoord_Full(isiteFull);
            coord = fullCoordToCoord(coordFull);
            isite = coordToIndex_Bulk(coord);
        } else if (LatLayout == Even) {
            coordFull = indexToCoord_Full_eo(isiteFull, 0);
            coord = fullCoordToCoord(coordFull);
            isite = coordToIndex_Bulk_eo(coord);
        } else if (LatLayout == Odd) {
            coordFull = indexToCoord_Full_eo(isiteFull, 1);
            coord = fullCoordToCoord(coordFull);
            isite = coordToIndex_Bulk_eo(coord);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSite getSiteFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx) {
        return getSiteFull(_blockDim.x * _blockIdx.x + _threadIdx.x);
    }
    #endif
    __device__ __host__ inline static gSite getSiteFull(int x, int y, int z, int t) {
        sitexyzt coordFull = sitexyzt(x, y, z, t);
        sitexyzt coord = fullCoordToCoord(coordFull);
        size_t isite = 0;
        size_t isiteFull = 0;
        if (LatLayout == All) {
            isite = coordToIndex_Bulk(coord);
            isiteFull = coordToIndex_Full(coordFull);
        } else {
            isite = coordToIndex_Bulk_eo(coord);
            isiteFull = coordToIndex_Full_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    __device__ __host__ inline static gSite getSiteFull(sitexyzt coordfull) {
        return getSiteFull(coordfull.x,coordfull.y,coordfull.z,coordfull.t);
    }

    __device__ __host__ inline static gSite getSiteSpatialFull(size_t isiteFull) {
        sitexyzt coord(0, 0, 0, 0);
        sitexyzt coordFull(0, 0, 0, 0);
        size_t isite = 0;
        if (LatLayout == All) {
            coordFull = indexToCoord_SpatialFull(isiteFull);
            coord     = fullCoordToCoord(coordFull);
            isite     = coordToIndex_SpatialBulk(coord);
        } else if (LatLayout == Even) {
            coordFull = indexToCoord_Full_eo(isiteFull, 0);
            coord     = fullCoordToCoord(coordFull);
            isite     = coordToIndex_SpatialBulk_eo(coord);
        } else if (LatLayout == Odd) {
            coordFull = indexToCoord_SpatialFull_eo(isiteFull, 1);
            coord     = fullCoordToCoord(coordFull);
            isite     = coordToIndex_SpatialBulk_eo(coord);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSite getSiteSpatialFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx) {
        return getSiteSpatialFull(_blockDim.x * _blockIdx.x + _threadIdx.x);
    }
    #endif
    __device__ __host__ inline static gSite getSiteSpatialFull(int x, int y, int z, int t) {
        sitexyzt coordFull = sitexyzt(x, y, z, t);
        sitexyzt coord = fullCoordToCoord(coordFull);
        size_t isite = 0;
        size_t isiteFull = 0;
        if (LatLayout == All) {
            isite     = coordToIndex_SpatialBulk(coord);
            isiteFull = coordToIndex_SpatialFull(coordFull);
        } else {
            isite     = coordToIndex_SpatialBulk_eo(coord);
            isiteFull = coordToIndex_SpatialFull_eo(coordFull);
        }
        return gSite(isite, isiteFull, coord, coordFull);
    }

    /// -------------------------------------------------------------------------------------------------- getSiteMu*
    /// BULK (NO HALOS)

    //! two helper functions for getSiteMu*
    __device__ __host__ inline static size_t coordMuToIndexMu_Full(const int x, const int y, const int z, const int t, const int mu) {
        return (((x + y*getLatData().vol1Full
                  + z*getLatData().vol2Full
                  + t*getLatData().vol3Full) >> 0x1) // integer division by two
                +getLatData().sizehFull*((x + y + z + t) & 0x1) // 0 if x+y+z+t is even, 1 if it is odd
                + mu*getLatData().vol4Full);
    }
    __device__ __host__ inline static size_t indexMu_Full(const gSite site, const int mu) {
        return coordMuToIndexMu_Full(site.coordFull.x, site.coordFull.y, site.coordFull.z, site.coordFull.t, mu);
    }

    __device__ __host__ inline static gSiteMu getSiteMu(size_t isite, size_t mu) {
        gSite site(getSite(isite));
        size_t indexmufull = indexMu_Full(site, mu);
        return gSiteMu(site, indexmufull, mu);
    }
    #ifndef USE_SYCL    
    __device__ __host__ inline static gSiteMu getSiteMu(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx, size_t mu){
        return getSiteMu(_blockDim.x * _blockIdx.x + _threadIdx.x, mu);
    }
    
    __device__ __host__ inline static gSiteMu getSiteMu(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx){
        //! It gets the mu index from the y direction of the block.
        return getSiteMu(_blockDim.x * _blockIdx.x + _threadIdx.x, _threadIdx.y);
    }
    #endif
    __device__ __host__ inline static gSiteMu getSiteMu(gSite site, size_t mu) {
        size_t indexmufull = indexMu_Full(site, mu);
        return gSiteMu(site, indexmufull, mu);
    }
    __device__ __host__ inline static gSiteMu getSiteMu(int x, int y, int z, int t, size_t mu){
        return getSiteMu(getSite(x, y, z, t), mu);
    }

    __device__ __host__ inline static gSiteMu getSiteSpatialMu(size_t isite, size_t mu) {
        gSite site(getSiteSpatial(isite));
        size_t indexmufull = indexMu_Full(site, mu);
        return gSiteMu(site, indexmufull, mu);
    }
    /// FULL (WITH HALOS)
    __device__ __host__ inline static gSiteMu getSiteMuFull(size_t isiteFull, size_t mu) {
        gSite site(getSiteFull(isiteFull));
        size_t indexmufull = indexMu_Full(site, mu);
        return gSiteMu(site, indexmufull, mu);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSiteMu getSiteMuFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx, size_t mu){
        return getSiteMuFull(_blockDim.x * _blockIdx.x + _threadIdx.x, mu);
    }
    __device__ __host__ inline static gSiteMu getSiteMuFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx){
        //!get the mu index from the y direction of the block.
        return getSiteMuFull(_blockDim.x * _blockIdx.x + _threadIdx.x, _threadIdx.y);
    }
    #endif
    __device__ __host__ inline static gSiteMu getSiteMuFull(int x, int y, int z, int t, size_t mu){
        return getSiteMu(getSiteFull(x, y, z, t), mu);
    }

    /// --------------------------------------------------------------------------------------------------- getSiteStack
    /// BULK (NO HALOS)
    __device__ __host__ inline static gSiteStack getSiteStack(const gSite& site, const size_t stack){
        size_t isiteStack;
        size_t isiteStackFull;
        if (LatLayout == All) {
            isiteStack = site.isite + stack * getLatData().vol4;
            isiteStackFull = site.isiteFull + stack * getLatData().vol4Full;
        } else {
            isiteStack = site.isite + stack * getLatData().sizeh;
            isiteStackFull = site.isiteFull + stack * getLatData().sizehFull;
        }
        gSiteStack ret(site, isiteStack, isiteStackFull, stack);
        return ret;
    }
    __device__ __host__ inline static gSiteStack getSiteStack(const size_t isite, const size_t stack){
        return getSiteStack(getSite(isite), stack);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSiteStack getSiteStack(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx){
        return getSiteStack(_blockDim.x * _blockIdx.x + _threadIdx.x, _threadIdx.y);
    }
    __device__ __host__ inline static gSiteStack getSiteStack(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx, const size_t stack){
        return getSiteStack(_blockDim.x * _blockIdx.x + _threadIdx.x, stack);
    }
    #endif
    __device__ __host__ inline static gSiteStack getSiteStack(int x, int y, int z, int t, int stack) {
        return getSiteStack(getSite(x, y, z, t), stack);
    }
    __device__ __host__ inline static gSiteStack getSiteStack(sitexyzt coord, int stack) {
        return getSiteStack(getSite(coord.x, coord.y, coord.z, coord.t), stack);
    }
    __device__ __host__ inline static size_t getStack(const gSiteStack& site) {
        return site.stack;
    }
    
    __device__ __host__ inline static gSiteStack getSiteStackOdd(const gSite& site, const size_t stack){
        size_t isiteStack;
        size_t isiteStackFull;
        if (LatLayout == All) {
            isiteStack = site.isite + getLatData().sizeh + stack * getLatData().vol4;
            isiteStackFull = site.isiteFull + getLatData().sizehFull + stack * getLatData().vol4Full;
        } else {
            isiteStack = site.isite + stack * getLatData().sizeh;
            isiteStackFull = site.isiteFull + stack * getLatData().sizehFull;
        }
        gSiteStack ret(site, isiteStack, isiteStackFull, stack);
        return ret;
    }
    __device__ __host__ inline static gSiteStack getSiteStackOdd(const size_t isite, const size_t stack){
        return getSiteStackOdd(getSite(isite), stack);
    }

    #ifndef USE_SYCL
    __device__ __host__ inline static gSiteStack getSiteStackOdd(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx){
        return getSiteStackOdd(_blockDim.x * _blockIdx.x + _threadIdx.x, _threadIdx.y);
    }
    __device__ __host__ inline static gSiteStack getSiteStackOdd(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx, const size_t stack){
        return getSiteStackOdd(_blockDim.x * _blockIdx.x + _threadIdx.x, stack);
    }
    #endif

    /// FULL (WITH HALOS)
    __device__ __host__ inline static gSiteStack getSiteStackFull(const size_t isiteFull, const size_t stack){
        return getSiteStack(getSiteFull(isiteFull), stack);
    }
    #ifndef USE_SYCL
    __device__ __host__ inline static gSiteStack getSiteStackFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx){
        gSiteStack ret = getSiteStackFull(_blockDim.x * _blockIdx.x + _threadIdx.x, _threadIdx.y);
        return ret;
    }
    __device__ __host__ inline static gSiteStack getSiteStackFull(const dim3& _blockDim, const uint3& _blockIdx, const uint3& _threadIdx, const size_t stack){
        gSiteStack ret = getSiteStackFull(_blockDim.x * _blockIdx.x + _threadIdx.x, stack);
        return ret;
    }
    #endif
    __device__ __host__ inline static gSiteStack getSiteStackFull(int x, int y, int z, int t, int stack) {
        return getSiteStack(getSiteFull(x, y, z, t), stack);
    }

    /// ----------------------------------------------------------------------------------- CONVERT BETWEEN EVEN AND ODD

    template<Layout LatLayout2, size_t HaloDepth2> __device__ __host__ inline static gSite convertSite(const gSite& site){
        return GIndexer<LatLayout2, HaloDepth2>::getSite(site.coord.x, site.coord.y, site.coord.z, site.coord.t);
    }
    template<Layout LatLayout2, size_t HaloDepth2> __device__ __host__ inline static gSiteMu convertSite(const gSiteMu& site){
        return GIndexer<LatLayout2, HaloDepth2>::getSiteMu(site.coord.x, site.coord.y, site.coord.z, site.coord.t, site.mu);
    }
    template<Layout LatLayout2, size_t HaloDepth2> __device__ __host__ inline static gSiteStack convertSite(const gSiteStack& site){
        return GIndexer<LatLayout2, HaloDepth2>::getSiteStack(site.coord.x, site.coord.y, site.coord.z, site.coord.t, site.stack);
    }
    //! Given an Even/Odd gSite object, this returns an All gSite object.
    __device__ __host__ inline static gSite convertToAll(gSite& site) {
        size_t isite = site.isite + (LatLayout == Odd)*getLatData().sizeh;
        size_t isiteFull = site.isiteFull + (LatLayout == Odd)*getLatData().sizehFull;
        return gSite(isite, isiteFull, site.coord, site.coordFull);
    }

    /// ------------------------------------------------ CONVERT BETWEEN BULK SPACETIME COORDINATES AND FULL COORDINATES
    __device__ __host__ inline static sitexyzt coordToFullCoord(sitexyzt coord) {
        coord.x += getLatData().HaloDepth[0];
        coord.y += getLatData().HaloDepth[1];
        coord.z += getLatData().HaloDepth[2];
        coord.t += getLatData().HaloDepth[3];
        return coord;
    }
    __device__ __host__ inline static sitexyzt fullCoordToCoord(sitexyzt fullCoord) {
        fullCoord.x -= getLatData().HaloDepth[0];
        fullCoord.y -= getLatData().HaloDepth[1];
        fullCoord.z -= getLatData().HaloDepth[2];
        fullCoord.t -= getLatData().HaloDepth[3];
        return fullCoord;
    }

    __device__ __host__ inline static sitexyzt globalCoordToLocalCoord(sitexyzt coord) {
        coord.x -= getLatData().gPosX;
        coord.y -= getLatData().gPosY;
        coord.z -= getLatData().gPosZ;
        coord.t -= getLatData().gPosT;
        return coord;
    }

    /// -------------------------------------------------------------------- CONVERT SPACETIME COORDINATES TO DATA INDEX
    /// BULK (NO HALOS)
    __device__ __host__ inline static size_t coordToIndex_Bulk(const sitexyzt coord) {
        return (((coord.x + coord.y*getLatData().vol1
                  + coord.z*getLatData().vol2
                  + coord.t*getLatData().vol3) >> 0x1) // integer division by two
                +getLatData().sizeh * ((coord.x + coord.y + coord.z + coord.t) & 0x1)); // 0 if x+y+z+t is even, 1 if it is odd
    }
    __device__ __host__ inline static size_t coordToIndex_Bulk_eo(const sitexyzt coord) {
        return ((coord.x + coord.y*getLatData().vol1
                 + coord.z*getLatData().vol2
                 + coord.t*getLatData().vol3) >> 0x1);
    }
    __device__ __host__ inline static size_t coordToIndex_SpatialBulk(const sitexyzt coord) {
        return (((coord.x + coord.y*getLatData().vol1
                  + coord.z*getLatData().vol2) >> 0x1)
                + getLatData().vol3h*((coord.x + coord.y + coord.z) & 0x1));
    }
    __device__ __host__ inline static size_t coordToIndex_SpatialBulk_eo(const sitexyzt coord) {
        return ((coord.x + coord.y*getLatData().vol1
                 + coord.z*getLatData().vol2) >> 0x1);
    }
    /// FULL (WITH HALOS)
    __device__ __host__ inline static size_t coordToIndex_Full(const sitexyzt coordFull) {
        return (((coordFull.x + coordFull.y*getLatData().vol1Full
                  + coordFull.z*getLatData().vol2Full
                  + coordFull.t*getLatData().vol3Full) >> 0x1)
                + getLatData().sizehFull*((coordFull.x + coordFull.y + coordFull.z + coordFull.t) & 0x1));
    }
    __device__ __host__ inline static size_t coordToIndex_Full_eo(const sitexyzt coordFull) {
        return ((coordFull.x + coordFull.y * getLatData().vol1Full + coordFull.z * getLatData().vol2Full +
                 coordFull.t * getLatData().vol3Full) >> 0x1);
    }
    __device__ __host__ inline static size_t coordToIndex_SpatialFull(const sitexyzt coordFull) {
        return (((coordFull.x + coordFull.y*getLatData().vol1Full
                  + coordFull.z*getLatData().vol2Full) >> 0x1)
                + getLatData().vol3hFull*((coordFull.x + coordFull.y + coordFull.z) & 0x1));
    }
    __device__ __host__ inline static size_t coordToIndex_SpatialFull_eo(const sitexyzt coordFull) {
        return ((coordFull.x + coordFull.y*getLatData().vol1Full
                 + coordFull.z*getLatData().vol2Full) >> 0x1);
    }
    __host__ inline static size_t localCoordToGlobalIndex(LatticeDimensions coord) {
        LatticeData lat = GIndexer<LatLayout, HaloDepth>::getLatData();
        LatticeDimensions globCoord = lat.globalPos(coord);
        return (globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
               globCoord[3] * lat.globLX * lat.globLY * lat.globLZ);
    }

    /// -------------------------------------------------------------------- CONVERT DATA INDEX TO SPACETIME COORDINATES
    /// BULK (NO HALOS)
    __device__ __host__ inline static sitexyzt indexToCoord(const size_t site) {
        int x, y, z, t;
        int par, normInd, tmp;

        //! figure out the parity:
        divmod(site, getLatData().sizeh, par, normInd);
        //! par now contains site/sizeh (integer division), so it should be 0 (even) or 1 (odd).
        //! normInd contains the remainder.
        //! Adjacent odd and even sites will have the same remainder.

        //! Now think of an interlaced list of all even and all odd sites, such that the entries alternate
        //! between even and odd sites. Since adjacent sites have the same remainder, the remainder functions as
        //! the index of the *pairs* of adjacent sites.
        //! The next step is now to double this remainder so that we can work with it as an index for the single sites
        //! and not the pairs.
        normInd = normInd << 0x1; //! multiply remainder by two

        //! Now get the slower running coordinates y,z,t:
        //! To get these, we simply integer-divide the index by the product of all faster running lattice extents,
        //! and then use the remainder as the index for the next-faster coordinate and so on.
        divmod(normInd, getLatData().vol3, t, tmp); //! t now contains normInd/vol3, tmp the remainder
        divmod(tmp,     getLatData().vol2, z, tmp); //! z now contains tmp/vol2, tmp the remainder
        divmod(tmp,     getLatData().vol1, y, x);   //! x now contains tmp/vol1, x the remainder

        //! One problem remains: since we doubled the remainder and since the lattice extents have to be even,
        //! x is now also always even, which is of course not correct.
        //! We may need to correct it to be odd, depending on the supposed parity we found in the beginning,
        //! and depending on whether y+z+t is even or odd:
        if (!isOdd(x)){ //TODO isn't x always even? ...
            if (   ( par && !isOdd(y + z + t))    //!  odd parity but y+z+t is even, so x should be odd
                   or (!par &&  isOdd(y + z + t))) { //! even parity but y+z+t is  odd, so x should be odd
                ++x;
            }
        }
        //! Note that we always stay inside of a pair of adjacent sites when incrementing x here.

        return sitexyzt(x, y, z, t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_eo(const size_t site, int par) {
        int x, y, z, t;
        int tmp;
        // double site
        size_t sited = site<<0x1;
        // get x,y,z,t
        divmod(sited, getLatData().vol3, t, tmp);
        divmod(tmp,   getLatData().vol2, z, tmp);
        divmod(tmp,   getLatData().vol1, y, x);

        // correct effect of divison by two (adjacent odd and even numbers mapped to same number)
        if (par && !isOdd(x) && !isOdd(y + z + t))
            ++x;
        if (!par && !isOdd(x) && isOdd(y + z + t))
            ++x;

        return sitexyzt(x, y, z, t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_Spatial(const size_t site) {
        int x, y, z, t;
        int par, normInd, tmp;

        divmod(site,getLatData().vol3h,par,normInd);
        normInd = normInd << 0x1;

        t=0; // We are on the space-like volume.

        divmod(normInd, getLatData().vol2, z, tmp);
        divmod(tmp,     getLatData().vol1, y, x);

        if ( par && !isOdd(x) && !isOdd(y + z))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z))
            ++x;

        return sitexyzt(x,y,z,t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_Spatial_eo(const size_t site, int par) {
        int x, y, z, t;
        int tmp;
        size_t sited = site << 0x1;

        divmod(sited, getLatData().vol2, z, tmp);
        divmod(tmp,   getLatData().vol1, y, x);

        if ( par && !isOdd(x) && !isOdd(y + z))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z))
            ++x;

        t=0;

        return sitexyzt(x, y, z, t);
    }
    /// FULL (WITH HALOS)
    __device__ __host__ inline static sitexyzt indexToCoord_Full(const size_t siteFull) {
        int x, y, z, t;
        int par, normInd, tmp;

        // figure out the direction
        divmod(siteFull, getLatData().sizehFull, par, normInd);
        normInd = normInd << 0x1;
        // get x,y,z,t
        divmod(normInd, getLatData().vol3Full, t, tmp);
        divmod(tmp,     getLatData().vol2Full, z, tmp);
        divmod(tmp,     getLatData().vol1Full, y, x);

        // correct effect of divison by two (adjacent odd and even numbers mapped to same number)
        if (par && !isOdd(x) && !isOdd(y + z + t))
            ++x;
        if (!par && !isOdd(x) && isOdd(y + z + t))
            ++x;

        return sitexyzt(x, y, z, t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_SpatialFull(const size_t siteFull) {
        int x, y, z, t;
        int par, normInd, tmp;

        divmod(siteFull, getLatData().vol3hFull, par, normInd);
        normInd = normInd << 0x1;

        t=getLatData().HaloDepth[3]; // t=0 timeslice in FullCoord language

        divmod(normInd, getLatData().vol2Full, z, tmp);
        divmod(tmp,     getLatData().vol1Full, y, x);

        // The indexing should be independent of t
        if ( par && !isOdd(x) && !isOdd(y + z))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z))
            ++x;

        return sitexyzt(x, y, z, t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_Full_eo(const size_t siteFull, int par) {
        int x, y, z, t;
        int tmp;

        size_t siteFulld = siteFull << 0x1;

        divmod(siteFulld, getLatData().vol3Full, t, tmp);
        divmod(tmp, getLatData().vol2Full, z, tmp);
        divmod(tmp, getLatData().vol1Full, y, x);

        if ( par && !isOdd(x) && !isOdd(y + z + t))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z + t))
            ++x;

        return sitexyzt(x, y, z, t);
    }
    __device__ __host__ inline static sitexyzt indexToCoord_SpatialFull_eo(const size_t siteFull, int par) {
        int x, y, z, t;
        int tmp;

        size_t siteFulld = siteFull << 0x1;

        t=getLatData().HaloDepth[3];

        divmod(siteFulld, getLatData().vol2Full, z, tmp);
        divmod(tmp,       getLatData().vol1Full, y, x);

        if ( par && !isOdd(x) && !isOdd(y + z))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z))
            ++x;

        return sitexyzt(x, y, z, t);
    }

    //! This function is needed when one wants to have the sites time ordered. For example if one wants to reduce only
    //! values on each timeslice.
    __device__ __host__ inline static size_t siteTimeOrdered(const gSite &site) {
        sitexyzt c = site.coord;
        return c.x + c.y*getLatData().vol1 + c.z*getLatData().vol2 + c.t*getLatData().vol3;
    }

    /// --------------------------------------------------------------------------------------- site_move (COMPILE TIME)

    //! Functions to move an (almost) arbitrary number of steps in a given direction. mu_steps is given as a template
    //! parameter, because it guarantees the compiler will optimize. Unfortunately because it's determined at compile
    //! time, this means you cannot pass these functions a dynamic argument.

    /// --------------------------------------------------------------------------------------- site_move: ONE DIRECTION
    template<int mu_steps> __device__ __host__ inline static gSite site_move(const gSite &s, const int mu) {
        sitexyzt tmp = site_move<mu_steps>(s.coordFull, mu);
        return getSiteFull(tmp.x, tmp.y, tmp.z, tmp.t);
    }
    template<int mu_steps> __device__ __host__ inline static gSiteMu site_move(const gSiteMu &s, const int mu) {
        sitexyzt tmp = site_move<mu_steps>(s.coordFull, mu);
        return getSiteMuFull(tmp.x, tmp.y, tmp.z, tmp.t, s.mu);
    }
    template<int mu_steps> __device__ __host__ inline static gSiteStack site_move(const gSiteStack &s, const int mu) {
        sitexyzt tmp = site_move<mu_steps>(s.coordFull, mu);
        return getSiteStackFull(tmp.x, tmp.y, tmp.z, tmp.t, s.stack);
    }
    template<int mu_steps> __device__ __host__ inline static sitexyzt site_move(sitexyzt s, const int mu) {

        int x = s.x;
        int y = s.y;
        int z = s.z;
        int t = s.t;
        // move index depending on direction
        if (mu_steps > 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (mu_steps < 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        return sitexyzt(x, y, z, t);
    }

    /// -------------------------------------------------------------------------------------- site_move: TWO DIRECTIONS
    template<int mu_steps, int nu_steps> __device__ __host__ inline static gSite site_move(const gSite &s, const int mu, const int nu) {
        sitexyzt tmp = site_move<mu_steps, nu_steps>(s.coordFull, mu, nu);
        return getSiteFull(tmp.x, tmp.y, tmp.z, tmp.t);
    }
    template<int mu_steps, int nu_steps> __device__ __host__ inline static gSiteMu site_move(const gSiteMu &s, const int mu, const int nu) {
        sitexyzt tmp = site_move<mu_steps, nu_steps>(s.coordFull, mu, nu);
        return getSiteMuFull(tmp.x, tmp.y, tmp.z, tmp.t, s.mu);
    }
    template<int mu_steps, int nu_steps> __device__ __host__ inline static gSiteStack site_move(const gSiteStack &s, const int mu, const int nu) {
        sitexyzt tmp = site_move<mu_steps, nu_steps>(s.coordFull, mu, nu);
        return getSiteStackFull(tmp.x, tmp.y, tmp.z, tmp.t, s.stack);
    }
    template<int mu_steps, int nu_steps> __device__ __host__ inline static sitexyzt site_move(const sitexyzt &s, const int mu, const int nu) {
        int x = s.x;
        int y = s.y;
        int z = s.z;
        int t = s.t;
        // move index depending on direction
        if (mu_steps > 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (mu_steps < 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps > 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps < 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        return sitexyzt(x, y, z, t);
    }

    /// ------------------------------------------------------------------------------------ site_move: THREE DIRECTIONS
    template<int mu_steps, int nu_steps, int rho_steps>
    __device__ __host__ inline static gSite site_move(const gSite &s, const int mu, const int nu, const int rho) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps>(s.coordFull, mu, nu, rho);
        return getSiteFull(tmp.x, tmp.y, tmp.z, tmp.t);
    }
    template<int mu_steps, int nu_steps, int rho_steps>
    __device__ __host__ inline static gSiteMu site_move(const gSiteMu &s, const int mu, const int nu, const int rho) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps>(s.coordFull, mu, nu, rho);
        return getSiteMuFull(tmp.x, tmp.y, tmp.z, tmp.t, s.mu);
    }
    template<int mu_steps, int nu_steps, int rho_steps>
    __device__ __host__ inline static gSiteStack site_move(const gSiteStack &s, const int mu, const int nu, const int rho) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps>(s.coordFull, mu, nu, rho);
        return getSiteStackFull(tmp.x, tmp.y, tmp.z, tmp.t, s.stack);
    }
    template<int mu_steps, int nu_steps, int rho_steps>
    __device__ __host__ inline static sitexyzt site_move(const sitexyzt &s, const int mu, const int nu, const int rho) {
        int x = s.x;
        int y = s.y;
        int z = s.z;
        int t = s.t;
        // move index depending on direction
        if (mu_steps > 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (mu_steps < 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps > 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps < 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (rho_steps > 0) {
            switch (rho) {
                case 0:
                    //                      x = (x+rho_steps) % size.lx();
                    x = x + rho_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+rho_steps) % size.ly();
                    y = y + rho_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+rho_steps) % size.lz();
                    z = z + rho_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+rho_steps) % size.lt();
                    t = t + rho_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (rho_steps < 0) {
            switch (rho) {
                case 0:
                    //                      x = (x+rho_steps) % size.lx();
                    x = x + rho_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+rho_steps) % size.ly();
                    y = y + rho_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+rho_steps) % size.lz();
                    z = z + rho_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+rho_steps) % size.lt();
                    t = t + rho_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        return sitexyzt(x, y, z, t);
    }

    /// ------------------------------------------------------------------------------------- site_move: FOUR DIRECTIONS
    template<int mu_steps, int nu_steps, int rho_steps, int sig_steps>
    __device__ __host__ inline static gSite site_move(const gSite &s, const int mu, const int nu, const int rho, const int sig) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps, sig_steps>(s.coordFull, mu, nu, rho, sig);
        return getSiteFull(tmp.x, tmp.y, tmp.z, tmp.t);
    }
    template<int mu_steps, int nu_steps, int rho_steps, int sig_steps>
    __device__ __host__ inline static gSiteMu site_move(const gSiteMu &s, const int mu, const int nu, const int rho, const int sig) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps, sig_steps>(s.coordFull, mu, nu, rho, sig);
        return getSiteMuFull(tmp.x, tmp.y, tmp.z, tmp.t, s.mu);
    }
    template<int mu_steps, int nu_steps, int rho_steps, int sig_steps>
    __device__ __host__ inline static gSiteStack site_move(const gSiteStack &s, const int mu, const int nu, const int rho, const int sig) {
        sitexyzt tmp = site_move<mu_steps, nu_steps, rho_steps, sig_steps>(s.coordFull, mu, nu, rho, sig);
        return getSiteStackFull(tmp.x, tmp.y, tmp.z, tmp.t, s.stack);
    }
    template<int mu_steps, int nu_steps, int rho_steps, int sig_steps>
    __device__ __host__ inline static sitexyzt site_move(const sitexyzt &s, const int mu, const int nu, const int rho, const int sig) {
        int x = s.x;
        int y = s.y;
        int z = s.z;
        int t = s.t;
        // move index depending on direction
        if (mu_steps > 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (mu_steps < 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps > 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (nu_steps < 0) {
            switch (nu) {
                case 0:
                    //                      x = (x+nu_steps) % size.lx();
                    x = x + nu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+nu_steps) % size.ly();
                    y = y + nu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+nu_steps) % size.lz();
                    z = z + nu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+nu_steps) % size.lt();
                    t = t + nu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (rho_steps > 0) {
            switch (rho) {
                case 0:
                    //                      x = (x+rho_steps) % size.lx();
                    x = x + rho_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+rho_steps) % size.ly();
                    y = y + rho_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+rho_steps) % size.lz();
                    z = z + rho_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+rho_steps) % size.lt();
                    t = t + rho_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (rho_steps < 0) {
            switch (rho) {
                case 0:
                    //                      x = (x+rho_steps) % size.lx();
                    x = x + rho_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+rho_steps) % size.ly();
                    y = y + rho_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+rho_steps) % size.lz();
                    z = z + rho_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+rho_steps) % size.lt();
                    t = t + rho_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        if (sig_steps > 0) {
            switch (sig) {
                case 0:
                    //                      x = (x+sig_steps) % size.lx();
                    x = x + sig_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+sig_steps) % size.ly();
                    y = y + sig_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+sig_steps) % size.lz();
                    z = z + sig_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+sig_steps) % size.lt();
                    t = t + sig_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (sig_steps < 0) {
            switch (sig) {
                case 0:
                    //                      x = (x+sig_steps) % size.lx();
                    x = x + sig_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+sig_steps) % size.ly();
                    y = y + sig_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+sig_steps) % size.lz();
                    z = z + sig_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+sig_steps) % size.lt();
                    t = t + sig_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        return sitexyzt(x, y, z, t);
    }

/// ------------------------------------------------------------------------------------------------ site_up and site_dn
    template<class T> __device__ __host__ inline static T site_up(const T &s, const int mu) {
        return site_move<1>(s, mu);
    }
    template<class T> __device__ __host__ inline static T site_dn(const T &s, const int mu) {
        return site_move<-1>(s, mu);
    }
    template<class T> __device__ __host__ inline static T site_up_up(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<1, 1>(s, mu, nu);
#else
        return site_up(site_up(s, mu), nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_dn(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<1, -1>(s, mu, nu);
#else
        return site_dn(site_up(s, mu), nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_dn_dn(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<-1, -1>(s, mu, nu);
#else
        return site_dn(site_dn(s, mu), nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_up_up(const T &s, const int mu, const int nu, const int rho) {
#ifdef __GPU_ARCH__
        return site_move<1, 1, 1>(s, mu, nu, rho);
#else
        return site_up(site_up_up(s, mu, nu), rho);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_up_dn(const T &s, const int mu, const int nu, const int rho) {
#ifdef __GPU_ARCH__
        return site_move<1, 1, -1>(s, mu, nu, rho);
#else
        return site_dn(site_up_up(s, mu, nu), rho);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_dn_dn(const T &s, const int mu, const int nu, const int rho) {
#ifdef __GPU_ARCH__
        return site_move<1, -1, -1>(s, mu, nu, rho);
#else
        return site_dn(site_up_dn(s, mu, nu), rho);
#endif
    }
    template<class T> __device__ __host__ inline static T site_dn_dn_dn(const T &s, const int mu, const int nu, const int rho) {
#ifdef __GPU_ARCH__
        return site_move<-1, -1, -1>(s, mu, nu, rho);
#else
        return site_dn(site_dn_dn(s, mu, nu), rho);
#endif
    }
    //! The following are currently unused but can be commented in if needed:
    template<class T> __device__ __host__ inline static T site_up_up_up_up(const T &s, const int mu, const int nu, const int rho, const int sig) {
#ifdef __GPU_ARCH__
        return site_move<1, 1, 1, 1>(s, mu, nu, rho, sig);
#else
        return site_up(site_up_up_up(s, mu, nu, rho), sig);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_up_up_dn(const T &s, const int mu, const int nu, const int rho, const int sig) {
#ifdef __GPU_ARCH__
        return site_move<1, 1, 1, -1>(s, mu, nu, rho, sig);
#else
        return site_dn(site_up_up_up(s, mu, nu, rho), sig);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_up_dn_dn(const T &s, const int mu, const int nu, const int rho, const int sig) {
#ifdef __GPU_ARCH__
        return site_move<1, 1, -1, -1>(s, mu, nu, rho, sig);
#else
        return site_dn(site_up_up_dn(s, mu, nu, rho), sig);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_dn_dn_dn(const T &s, const int mu, const int nu, const int rho, const int sig) {
#ifdef __GPU_ARCH__
        return site_move<1, -1, -1, -1>(s, mu, nu, rho, sig);
#else
        return site_dn(site_up_dn_dn(s, mu, nu, rho), sig);
#endif
    }
    template<class T> __device__ __host__ inline static T site_dn_dn_dn_dn(const T &s, const int mu, const int nu, const int rho, const int sig) {
#ifdef __GPU_ARCH__
        return site_move<-1, -1, -1, -1>(s, mu, nu, rho, sig);
#else
        return site_dn(site_dn_dn_dn(s, mu, nu, rho), sig);
#endif
    }
    template<class T> __device__ __host__ inline static T site_2up_up(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<2, 1>(s, mu, nu);
#else
        return site_up_up_up(s, mu, mu, nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_2up_dn(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<2, -1>(s, mu, nu);
#else
        return site_up_up_dn(s, mu, mu, nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_up_2dn(const T &s, const int mu, const int nu) {
#ifdef __GPU_ARCH__
        return site_move<1, -2>(s, mu, nu);
#else
        return site_up_dn_dn(s, mu, mu, nu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_2up(const T &s, const int mu) {
#ifdef __GPU_ARCH__
        return site_move<2>(s, mu);
#else
        return site_up_up(s, mu, mu);
#endif
    }
    template<class T> __device__ __host__ inline static T site_2dn(const T &s, const int mu) {
#ifdef __GPU_ARCH__
        return site_move<-2>(s, mu);
#else
        return site_dn_dn(s, mu, mu);
#endif

    }
/// ------------------------------------------------------------------------------------------- SITE MOVE (RUN TIME)
    //! Unlike the above implementation of site_move, this can be used in a for loop. Presumably it is slower?
    //! Currently unused but can be commented in if needed:

    __device__ __host__ inline static sitexyzt dynamic_move(sitexyzt s, const int mu, int mu_steps) {
        int x = s.x;
        int y = s.y;
        int z = s.z;
        int t = s.t;
        if (mu_steps > 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x >= (int)getLatData().lxFull)
                        x -= getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y >= (int)getLatData().lyFull)
                        y -= getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z >= (int)getLatData().lzFull)
                        z -= getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t >= (int)getLatData().ltFull)
                        t -= getLatData().ltFull;
                    break;
            }
        }
        if (mu_steps < 0) {
            switch (mu) {
                case 0:
                    //                      x = (x+mu_steps) % size.lx();
                    x = x + mu_steps;
                    if (x < 0)
                        x += getLatData().lxFull;
                    break;
                case 1:
                    //                      y = (y+mu_steps) % size.ly();
                    y = y + mu_steps;
                    if (y < 0)
                        y += getLatData().lyFull;
                    break;
                case 2:
                    //                      z = (z+mu_steps) % size.lz();
                    z = z + mu_steps;
                    if (z < 0)
                        z += getLatData().lzFull;
                    break;
                case 3:
                    //                      t = (t+mu_steps) % size.lt();
                    t = t + mu_steps;
                    if (t < 0)
                        t += getLatData().ltFull;
                    break;
            }
        }
        return sitexyzt(x, y, z, t);
    }
    __attribute__((unused)) __device__ __host__ inline static gSite dynamic_move(const gSite &s, const int mu, int mu_steps) {
        sitexyzt tmp = dynamic_move(s.coordFull, mu, mu_steps);
        return getSiteFull(tmp.x, tmp.y, tmp.z, tmp.t);
    }

};

#endif //INDEXERDEVICE
