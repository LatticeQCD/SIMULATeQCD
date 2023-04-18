/*
 * HaloIndexer.h
 *
 * L. Mazur
 *
 */

#ifndef HALOINDEXER_H
#define HALOINDEXER_H

#include "BulkIndexer.h"
#include <vector>
#include <string>
#include "../../define.h"

/*
        Offsets define as:

HaloType    T Z Y X
	        0 0 0 0         (everything local)
====================================================
0   	    0 0 0 1	        (X, offset 2 hplanes LY*LZ*LT )
1   	    0 0 1 0         (Y, offset 2 hplanes LX*LZ*LT )
2   	    0 1 0 0         (Z, offset 2 hplanes LX*LY*LT )
3   	    1 0 0 0         (T, offset 2 hplanes LX*LY*LZ )
====================================================
4   	    0 0 1 1         (XY, offset 4 planes LZ*LT )
5   	    0 1 0 1         (XZ, offset 4 planes LY*LT )
6   	    1 0 0 1         (XT, offset 4 planes LY*LZ )

7   	    0 1 1 0         (YZ, offset 4 planes LX*LT )
8   	    1 0 1 0         (YT, offset 4 planes LX*LZ )

9     	    1 1 0 0         (ZT, offset 4 planes LX*LY )
====================================================
10  	    0 1 1 1         (XYZ,  offset 8 length LT )
11  	    1 0 1 1         (XYT,  offset 8 length LZ )
12  	    1 1 0 1         (XZT,  offset 8 length LY )
13  	    1 1 1 0         (YZT,  offset 8 length LX )
====================================================
14  	    1 1 1 1         (XYZT, offset 16 lattice corners)


Each HaloType consists of several sub-Halos. For example we have one HaloType
in X-direction with two sub-Halos (HyperPlanes).

right=1
left=0
        T Z Y X
0       0 0 0 0
1       0 0 0 1
2       0 0 1 0
3       0 0 1 1
4       0 1 0 0
5       0 1 0 1
6       0 1 1 0
7       0 1 1 1
8       1 0 0 0
9       1 0 0 1
10      1 0 1 0
11      1 0 1 1
12      1 1 0 0
13      1 1 0 1
14      1 1 1 0
15      1 1 1 1


*/




struct HaloData {
public:
    size_t h_LXi, h_LYi, h_LZi, h_LTi, h_HaloDepth[4], h_2HaloDepth[4];
    size_t h_XY, h_XZ, h_XT, h_YZ, h_YT, h_ZT, h_HH;
    size_t h_XYZ, h_XYT, h_XZT, h_YZT, h_HHH;
    size_t h_HHHH;

    size_t h_XH, h_YH, h_ZH, h_TH;
    size_t h_XHH, h_YHH, h_ZHH, h_THH;
    size_t h_XHHH, h_YHHH, h_ZHHH, h_THHH;

    size_t h_XYH, h_XZH, h_XTH, h_YZH, h_YTH, h_ZTH;
    size_t h_XYZH, h_XYTH, h_XZTH, h_YZTH;

    size_t h_XYHH, h_XZHH, h_XTHH, h_YZHH, h_YTHH, h_ZTHH;

    size_t h_LX_mH, h_LY_mH, h_LZ_mH, h_LT_mH;  // Full dimension minus 1*Halo_Depth
    size_t h_LX_mH2, h_LY_mH2, h_LZ_mH2, h_LT_mH2;  // Full dimension minus 1*Halo_Depth
    size_t h_LX_2H, h_LY_2H, h_LZ_2H, h_LT_2H;  // Full dimension minus 2*Halo_Depth
    size_t h_vol1Center, h_vol2Center, h_vol3Center, h_vol4Center;
    // summed up sizes of Halo types. For example h_summed_buffer[1] = sizeof(HaloType 0) +  sizeof(Halotype 1)
    size_t h_summed_buffer[16];
    size_t h_summed_bufferHalf[16];
    // Offsets of each Sub-Halo
    size_t h_offsets[80];
    size_t h_offsetsHalf[80];


    __device__ __host__ HaloData() {}

    __device__ __host__ HaloData(size_t lx, size_t ly, size_t lz, size_t lt, size_t halo_depth, unsigned int Nodes[4]) {


        h_HaloDepth[0] = Nodes[0] != 1 ? halo_depth : 0;
        h_HaloDepth[1] = Nodes[1] != 1 ? halo_depth : 0;
        h_HaloDepth[2] = Nodes[2] != 1 ? halo_depth : 0;
        h_HaloDepth[3] = Nodes[3] != 1 ? halo_depth : 0;

        h_2HaloDepth[0] = 2 * h_HaloDepth[0];
        h_2HaloDepth[1] = 2 * h_HaloDepth[1];
        h_2HaloDepth[2] = 2 * h_HaloDepth[2];
        h_2HaloDepth[3] = 2 * h_HaloDepth[3];


        h_LXi = lx;
        h_LYi = ly;
        h_LZi = lz;
        h_LTi = lt;

        h_vol1Center = lx - h_2HaloDepth[0];
        h_vol2Center = h_vol1Center * (ly - h_2HaloDepth[1]);
        h_vol3Center = h_vol2Center * (lz - h_2HaloDepth[2]);
        h_vol4Center = h_vol3Center * (lt - h_2HaloDepth[3]);

        h_LX_mH = lx + h_HaloDepth[0];
        h_LY_mH = ly + h_HaloDepth[1];
        h_LZ_mH = lz + h_HaloDepth[2];
        h_LT_mH = lt + h_HaloDepth[3];

        h_LX_mH2 = lx - h_HaloDepth[0];
        h_LY_mH2 = ly - h_HaloDepth[1];
        h_LZ_mH2 = lz - h_HaloDepth[2];
        h_LT_mH2 = lt - h_HaloDepth[3];

        h_YZT = h_LYi * h_LZi * h_LTi;
        h_XZT = h_LXi * h_LZi * h_LTi;
        h_XYT = h_LXi * h_LYi * h_LTi;
        h_XYZ = h_LXi * h_LYi * h_LZi;

        h_YZTH = h_LYi * h_LZi * h_LTi * h_HaloDepth[0];
        h_XZTH = h_LXi * h_LZi * h_LTi * h_HaloDepth[1];
        h_XYTH = h_LXi * h_LYi * h_LTi * h_HaloDepth[2];
        h_XYZH = h_LXi * h_LYi * h_LZi * h_HaloDepth[3];

        h_ZT = h_LZi * h_LTi;
        h_YT = h_LYi * h_LTi;
        h_YZ = h_LYi * h_LZi;
        h_XT = h_LXi * h_LTi;
        h_XZ = h_LXi * h_LZi;
        h_XY = h_LXi * h_LYi;

        h_ZTH = h_LZi * h_LTi * h_HaloDepth[0];
        h_YTH = h_LYi * h_LTi * h_HaloDepth[0];
        h_YZH = h_LYi * h_LZi * h_HaloDepth[0];
        h_XTH = h_LXi * h_LTi * h_HaloDepth[1];
        h_XZH = h_LXi * h_LZi * h_HaloDepth[1];
        h_XYH = h_LXi * h_LYi * h_HaloDepth[2];

        h_ZTHH = h_LZi * h_LTi * h_HaloDepth[0] * h_HaloDepth[1];
        h_YTHH = h_LYi * h_LTi * h_HaloDepth[0] * h_HaloDepth[2];
        h_YZHH = h_LYi * h_LZi * h_HaloDepth[0] * h_HaloDepth[3];
        h_XTHH = h_LXi * h_LTi * h_HaloDepth[1] * h_HaloDepth[2];
        h_XZHH = h_LXi * h_LZi * h_HaloDepth[1] * h_HaloDepth[3];
        h_XYHH = h_LXi * h_LYi * h_HaloDepth[2] * h_HaloDepth[3];

        h_TH = h_LTi * h_HaloDepth[0];
        h_ZH = h_LZi * h_HaloDepth[0];
        h_YH = h_LYi * h_HaloDepth[0];
        h_XH = h_LXi * h_HaloDepth[1];
        h_HH = h_HaloDepth[0] * h_HaloDepth[1];

        h_THH = h_LTi * h_HaloDepth[0] * h_HaloDepth[1];
        h_ZHH = h_LZi * h_HaloDepth[0] * h_HaloDepth[1];
        h_YHH = h_LYi * h_HaloDepth[0] * h_HaloDepth[2];
        h_XHH = h_LXi * h_HaloDepth[1] * h_HaloDepth[2];
        h_HHH = h_HaloDepth[0] * h_HaloDepth[1] * h_HaloDepth[2];

        h_THHH = h_LTi * h_HaloDepth[0] * h_HaloDepth[1] * h_HaloDepth[2];
        h_ZHHH = h_LZi * h_HaloDepth[0] * h_HaloDepth[1] * h_HaloDepth[3];
        h_YHHH = h_LYi * h_HaloDepth[0] * h_HaloDepth[2] * h_HaloDepth[3];
        h_XHHH = h_LXi * h_HaloDepth[1] * h_HaloDepth[2] * h_HaloDepth[3];
        h_HHHH = h_HaloDepth[0] * h_HaloDepth[1] * h_HaloDepth[2] * h_HaloDepth[3];


        // Fill h_summed_buffer
        for (int number = 0; number < 16; number++) {
            size_t off = 0;
            for (int i = 0; (i < number) && (i <= 3); i++) off += 2 * get_SubHaloSizeFromType(i);
            for (int i = 4; (i < number) && (i <= 9); i++) off += 4 * get_SubHaloSizeFromType(i);
            for (int i = 10; (i < number) && (i <= 13); i++) off += 8 * get_SubHaloSizeFromType(i);
            for (int i = 14; (i < number) && (i <= 14); i++) off += 16 * get_SubHaloSizeFromType(i);

            h_summed_buffer[number] = off;
            h_summed_bufferHalf[number] = off / 2;
        }


        // Fill h_offsets
        h_offsets[0] = 0;
        h_offsetsHalf[0] = 0;
        for (int i = 1; i < 80; i++) {
            h_offsets[i] = h_offsets[i - 1] + get_SubHaloSize(i - 1, All);
            h_offsetsHalf[i] = h_offsetsHalf[i - 1] + get_SubHaloSize(i - 1, Even);
        }
    }


    __device__ __host__ size_t getBufferSize(Layout LatLayout) {
        if (LatLayout == All)return h_summed_buffer[15];
        else return h_summed_bufferHalf[15];
    }


    /// For each Halotype we have different sub-Halos with the same size.
    /// This function returns the size of these sub_Halos.

    /// The argument is the number of the Sub-Halo!
    __device__ __host__ inline size_t get_SubHaloSize(const short number, Layout LatLayout) const {

        size_t EvenFactor = 1;
        if (LatLayout != All) EvenFactor = 2;
        if (number >= 0 && number < 2) return get_SubHaloSizeFromType(0) / EvenFactor;
        if (number >= 2 && number < 4) return get_SubHaloSizeFromType(1) / EvenFactor;
        if (number >= 4 && number < 6) return get_SubHaloSizeFromType(2) / EvenFactor;
        if (number >= 6 && number < 8) return get_SubHaloSizeFromType(3) / EvenFactor;

        if (number >= 8 && number < 12) return get_SubHaloSizeFromType(4) / EvenFactor;
        if (number >= 12 && number < 16) return get_SubHaloSizeFromType(5) / EvenFactor;
        if (number >= 16 && number < 20) return get_SubHaloSizeFromType(6) / EvenFactor;
        if (number >= 20 && number < 24) return get_SubHaloSizeFromType(7) / EvenFactor;
        if (number >= 24 && number < 28) return get_SubHaloSizeFromType(8) / EvenFactor;
        if (number >= 28 && number < 32) return get_SubHaloSizeFromType(9) / EvenFactor;

        if (number >= 32 && number < 40) return get_SubHaloSizeFromType(10) / EvenFactor;
        if (number >= 40 && number < 48) return get_SubHaloSizeFromType(11) / EvenFactor;
        if (number >= 48 && number < 56) return get_SubHaloSizeFromType(12) / EvenFactor;
        if (number >= 56 && number < 64) return get_SubHaloSizeFromType(13) / EvenFactor;

        if (number >= 64 && number < 80) return get_SubHaloSizeFromType(14) / EvenFactor;
        return 0xAAFFFFFF;
    }

private:
    /// The argument is the number of the Halo Type! It returns the size of an All Halo Type!
    __device__ __host__ inline size_t get_SubHaloSizeFromType(const short number) const {
        if (number == 0) return h_YZTH;
        if (number == 1) return h_XZTH;
        if (number == 2) return h_XYTH;
        if (number == 3) return h_XYZH;

        if (number == 4) return h_ZTHH;
        if (number == 5) return h_YTHH;
        if (number == 6) return h_YZHH;
        if (number == 7) return h_XTHH;
        if (number == 8) return h_XZHH;
        if (number == 9) return h_XYHH;

        if (number == 10) return h_THHH;
        if (number == 11) return h_ZHHH;
        if (number == 12) return h_YHHH;
        if (number == 13) return h_XHHH;

        if (number == 14) return h_HHHH;
        return 0xAAFFFFFF;
    }


};


extern __device__ __constant__ struct HaloData globHalDataGPU[MAXHALO + 1];
extern __device__ __constant__ struct HaloData globHalDataGPUReduced[MAXHALO + 1];
extern struct HaloData globHalDataCPU[MAXHALO + 1];
extern struct HaloData globHalDataCPUReduced[MAXHALO + 1];

void initGPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt);

void initCPUHaloIndexer(size_t lx, size_t ly, size_t lz, size_t lt);

template<Layout LatLayout, size_t HaloDepth>
class HaloIndexer {

private:

    __device__ __host__ inline static size_t _getHaloNumber(size_t index, size_t *LocHalIndex) {
        if (LatLayout == All) {
            for (int i = 1; i < 80; i++) {
                if (getHalData().h_offsets[i] > index) {
                    *LocHalIndex = index - getHalData().h_offsets[i - 1];
                    return i - 1;
                }
            }
            if (getBufferSize() > index) {
                *LocHalIndex = index - getHalData().h_offsets[79];
                return 79;
            }
        } else {

            for (int i = 1; i < 80; i++) {
                if (getHalData().h_offsetsHalf[i] > index) {
                    *LocHalIndex = index - getHalData().h_offsetsHalf[i - 1];
                    return i - 1;
                }
            }
            if (getBufferSize() > index) {
                *LocHalIndex = index - getHalData().h_offsetsHalf[79];
                return 79;
            }
        }
        return 0;
    };

    __device__ __host__ inline static size_t _getHaloNumberReduced(size_t index, size_t *LocHalIndex) {
        if (LatLayout == All) {
            for (int i = 1; i < 80; i++) {
                if (getHalDataReduced().h_offsets[i] > index) {
                    *LocHalIndex = index - getHalDataReduced().h_offsets[i - 1];
                    return i - 1;
                }
            }
            if (getBufferSize() > index) {
                *LocHalIndex = index - getHalDataReduced().h_offsets[79];
                return 79;
            }
        } else {

            for (int i = 1; i < 80; i++) {
                if (getHalDataReduced().h_offsetsHalf[i] > index) {
                    *LocHalIndex = index - getHalDataReduced().h_offsetsHalf[i - 1];
                    return i - 1;
                }
            }
            if (getBufferSize() > index) {
                *LocHalIndex = index - getHalDataReduced().h_offsetsHalf[79];
                return 79;
            }
        }
        return 0;
    };

public:
    __device__ __host__ HaloIndexer();

    __device__ __host__ ~HaloIndexer() {};

    __device__ __host__ inline static HaloData getHalData() {
#ifdef __GPU_ARCH__
        return globHalDataGPU[HaloDepth];
#else
        return globHalDataCPU[HaloDepth];
#endif
    }

    __device__ __host__ inline static HaloData getHalDataReduced() {
#ifdef __GPU_ARCH__
        return globHalDataGPUReduced[HaloDepth];
#else
        return globHalDataCPUReduced[HaloDepth];
#endif
    }

    __device__ __host__ inline static size_t getBufferSize() {
        if (LatLayout == All)return getHalData().h_summed_buffer[15];
        else return getHalData().h_summed_bufferHalf[15];

    }


    __device__ __host__ inline static size_t get_SubHaloOffset(const short number) {

        if (LatLayout == All)return getHalData().h_offsets[number];
        else return getHalData().h_offsetsHalf[number];

    }

    __device__ __host__ inline static size_t get_SubHaloSize(const short number) {
        return getHalData().get_SubHaloSize(number, LatLayout);

    }

    __device__ __host__ inline static size_t get_ReducedSubHaloSize(const short number) {
        return getHalDataReduced().get_SubHaloSize(number, LatLayout);

    }

    __device__ __host__ inline static void getCoord_eo(size_t &x, size_t &y, size_t &z, size_t &t,
                                                       const size_t index,
                                                       const size_t vol1, const size_t vol2, const size_t vol3,
                                                       const bool par) {

        size_t normInd;
        size_t tmp;
        normInd = index << 0x1;
        // get x,y,z,t
        divmod(normInd, vol3, t, tmp);
        divmod(tmp, vol2, z, tmp);
        divmod(tmp, vol1, y, x);

        // correct effect of divison by two (adjacent odd and even numbers mapped to same number)
        if (par && !isOdd(x) && !isOdd(y + z + t))
            ++x;
        if (!par && !isOdd(x) && isOdd(y + z + t))
            ++x;
    }

    __device__ __host__ inline static void getCoord(size_t &x, size_t &y, size_t &z, size_t &t,
                                                    const size_t index,
                                                    const size_t vol1, const size_t vol2, const size_t vol3) {

        if (LatLayout == All) {
            size_t tmp;

            divmod(index, vol3, t, tmp);
            divmod(tmp, vol2, z, tmp);
            divmod(tmp, vol1, y, x);
        } else if (LatLayout == Even)
            getCoord_eo(x, y, z, t, index, vol1, vol2, vol3, 0);
        else if (LatLayout == Odd)
            getCoord_eo(x, y, z, t, index, vol1, vol2, vol3, 1);


    }


    __device__ __host__ inline static void
    getHypPlanePos(size_t number, size_t &pos_a, size_t &pos_b) {
        pos_a = number * 2;
        pos_b = number * 2 + 1;
    }

    __device__ __host__ inline static void
    getPlanePos(size_t number, size_t dir, size_t &pos_a, size_t &pos_b) {
        number -= 4;
        pos_a = 8 + number * 4 + dir;
        pos_b = 8 + number * 4 + dir + (3 - 2 * dir);
    }

    __device__ __host__ inline static void
    getStripePos(size_t number, size_t dir, size_t &pos_a, size_t &pos_b) {

        number -= 10;
        pos_a = 32 + number * 8 + dir;
        pos_b = 32 + number * 8 + dir + (7 - 2 * dir);
    }

    __device__ __host__ inline static void
    getCornerPos(size_t number, size_t dir, size_t &pos_a, size_t &pos_b) {

        number -= 14;
        pos_a = 64 + number * 16 + dir;
        pos_b = 64 + number * 16 + dir + (15 - 2 * dir);
    }


    __device__ __host__ inline static HaloSegment mapIntToHSeg(int bits) {
        if (bits == 1) return X;
        if (bits == 2) return Y;
        if (bits == 4) return Z;
        if (bits == 8) return T;

        if (bits == 3) return XY;
        if (bits == 5) return XZ;
        if (bits == 9) return XT;
        if (bits == 6) return YZ;
        if (bits == 10) return YT;
        if (bits == 12) return ZT;


        if (bits == 7) return XYZ;
        if (bits == 11) return XYT;
        if (bits == 13) return XZT;
        if (bits == 14) return YZT;

        if (bits == 15) return XYZT;
        printf("mapIntToHSeg: wrong bits...");
        return X;
    }

    __device__ __host__ inline static HaloSegment getHSeg(sitexyzt coord) {

        int bits = 0;

        if ((coord[0] < (int) getHalData().h_HaloDepth[0]) || (coord[0] >= (int) getHalData().h_LX_mH)) {
            bits = bits | 1;
        }
        if ((coord[1] < (int) getHalData().h_HaloDepth[1]) || (coord[1] >= (int) getHalData().h_LY_mH)) {
            bits = bits | 2;
        }
        if ((coord[2] < (int) getHalData().h_HaloDepth[2]) || (coord[2] >= (int) getHalData().h_LZ_mH)) {
            bits = bits | 3;
        }
        if ((coord[3] < (int) getHalData().h_HaloDepth[3]) || (coord[3] >= (int) getHalData().h_LT_mH)) {
            bits = bits | 4;
        }
        return mapIntToHSeg(bits);
    }

    __device__ __host__ inline static short getlr(sitexyzt coord) {
        short lr = 0;
        HaloSegment hseg = getHSeg(coord);

        if (hseg == 0) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
        } else if (hseg == 1) {
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 1);
        } else if (hseg == 2) {
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 1);
        } else if (hseg == 3) {
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 1);
        } else if (hseg == 4) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 2);
        } else if (hseg == 5) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 2);
        } else if (hseg == 6) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 2);
        } else if (hseg == 7) {
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 1);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 2);
        } else if (hseg == 8) {
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 1);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 2);
        } else if (hseg == 9) {
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 1);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 2);
        } else if (hseg == 10) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 2);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 4);
        } else if (hseg == 11) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 2);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 4);
        } else if (hseg == 12) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 2);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 4);
        } else if (hseg == 13) {
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 1);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 2);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 4);
        } else if (hseg == 14) {
            lr = lr | ((coord[0] >= (int) getHalData().h_LX_mH) * 1);
            lr = lr | ((coord[1] >= (int) getHalData().h_LY_mH) * 2);
            lr = lr | ((coord[2] >= (int) getHalData().h_LZ_mH) * 4);
            lr = lr | ((coord[3] >= (int) getHalData().h_LT_mH) * 8);
        }
        return lr;
    }


    short getDir(sitexyzt coord) {
        return getlr(coord) / 2;
    }


    __device__ __host__ inline static size_t getOuterHaloSize() {
        return getHalData().getBufferSize(LatLayout);
    }

    __device__ __host__ inline static size_t getInnerHaloSize() {
        return getHalDataReduced().getBufferSize(LatLayout);
    }

    __device__ __host__ inline static size_t getCenterSize() {
        return GIndexer<LatLayout, HaloDepth>::getLatData().vol4 - getInnerHaloSize();
    }

/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++ Translate Halo-Buffer Index to Lattice coordinates ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++                for INNER Halos                     ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///    It translates an continuous Index < (size of inner Halo Part (I)) into the corresponding coordinate in I
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord(size_t HalIndex, size_t &HalNumber, size_t &LocHalIndex) {


        HalNumber = _getHaloNumber(HalIndex, &LocHalIndex);

        size_t left_right = 0;

        /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
        if (HalNumber < 2) {
            left_right = HalNumber;
            return getInnerHaloCoord_Hyperplane_X(LocHalIndex, left_right);
        } else if (HalNumber >= 2 && HalNumber < 4) {
            left_right = HalNumber - 2;
            return getInnerHaloCoord_Hyperplane_Y(LocHalIndex, left_right);
        } else if (HalNumber >= 4 && HalNumber < 6) {
            left_right = HalNumber - 4;
            return getInnerHaloCoord_Hyperplane_Z(LocHalIndex, left_right);
        } else if (HalNumber >= 6 && HalNumber < 8) {
            left_right = HalNumber - 6;
            return getInnerHaloCoord_Hyperplane_T(LocHalIndex, left_right);
        }



            /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 8 && HalNumber < 12) {
            left_right = HalNumber - 8;
            return getInnerHaloCoord_Plane_XY(LocHalIndex, left_right);
        } else if (HalNumber >= 12 && HalNumber < 16) {
            left_right = HalNumber - 12;
            return getInnerHaloCoord_Plane_XZ(LocHalIndex, left_right);
        } else if (HalNumber >= 16 && HalNumber < 20) {
            left_right = HalNumber - 16;
            return getInnerHaloCoord_Plane_XT(LocHalIndex, left_right);
        } else if (HalNumber >= 20 && HalNumber < 24) {
            left_right = HalNumber - 20;
            return getInnerHaloCoord_Plane_YZ(LocHalIndex, left_right);
        } else if (HalNumber >= 24 && HalNumber < 28) {
            left_right = HalNumber - 24;
            return getInnerHaloCoord_Plane_YT(LocHalIndex, left_right);
        } else if (HalNumber >= 28 && HalNumber < 32) {
            left_right = HalNumber - 28;
            return getInnerHaloCoord_Plane_ZT(LocHalIndex, left_right);
        }



            /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 32 && HalNumber < 40) {
            left_right = HalNumber - 32;
            return getInnerHaloCoord_Stripe_XYZ(LocHalIndex, left_right);
        } else if (HalNumber >= 40 && HalNumber < 48) {
            left_right = HalNumber - 40;
            return getInnerHaloCoord_Stripe_XYT(LocHalIndex, left_right);
        } else if (HalNumber >= 48 && HalNumber < 56) {
            left_right = HalNumber - 48;
            return getInnerHaloCoord_Stripe_XZT(LocHalIndex, left_right);
        } else if (HalNumber >= 56 && HalNumber < 64) {
            left_right = HalNumber - 56;
            return getInnerHaloCoord_Stripe_YZT(LocHalIndex, left_right);
        }




            /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 64 && HalNumber < 80) {
            left_right = HalNumber - 64;
            return getInnerHaloCoord_Corner(LocHalIndex, left_right);
        }

        printf("ERROR: getInnerHaloCoord");
        return sitexyzt(-99, -99, -99, -99);
    }


/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++ Translate Halo-Buffer Index to Lattice coordinates ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++                for OUTER Halos                     ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///    It translates an continuous Index < (size of outer Halo Part (O)) into the corresponding coordinate in O
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord(size_t HalIndex, size_t &HalNumber, size_t &LocHalIndex) {

        HalNumber = _getHaloNumber(HalIndex, &LocHalIndex);

        size_t left_right = 0;


        /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
        if (HalNumber < 2) {
            left_right = HalNumber;
            return getOuterHaloCoord_Hyperplane_X(LocHalIndex, left_right);
        } else if (HalNumber >= 2 && HalNumber < 4) {
            left_right = HalNumber - 2;
            return getOuterHaloCoord_Hyperplane_Y(LocHalIndex, left_right);
        } else if (HalNumber >= 4 && HalNumber < 6) {
            left_right = HalNumber - 4;
            return getOuterHaloCoord_Hyperplane_Z(LocHalIndex, left_right);
        } else if (HalNumber >= 6 && HalNumber < 8) {
            left_right = HalNumber - 6;
            return getOuterHaloCoord_Hyperplane_T(LocHalIndex, left_right);
        }


        /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
        if (HalNumber >= 8 && HalNumber < 12) {
            left_right = HalNumber - 8;
            return getOuterHaloCoord_Plane_XY(LocHalIndex, left_right);
        } else if (HalNumber >= 12 && HalNumber < 16) {
            left_right = HalNumber - 12;
            return getOuterHaloCoord_Plane_XZ(LocHalIndex, left_right);
        } else if (HalNumber >= 16 && HalNumber < 20) {
            left_right = HalNumber - 16;
            return getOuterHaloCoord_Plane_XT(LocHalIndex, left_right);
        } else if (HalNumber >= 20 && HalNumber < 24) {
            left_right = HalNumber - 20;
            return getOuterHaloCoord_Plane_YZ(LocHalIndex, left_right);
        } else if (HalNumber >= 24 && HalNumber < 28) {
            left_right = HalNumber - 24;
            return getOuterHaloCoord_Plane_YT(LocHalIndex, left_right);
        } else if (HalNumber >= 28 && HalNumber < 32) {
            left_right = HalNumber - 28;
            return getOuterHaloCoord_Plane_ZT(LocHalIndex, left_right);
        }

            /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 32 && HalNumber < 40) {
            left_right = HalNumber - 32;
            return getOuterHaloCoord_Stripe_XYZ(LocHalIndex, left_right);
        } else if (HalNumber >= 40 && HalNumber < 48) {
            left_right = HalNumber - 40;
            return getOuterHaloCoord_Stripe_XYT(LocHalIndex, left_right);
        } else if (HalNumber >= 48 && HalNumber < 56) {
            left_right = HalNumber - 48;
            return getOuterHaloCoord_Stripe_XZT(LocHalIndex, left_right);
        } else if (HalNumber >= 56 && HalNumber < 64) {
            left_right = HalNumber - 56;
            return getOuterHaloCoord_Stripe_YZT(LocHalIndex, left_right);
        }




            /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 64 && HalNumber < 80) {
            left_right = HalNumber - 64;
            return getOuterHaloCoord_Corner(LocHalIndex, left_right);
        }

        printf("ERROR: getOuterHaloCoord");

        return sitexyzt(-99, -99, -99, -99);

    }


/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++ Translate inner Halo-Buffer Index to Lattice coordinates +++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++                for COMPUTATION                           +++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///  It translates an continuous Index < (size of inner Halo Part (I)) into the corresponding coordinate in Bulk Lattice
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///
///  This is duplicate code and I don't like it... In principle one could reuse getOuterHaloCoord(...) for that.
///  However if one does that by templating it, the compiler is not smart enough to optimize it away,
///  so that this indexer become slower...

    __device__ __host__ inline static sitexyzt getInnerCoord(size_t HalIndex) {

        size_t HalNumber = 0, LocHalIndex = 0;
        HalNumber = _getHaloNumberReduced(HalIndex, &LocHalIndex);

        size_t left_right = 0;


        /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
        if (HalNumber < 2) {
            left_right = HalNumber;
            return getInnerCoord_Hyperplane_X(LocHalIndex, left_right);
        } else if (HalNumber >= 2 && HalNumber < 4) {
            left_right = HalNumber - 2;
            return getInnerCoord_Hyperplane_Y(LocHalIndex, left_right);
        } else if (HalNumber >= 4 && HalNumber < 6) {
            left_right = HalNumber - 4;
            return getInnerCoord_Hyperplane_Z(LocHalIndex, left_right);
        } else if (HalNumber >= 6 && HalNumber < 8) {
            left_right = HalNumber - 6;
            return getInnerCoord_Hyperplane_T(LocHalIndex, left_right);
        }


        /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
        if (HalNumber >= 8 && HalNumber < 12) {
            left_right = HalNumber - 8;
            return getInnerCoord_Plane_XY(LocHalIndex, left_right);
        } else if (HalNumber >= 12 && HalNumber < 16) {
            left_right = HalNumber - 12;
            return getInnerCoord_Plane_XZ(LocHalIndex, left_right);
        } else if (HalNumber >= 16 && HalNumber < 20) {
            left_right = HalNumber - 16;
            return getInnerCoord_Plane_XT(LocHalIndex, left_right);
        } else if (HalNumber >= 20 && HalNumber < 24) {
            left_right = HalNumber - 20;
            return getInnerCoord_Plane_YZ(LocHalIndex, left_right);
        } else if (HalNumber >= 24 && HalNumber < 28) {
            left_right = HalNumber - 24;
            return getInnerCoord_Plane_YT(LocHalIndex, left_right);
        } else if (HalNumber >= 28 && HalNumber < 32) {
            left_right = HalNumber - 28;
            return getInnerCoord_Plane_ZT(LocHalIndex, left_right);
        }

            /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 32 && HalNumber < 40) {
            left_right = HalNumber - 32;
            return getInnerCoord_Stripe_XYZ(LocHalIndex, left_right);
        } else if (HalNumber >= 40 && HalNumber < 48) {
            left_right = HalNumber - 40;
            return getInnerCoord_Stripe_XYT(LocHalIndex, left_right);
        } else if (HalNumber >= 48 && HalNumber < 56) {
            left_right = HalNumber - 48;
            return getInnerCoord_Stripe_XZT(LocHalIndex, left_right);
        } else if (HalNumber >= 56 && HalNumber < 64) {
            left_right = HalNumber - 56;
            return getInnerCoord_Stripe_YZT(LocHalIndex, left_right);
        }




            /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
        else if (HalNumber >= 64 && HalNumber < 80) {
            left_right = HalNumber - 64;
            return getInnerCoord_Corner(LocHalIndex, left_right);
        }


        printf("ERROR: getInnerCoord");
        return sitexyzt(-99, -99, -99, -99);


    }

/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++ Translate Halo-Buffer Index to Lattice coordinates ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++                for INNER Halos                     ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///    It translates an continuous Index < (size of inner Halo Part (I)) into the corresponding coordinate in I
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///

    /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
    /// lr = 0,1
    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Hyperplane_X(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(y, z, t, x, LocHalIndex, getHalData().h_LYi, getHalData().h_YZ, getHalData().h_YZT);
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Hyperplane_Y(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, z, t, y, LocHalIndex, getHalData().h_LXi, getHalData().h_XZ, getHalData().h_XZT);
        x = x + getHalData().h_HaloDepth[0];
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Hyperplane_Z(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, t, z, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYT);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Hyperplane_T(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYZ);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];

        if (lr) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
    /// lr = 0-3
    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_XY(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, t, x, y, LocHalIndex, getHalData().h_LZi, getHalData().h_ZT, getHalData().h_ZTH);
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];

        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_XZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, t, x, z, LocHalIndex, getHalData().h_LYi, getHalData().h_YT, getHalData().h_YTH);
        y = y + getHalData().h_HaloDepth[1];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_XT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, z, x, t, LocHalIndex, getHalData().h_LYi, getHalData().h_YZ, getHalData().h_YZH);
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_YZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, t, y, z, LocHalIndex, getHalData().h_LXi, getHalData().h_XT, getHalData().h_XTH);
        x = x + getHalData().h_HaloDepth[0];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 2) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_YT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, z, y, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XZ, getHalData().h_XZH);
        x = x + getHalData().h_HaloDepth[0];
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 2) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Plane_ZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYH);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];

        if (lr & 1) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        if (lr & 2) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
    /// lr = 0-7
    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Stripe_XYZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(t, x, y, z, LocHalIndex, getHalData().h_LTi, getHalData().h_TH, getHalData().h_THH);
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 4) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Stripe_XYT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, x, y, t, LocHalIndex, getHalData().h_LZi, getHalData().h_ZH, getHalData().h_ZHH);
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 4) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Stripe_XZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, x, z, t, LocHalIndex, getHalData().h_LYi, getHalData().h_YH, getHalData().h_YHH);
        y = y + getHalData().h_HaloDepth[1];

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        if (lr & 4) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Stripe_YZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XH, getHalData().h_XHH);
        x = x + getHalData().h_HaloDepth[0];

        if (lr & 1) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 2) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        if (lr & 4) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
    /// lr = 0-15
    __device__ __host__ inline static sitexyzt
    getInnerHaloCoord_Corner(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalData().h_HaloDepth[0], getHalData().h_HH, getHalData().h_HHH);

        if (lr & 1) x = (x + getHalData().h_LXi);
        else x = x + getHalData().h_HaloDepth[0];
        if (lr & 2) y = (y + getHalData().h_LYi);
        else y = y + getHalData().h_HaloDepth[1];
        if (lr & 4) z = (z + getHalData().h_LZi);
        else z = z + getHalData().h_HaloDepth[2];
        if (lr & 8) t = (t + getHalData().h_LTi);
        else t = t + getHalData().h_HaloDepth[3];

        return sitexyzt(x, y, z, t);
    }


/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++ Translate Halo-Buffer Index to Lattice coordinates ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++                for OUTER Halos                     ++++++++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///    It translates an continuous Index < (size of outer Halo Part (O)) into the corresponding coordinate in O
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///

    /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
    /// lr = 0,1
    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Hyperplane_X(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(y, z, t, x, LocHalIndex, getHalData().h_LYi, getHalData().h_YZ, getHalData().h_YZT);
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) x = (x + getHalData().h_LX_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Hyperplane_Y(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, z, t, y, LocHalIndex, getHalData().h_LXi, getHalData().h_XZ, getHalData().h_XZT);
        x = x + getHalData().h_HaloDepth[0];
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) y = (y + getHalData().h_LY_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Hyperplane_Z(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, t, z, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYT);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];
        t = t + getHalData().h_HaloDepth[3];

        if (lr) z = (z + getHalData().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Hyperplane_T(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYZ);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];

        if (lr) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
    /// lr = 0-3
    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_XY(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, t, x, y, LocHalIndex, getHalData().h_LZi, getHalData().h_ZT, getHalData().h_ZTH);
        z = z + getHalData().h_HaloDepth[2];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) y = (y + getHalData().h_LY_mH);

        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_XZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, t, x, z, LocHalIndex, getHalData().h_LYi, getHalData().h_YT, getHalData().h_YTH);
        y = y + getHalData().h_HaloDepth[1];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) z = (z + getHalData().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_XT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, z, x, t, LocHalIndex, getHalData().h_LYi, getHalData().h_YZ, getHalData().h_YZH);
        y = y + getHalData().h_HaloDepth[1];
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_YZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, t, y, z, LocHalIndex, getHalData().h_LXi, getHalData().h_XT, getHalData().h_XTH);
        x = x + getHalData().h_HaloDepth[0];
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) y = (y + getHalData().h_LY_mH);
        if (lr & 2) z = (z + getHalData().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_YT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, z, y, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XZ, getHalData().h_XZH);
        x = x + getHalData().h_HaloDepth[0];
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) y = (y + getHalData().h_LY_mH);
        if (lr & 2) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Plane_ZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XY, getHalData().h_XYH);
        x = x + getHalData().h_HaloDepth[0];
        y = y + getHalData().h_HaloDepth[1];

        if (lr & 1) z = (z + getHalData().h_LZ_mH);
        if (lr & 2) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
    /// lr = 0-7
    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Stripe_XYZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(t, x, y, z, LocHalIndex, getHalData().h_LTi, getHalData().h_TH, getHalData().h_THH);
        t = t + getHalData().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) y = (y + getHalData().h_LY_mH);
        if (lr & 4) z = (z + getHalData().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Stripe_XYT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, x, y, t, LocHalIndex, getHalData().h_LZi, getHalData().h_ZH, getHalData().h_ZHH);
        z = z + getHalData().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) y = (y + getHalData().h_LY_mH);
        if (lr & 4) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Stripe_XZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, x, z, t, LocHalIndex, getHalData().h_LYi, getHalData().h_YH, getHalData().h_YHH);
        y = y + getHalData().h_HaloDepth[1];

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) z = (z + getHalData().h_LZ_mH);
        if (lr & 4) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Stripe_YZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalData().h_LXi, getHalData().h_XH, getHalData().h_XHH);
        x = x + getHalData().h_HaloDepth[0];

        if (lr & 1) y = (y + getHalData().h_LY_mH);
        if (lr & 2) z = (z + getHalData().h_LZ_mH);
        if (lr & 4) t = (t + getHalData().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
    /// lr = 0-15
    __device__ __host__ inline static sitexyzt
    getOuterHaloCoord_Corner(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalData().h_HaloDepth[0], getHalData().h_HH, getHalData().h_HHH);

        if (lr & 1) x = (x + getHalData().h_LX_mH);
        if (lr & 2) y = (y + getHalData().h_LY_mH);
        if (lr & 4) z = (z + getHalData().h_LZ_mH);
        if (lr & 8) t = (t + getHalData().h_LT_mH);

        return sitexyzt(x, y, z, t);
    }

/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++ Translate inner Halo-Buffer Index to Lattice coordinates +++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++                for COMPUTATION                           +++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///  It translates an continuous Index < (size of inner Halo Part (I)) into the corresponding coordinate in Bulk Lattice
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///
///  This is duplicate code and I don't like it... In principle one could reuse getOuterHaloCoord(...) for that.
///  However if one does that by templating it, the compiler is not smart enough to optimise it away,
///  so that this indexer become slower...

    /// +++++++++++++++++++++++++++ 8 HYPERPLANES +++++++++++++++++++++++++++++++
    /// lr = 0,1
    __device__ __host__ inline static sitexyzt
    getInnerCoord_Hyperplane_X(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(y, z, t, x, LocHalIndex, getHalDataReduced().h_LYi, getHalDataReduced().h_YZ,
                 getHalDataReduced().h_YZT);
        y = y + getHalDataReduced().h_HaloDepth[1];
        z = z + getHalDataReduced().h_HaloDepth[2];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr) x = (x + getHalDataReduced().h_LX_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Hyperplane_Y(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, z, t, y, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XZ,
                 getHalDataReduced().h_XZT);
        x = x + getHalDataReduced().h_HaloDepth[0];
        z = z + getHalDataReduced().h_HaloDepth[2];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr) y = (y + getHalDataReduced().h_LY_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Hyperplane_Z(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, t, z, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XY,
                 getHalDataReduced().h_XYT);
        x = x + getHalDataReduced().h_HaloDepth[0];
        y = y + getHalDataReduced().h_HaloDepth[1];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr) z = (z + getHalDataReduced().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Hyperplane_T(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XY,
                 getHalDataReduced().h_XYZ);
        x = x + getHalDataReduced().h_HaloDepth[0];
        y = y + getHalDataReduced().h_HaloDepth[1];
        z = z + getHalDataReduced().h_HaloDepth[2];

        if (lr) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// +++++++++++++++++++++++++++++ 24 PLANES +++++++++++++++++++++++++++++++++
    /// lr = 0-3
    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_XY(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, t, x, y, LocHalIndex, getHalDataReduced().h_LZi, getHalDataReduced().h_ZT,
                 getHalDataReduced().h_ZTH);
        z = z + getHalDataReduced().h_HaloDepth[2];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) y = (y + getHalDataReduced().h_LY_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_XZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, t, x, z, LocHalIndex, getHalDataReduced().h_LYi, getHalDataReduced().h_YT,
                 getHalDataReduced().h_YTH);
        y = y + getHalDataReduced().h_HaloDepth[1];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) z = (z + getHalDataReduced().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_XT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, z, x, t, LocHalIndex, getHalDataReduced().h_LYi, getHalDataReduced().h_YZ,
                 getHalDataReduced().h_YZH);
        y = y + getHalDataReduced().h_HaloDepth[1];
        z = z + getHalDataReduced().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_YZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, t, y, z, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XT,
                 getHalDataReduced().h_XTH);
        x = x + getHalDataReduced().h_HaloDepth[0];
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr & 1) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 2) z = (z + getHalDataReduced().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_YT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, z, y, t, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XZ,
                 getHalDataReduced().h_XZH);
        x = x + getHalDataReduced().h_HaloDepth[0];
        z = z + getHalDataReduced().h_HaloDepth[2];

        if (lr & 1) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 2) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Plane_ZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XY,
                 getHalDataReduced().h_XYH);
        x = x + getHalDataReduced().h_HaloDepth[0];
        y = y + getHalDataReduced().h_HaloDepth[1];

        if (lr & 1) z = (z + getHalDataReduced().h_LZ_mH);
        if (lr & 2) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 32 STRIPES +++++++++++++++++++++++++++++++++
    /// lr = 0-7
    __device__ __host__ inline static sitexyzt
    getInnerCoord_Stripe_XYZ(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(t, x, y, z, LocHalIndex, getHalDataReduced().h_LTi, getHalDataReduced().h_TH,
                 getHalDataReduced().h_THH);
        t = t + getHalDataReduced().h_HaloDepth[3];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 4) z = (z + getHalDataReduced().h_LZ_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Stripe_XYT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(z, x, y, t, LocHalIndex, getHalDataReduced().h_LZi, getHalDataReduced().h_ZH,
                 getHalDataReduced().h_ZHH);
        z = z + getHalDataReduced().h_HaloDepth[2];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 4) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Stripe_XZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(y, x, z, t, LocHalIndex, getHalDataReduced().h_LYi, getHalDataReduced().h_YH,
                 getHalDataReduced().h_YHH);
        y = y + getHalDataReduced().h_HaloDepth[1];

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) z = (z + getHalDataReduced().h_LZ_mH);
        if (lr & 4) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    __device__ __host__ inline static sitexyzt
    getInnerCoord_Stripe_YZT(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;
        getCoord(x, y, z, t, LocHalIndex, getHalDataReduced().h_LXi, getHalDataReduced().h_XH,
                 getHalDataReduced().h_XHH);
        x = x + getHalDataReduced().h_HaloDepth[0];

        if (lr & 1) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 2) z = (z + getHalDataReduced().h_LZ_mH);
        if (lr & 4) t = (t + getHalDataReduced().h_LT_mH);
        return sitexyzt(x, y, z, t);
    }

    /// ++++++++++++++++++++++++++++ 16 CORNERS +++++++++++++++++++++++++++++++++
    /// lr = 0-15
    __device__ __host__ inline static sitexyzt
    getInnerCoord_Corner(size_t LocHalIndex, short lr) {
        size_t x, y, z, t;

        getCoord(x, y, z, t, LocHalIndex, getHalDataReduced().h_HaloDepth[0], getHalDataReduced().h_HH,
                 getHalDataReduced().h_HHH);

        if (lr & 1) x = (x + getHalDataReduced().h_LX_mH);
        if (lr & 2) y = (y + getHalDataReduced().h_LY_mH);
        if (lr & 4) z = (z + getHalDataReduced().h_LZ_mH);
        if (lr & 8) t = (t + getHalDataReduced().h_LT_mH);

        return sitexyzt(x, y, z, t);
    }

/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++ Translate inner Halo-Buffer Index to Lattice coordinates +++++++++++++++++++++++++++++++++
/// +++++++++++++++++++++++++                for COMPUTATION                           +++++++++++++++++++++++++++++++++
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

///  It translates an continuous Index < (size of inner Halo Part (I)) into the corresponding coordinate in Bulk Lattice
///
///  Left          Right
///    ______________
///   |  __________  |
///   | |  ______  | |
///   | | |      | | |
///   |O|I|  B   | | |
///   | | |______| | |
///   | |__________| |
///   |______________|
///


    __device__ __host__ inline static sitexyzt getCenterCoord(size_t CenterIndex) {


        size_t x = 0, y = 0, z = 0, t = 0;

        getCoord(x, y, z, t, CenterIndex, getHalData().h_vol1Center, getHalData().h_vol2Center,
                 getHalData().h_vol3Center);
        x += getHalData().h_HaloDepth[0];
        y += getHalData().h_HaloDepth[1];
        z += getHalData().h_HaloDepth[2];
        t += getHalData().h_HaloDepth[3];

        return {(int) x, (int) y, (int) z, (int) t};
    }

};

#endif
