/* 
 * calcGSiteHalo.h                                                               
 * 
 * L. Mazur 
 * 
 */

#ifndef CALCGSITEHALO_H
#define CALCGSITEHALO_H

#include "../../define.h"
#include "../../base/gutils.h"
#include "../math/operators.h"
#include "../indexer/HaloIndexer.h"


struct HaloSite {
    size_t HaloIndex;
    size_t LatticeIndex;
    size_t LocHalIndex;
    size_t HalNumber;
};


template<class floatT, Layout LatLayout, size_t HaloDepth>
struct CalcOuterHaloIndexComm {
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    inline __host__ __device__ HaloSite
    operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {

        HaloSite site;
        site.HaloIndex = blockDim.x * blockIdx.x + threadIdx.x;

        sitexyzt coord = HInd::getOuterHaloCoord(site.HaloIndex, site.HalNumber, site.LocHalIndex);

        site.LatticeIndex = GInd::getSiteFull(coord.x, coord.y, coord.z, coord.t).isiteFull;
        return site;
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth>
struct CalcInnerHaloIndexComm {
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    inline __host__ __device__ HaloSite
    operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {

        HaloSite site;
        site.HaloIndex = blockDim.x * blockIdx.x + threadIdx.x;

        sitexyzt coord = HInd::getInnerHaloCoord(site.HaloIndex, site.HalNumber, site.LocHalIndex);

        site.LatticeIndex = GInd::getSiteFull(coord.x, coord.y, coord.z, coord.t).isiteFull;
        return site;
    }
};


template<Layout LatLayout, size_t HaloDepth, HaloSegment hseg, short leftRight>
struct CalcOuterHaloSegCoord{
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    inline __host__ __device__ sitexyzt
    operator()(size_t LocHalIndex) {

        sitexyzt coord(0, 0, 0, 0);
        if (hseg == X) coord = HInd::getOuterHaloCoord_Hyperplane_X(LocHalIndex, leftRight);
        else if (hseg == Y) coord = HInd::getOuterHaloCoord_Hyperplane_Y(LocHalIndex, leftRight);
        else if (hseg == Z) coord = HInd::getOuterHaloCoord_Hyperplane_Z(LocHalIndex, leftRight);
        else if (hseg == T) coord = HInd::getOuterHaloCoord_Hyperplane_T(LocHalIndex, leftRight);
        else if (hseg == XY) coord = HInd::getOuterHaloCoord_Plane_XY(LocHalIndex, leftRight);
        else if (hseg == XZ) coord = HInd::getOuterHaloCoord_Plane_XZ(LocHalIndex, leftRight);
        else if (hseg == XT) coord = HInd::getOuterHaloCoord_Plane_XT(LocHalIndex, leftRight);
        else if (hseg == YT) coord = HInd::getOuterHaloCoord_Plane_YT(LocHalIndex, leftRight);
        else if (hseg == YZ) coord = HInd::getOuterHaloCoord_Plane_YZ(LocHalIndex, leftRight);
        else if (hseg == ZT) coord = HInd::getOuterHaloCoord_Plane_ZT(LocHalIndex, leftRight);
        else if (hseg == XYZ) coord = HInd::getOuterHaloCoord_Stripe_XYZ(LocHalIndex, leftRight);
        else if (hseg == XYT) coord = HInd::getOuterHaloCoord_Stripe_XYT(LocHalIndex, leftRight);
        else if (hseg == XZT) coord = HInd::getOuterHaloCoord_Stripe_XZT(LocHalIndex, leftRight);
        else if (hseg == YZT) coord = HInd::getOuterHaloCoord_Stripe_YZT(LocHalIndex, leftRight);
        else if (hseg == XYZT)coord = HInd::getOuterHaloCoord_Corner(LocHalIndex, leftRight);

        return coord;
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth, HaloSegment hseg, short leftRight>
struct CalcOuterHaloSegIndexComm{
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CalcOuterHaloSegCoord<LatLayout,HaloDepth, hseg, leftRight> calcSegCoord;

    inline __host__ __device__ HaloSite
    operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {

        HaloSite site;
        site.LocHalIndex = blockDim.x * blockIdx.x + threadIdx.x;

        sitexyzt coord = calcSegCoord(site.LocHalIndex);
        site.LatticeIndex = GInd::getSiteFull(coord.x, coord.y, coord.z, coord.t).isiteFull;
        return site;
    }
};


template<Layout LatLayout, size_t HaloDepth, HaloSegment hseg, short leftRight>
struct CalcInnerSegCoord{
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    inline __host__ __device__ sitexyzt
    operator()(size_t LocHalIndex) {

        sitexyzt coord(0, 0, 0, 0);
        if (hseg == X) coord = HInd::getInnerCoord_Hyperplane_X(LocHalIndex, leftRight);
        else if (hseg == Y) coord = HInd::getInnerCoord_Hyperplane_Y(LocHalIndex, leftRight);
        else if (hseg == Z) coord = HInd::getInnerCoord_Hyperplane_Z(LocHalIndex, leftRight);
        else if (hseg == T) coord = HInd::getInnerCoord_Hyperplane_T(LocHalIndex, leftRight);
        else if (hseg == XY) coord = HInd::getInnerCoord_Plane_XY(LocHalIndex, leftRight);
        else if (hseg == XZ) coord = HInd::getInnerCoord_Plane_XZ(LocHalIndex, leftRight);
        else if (hseg == XT) coord = HInd::getInnerCoord_Plane_XT(LocHalIndex, leftRight);
        else if (hseg == YT) coord = HInd::getInnerCoord_Plane_YT(LocHalIndex, leftRight);
        else if (hseg == YZ) coord = HInd::getInnerCoord_Plane_YZ(LocHalIndex, leftRight);
        else if (hseg == ZT) coord = HInd::getInnerCoord_Plane_ZT(LocHalIndex, leftRight);
        else if (hseg == XYZ) coord = HInd::getInnerCoord_Stripe_XYZ(LocHalIndex, leftRight);
        else if (hseg == XYT) coord = HInd::getInnerCoord_Stripe_XYT(LocHalIndex, leftRight);
        else if (hseg == XZT) coord = HInd::getInnerCoord_Stripe_XZT(LocHalIndex, leftRight);
        else if (hseg == YZT) coord = HInd::getInnerCoord_Stripe_YZT(LocHalIndex, leftRight);
        else if (hseg == XYZT)coord = HInd::getInnerCoord_Corner(LocHalIndex, leftRight);

        return coord;
    }
};


template<Layout LatLayout, size_t HaloDepth, HaloSegment hseg, short leftRight>
struct CalcInnerHaloSegCoord{
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    inline __host__ __device__ sitexyzt
    operator()(size_t LocHalIndex) {

        sitexyzt coord(0, 0, 0, 0);
        if (hseg == X) coord = HInd::getInnerHaloCoord_Hyperplane_X(LocHalIndex, leftRight);
        else if (hseg == Y) coord = HInd::getInnerHaloCoord_Hyperplane_Y(LocHalIndex, leftRight);
        else if (hseg == Z) coord = HInd::getInnerHaloCoord_Hyperplane_Z(LocHalIndex, leftRight);
        else if (hseg == T) coord = HInd::getInnerHaloCoord_Hyperplane_T(LocHalIndex, leftRight);
        else if (hseg == XY) coord = HInd::getInnerHaloCoord_Plane_XY(LocHalIndex, leftRight);
        else if (hseg == XZ) coord = HInd::getInnerHaloCoord_Plane_XZ(LocHalIndex, leftRight);
        else if (hseg == XT) coord = HInd::getInnerHaloCoord_Plane_XT(LocHalIndex, leftRight);
        else if (hseg == YT) coord = HInd::getInnerHaloCoord_Plane_YT(LocHalIndex, leftRight);
        else if (hseg == YZ) coord = HInd::getInnerHaloCoord_Plane_YZ(LocHalIndex, leftRight);
        else if (hseg == ZT) coord = HInd::getInnerHaloCoord_Plane_ZT(LocHalIndex, leftRight);
        else if (hseg == XYZ) coord = HInd::getInnerHaloCoord_Stripe_XYZ(LocHalIndex, leftRight);
        else if (hseg == XYT) coord = HInd::getInnerHaloCoord_Stripe_XYT(LocHalIndex, leftRight);
        else if (hseg == XZT) coord = HInd::getInnerHaloCoord_Stripe_XZT(LocHalIndex, leftRight);
        else if (hseg == YZT) coord = HInd::getInnerHaloCoord_Stripe_YZT(LocHalIndex, leftRight);
        else if (hseg == XYZT)coord = HInd::getInnerHaloCoord_Corner(LocHalIndex, leftRight);

        return coord;
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth, HaloSegment hseg, short leftRight>
struct CalcInnerHaloSegIndexComm{
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CalcInnerHaloSegCoord<LatLayout,HaloDepth, hseg, leftRight> calcSegCoord;

    CalcInnerHaloSegIndexComm(){}

    inline __host__ __device__ HaloSite
    operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {

        HaloSite site;
        site.LocHalIndex = blockDim.x * blockIdx.x + threadIdx.x;

        sitexyzt coord = calcSegCoord(site.LocHalIndex);
        site.LatticeIndex = GInd::getSiteFull(coord.x, coord.y, coord.z, coord.t).isiteFull;
        return site;
    }
};


template<Layout LatLayout, size_t HaloDepth, typename CalcIndexOp, HaloSegment hseg, int lr>
struct CalcGSiteHaloSeg {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CalcInnerSegCoord<LatLayout, HaloDepth, hseg, lr> calcSegCoord;
    CalcIndexOp calcIndexOp;

    CalcGSiteHaloSeg(CalcIndexOp calcIndexOp) : calcIndexOp(calcIndexOp) {
    }

    inline __host__ __device__ auto
    operator()(size_t HaloIndex, size_t mu) {

        sitexyzt coord = calcSegCoord(HaloIndex);

        auto site = calcIndexOp(GInd::getSite(coord.x, coord.y, coord.z, coord.t), mu);
        return site;
    }
};


template<Layout LatLayout, size_t HaloDepth, typename CalcIndexOp>
struct CalcGSiteInnerHalo {
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CalcIndexOp calcIndexOp;

    CalcGSiteInnerHalo(CalcIndexOp calcIndexOp) : calcIndexOp(calcIndexOp) {
    }

    inline __host__ __device__ auto
    operator()(size_t HaloIndex, size_t mu) {
        sitexyzt coord = HInd::getInnerCoord(HaloIndex);
        auto site = calcIndexOp(GInd::getSite(coord.x, coord.y, coord.z, coord.t), mu);
        return site;
    }
};


template<Layout LatLayout, size_t HaloDepth, typename CalcIndexOp>
struct CalcGSiteCenter {
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CalcIndexOp calcIndexOp;

    CalcGSiteCenter(CalcIndexOp calcIndexOp) : calcIndexOp(calcIndexOp) {
    }

    inline __host__ __device__ auto
    operator()(size_t HaloIndex, size_t mu) {
        sitexyzt coord = HInd::getCenterCoord(HaloIndex);
        auto site = calcIndexOp(GInd::getSite(coord.x, coord.y, coord.z, coord.t), mu);
        return site;
    }
};


#endif //CALCGSITEHALO_H
