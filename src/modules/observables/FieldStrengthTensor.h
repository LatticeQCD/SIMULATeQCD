//
//created by Marcel Rodekamp 30.05.2018
//


#ifndef FIELDSTRENGTHTENSOR_H
#define FIELDSTRENGTHTENSOR_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../../gauge/constructs/PlaqConstructs.h"


/*Qmunu(gaugeAccessor<floatT>, GSU3<floatT> &,  gSite,  int ,  int )
	Determine:
	Q_(mu,nu)(x) = U_(mu,nu)(x) + U_(-mu,nu)(x) + U_(-mu,-nu)(x) + U_(mu,nu)(x)
	where U_(mu,nu)(x) denotes the plaquette at site x.
*/
template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct plaqClover {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;

    plaqClover(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__  inline GSU3<floatT> operator()(gSite site, int mu, int nu) {

        return Plaq_P<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_Q<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_R<floatT, HaloDepth>(acc, site, mu, nu)
               + Plaq_S<floatT, HaloDepth>(acc, site, mu, nu);
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct rectClover {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    typedef GIndexer<All, HaloDepth> GInd;

    rectClover(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        gSite origin = site;
        gSite up = GInd::site_up(site, nu);
        gSite twoUp = GInd::site_up(up, nu);
        gSite dn = GInd::site_dn(site, nu);
        gSite twoDn = GInd::site_dn(dn, nu);

        gSite left = GInd::site_dn(site, mu);
        gSite twoLeft = GInd::site_dn(left, mu);
        gSite twoLeftUp = GInd::site_up(twoLeft, nu);
        gSite twoLeftDn = GInd::site_dn(twoLeft, nu);
        gSite leftUp = GInd::site_up(left, nu);
        gSite leftDn = GInd::site_dn(left, nu);
        gSite left2Up = GInd::site_up(leftUp, nu);
        gSite left2Dn = GInd::site_dn(leftDn, nu);

        gSite right = GInd::site_up(site, mu);
        gSite twoRight = GInd::site_up(right, mu);
        gSite twoRightDn = GInd::site_dn(twoRight, nu);
        gSite rightUp = GInd::site_up(right, nu);
        gSite rightDn = GInd::site_dn(right, nu);
        gSite right2Dn = GInd::site_dn(rightDn, nu);


        GSU3<floatT> temp;

        // top right
        temp = acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, mu))
               * acc.getLink(GInd::getSiteMu(twoRight, nu))
               * acc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));

        temp += acc.getLink(GInd::getSiteMu(origin, mu))
                * acc.getLink(GInd::getSiteMu(right, nu))
                * acc.getLink(GInd::getSiteMu(rightUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, nu));

        // top left
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, nu))
                * acc.getLink(GInd::getSiteMu(twoLeft, mu))
                * acc.getLink(GInd::getSiteMu(left, mu));

        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLink(GInd::getSiteMu(up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left, nu))
                * acc.getLink(GInd::getSiteMu(left, mu));

        // bottom left
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftDn, nu))
                * acc.getLink(GInd::getSiteMu(twoLeftDn, mu))
                * acc.getLink(GInd::getSiteMu(leftDn, mu))
                * acc.getLink(GInd::getSiteMu(dn, nu));


        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(leftDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Dn, nu))
                * acc.getLink(GInd::getSiteMu(left2Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(dn, nu));

        // bottom right

        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLink(GInd::getSiteMu(dn, mu))
                * acc.getLink(GInd::getSiteMu(rightDn, mu))
                * acc.getLink(GInd::getSiteMu(twoRightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(right, mu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));


        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(twoDn, mu))
                * acc.getLink(GInd::getSiteMu(right2Dn, nu))
                * acc.getLink(GInd::getSiteMu(rightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));

        return temp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct squareClover2x2 { 
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    typedef GIndexer<All, HaloDepth> GInd;

    squareClover2x2(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        gSite origin = site;
        gSite up = GInd::site_up(site, nu); //(0,1)
        gSite twoUp = GInd::site_up(up, nu); //(0,2)
        gSite dn = GInd::site_dn(site, nu); //(0,-1)
        gSite twoDn = GInd::site_dn(dn, nu); //(0,-2)

        gSite left = GInd::site_dn(site, mu); //(-1,0)
        gSite twoLeft = GInd::site_dn(left, mu); //(-2,0)
        gSite twoLeftUp = GInd::site_up(twoLeft, nu); //(-2,1)
        gSite twoLeft2Up = GInd::site_up(twoLeftUp, nu); //(-2,2)
        gSite twoLeftDn = GInd::site_dn(twoLeft, nu); //(-2,-1)
        gSite twoLeft2Dn = GInd::site_dn(twoLeftDn, nu); //(-2,-2)
        gSite leftUp = GInd::site_up(left, nu); //(-1,1)
        gSite leftDn = GInd::site_dn(left, nu); //(-1,-1)
        gSite left2Up = GInd::site_up(leftUp, nu); //(-1,2)
        gSite left2Dn = GInd::site_dn(leftDn, nu); //(-1,-2)

        gSite right = GInd::site_up(site, mu); //(1,0)
        gSite twoRight = GInd::site_up(right, mu); //(2,0)
        gSite twoRightUp = GInd::site_up(twoRight, nu); //(2,1)
        gSite twoRightDn = GInd::site_dn(twoRight, nu); //(2,-1)
        gSite twoRight2Dn = GInd::site_dn(twoRightDn, nu); //(2,-2)
        gSite rightUp = GInd::site_up(right, nu); //(1,1)
        gSite rightDn = GInd::site_dn(right, nu); //(1,-1)
        gSite right2Up = GInd::site_up(rightUp, nu); //(1,2)
        gSite right2Dn = GInd::site_dn(rightDn, nu); //(1,-2)


        GSU3<floatT> temp;

        // top right 
        temp = acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, mu))
               * acc.getLink(GInd::getSiteMu(twoRight, nu))
               * acc.getLink(GInd::getSiteMu(twoRightUp, nu))
               * acc.getLinkDagger(GInd::getSiteMu(right2Up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(twoUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(up, nu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));
        
        // top left
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLink(GInd::getSiteMu(up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft2Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, nu))
                * acc.getLink(GInd::getSiteMu(twoLeft, mu))
                * acc.getLink(GInd::getSiteMu(left, mu));

        // bottom left
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft2Dn, nu))
                * acc.getLink(GInd::getSiteMu(twoLeft2Dn, mu))
                * acc.getLink(GInd::getSiteMu(left2Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(dn, nu));

        // bottom right
        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(twoDn, mu))
                * acc.getLink(GInd::getSiteMu(right2Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoRight2Dn, nu))
                * acc.getLink(GInd::getSiteMu(twoRightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(right, mu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));

        return temp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct rectClover1x3 { 
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    typedef GIndexer<All, HaloDepth> GInd;

    rectClover1x3(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        gSite origin = site;
        gSite up = GInd::site_up(site, nu); //(0,1)
        gSite twoUp = GInd::site_up(up, nu); //(0,2)
        gSite threeUp = GInd::site_up(twoUp, nu); //(0,3)
        gSite dn = GInd::site_dn(site, nu); //(0,-1)
        gSite twoDn = GInd::site_dn(dn, nu); //(0,-2)
        gSite threeDn = GInd::site_dn(twoDn, nu); //(0,-3)

        gSite left = GInd::site_dn(site, mu); //(-1,0)
        gSite twoLeft = GInd::site_dn(left, mu); //(-2,0)
        gSite threeLeft = GInd::site_dn(twoLeft, mu); //(-3,0)
        gSite twoLeftUp = GInd::site_up(twoLeft, nu); //(-2,1)
        gSite twoLeft2Up = GInd::site_up(twoLeftUp, nu); //(-2,2)
        gSite twoLeftDn = GInd::site_dn(twoLeft, nu); //(-2,-1)
        gSite twoLeft2Dn = GInd::site_dn(twoLeftDn, nu); //(-2,-2)
        gSite leftUp = GInd::site_up(left, nu); //(-1,1)
        gSite leftDn = GInd::site_dn(left, nu); //(-1,-1)
        gSite left2Up = GInd::site_up(leftUp, nu); //(-1,2)
        gSite left2Dn = GInd::site_dn(leftDn, nu); //(-1,-2)
        gSite left3Up = GInd::site_up(left2Up, nu); //(-1,3)
        gSite left3Dn = GInd::site_dn(left2Dn, nu); //(-1,-3)
        gSite threeLeftUp = GInd::site_up(threeLeft, nu); //(-3,1)
        gSite threeLeftDn = GInd::site_dn(threeLeft, nu); //(-3,-1)

        gSite right = GInd::site_up(site, mu); //(1,0)
        gSite twoRight = GInd::site_up(right, mu); //(2,0)
        gSite threeRight = GInd::site_up(twoRight, mu); //(3,0)
        gSite twoRightUp = GInd::site_up(twoRight, nu); //(2,1)
        gSite twoRightDn = GInd::site_dn(twoRight, nu); //(2,-1)
        gSite rightUp = GInd::site_up(right, nu); //(1,1)
        gSite rightDn = GInd::site_dn(right, nu); //(1,-1)
        gSite right2Up = GInd::site_up(rightUp, nu); //(1,2)
        gSite right2Dn = GInd::site_dn(rightDn, nu); //(1,-2)
        gSite right3Dn = GInd::site_dn(right2Dn, nu); //(1,-3)
        gSite threeRightDn = GInd::site_dn(threeRight, nu); //(3,-1)

        GSU3<floatT> temp;

        // top right 
        // 1x3
        temp = acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, nu))
               * acc.getLink(GInd::getSiteMu(rightUp, nu))
               * acc.getLink(GInd::getSiteMu(right2Up, nu))
               * acc.getLinkDagger(GInd::getSiteMu(threeUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(twoUp, nu))
               * acc.getLinkDagger(GInd::getSiteMu(up, nu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));
        // 3x1 
        temp += acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, mu))
               * acc.getLink(GInd::getSiteMu(twoRight, mu))
               * acc.getLink(GInd::getSiteMu(threeRight, nu))
               * acc.getLinkDagger(GInd::getSiteMu(twoRightUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(rightUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));
       
        // top left
        // 1x3
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLink(GInd::getSiteMu(up, nu))
                * acc.getLink(GInd::getSiteMu(twoUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left3Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left, nu))
                * acc.getLink(GInd::getSiteMu(left, mu));
        // 3x1
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLinkDagger(GInd::getSiteMu(leftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeftUp, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft, nu))
                * acc.getLink(GInd::getSiteMu(threeLeft, mu))
                * acc.getLink(GInd::getSiteMu(twoLeft, mu))
                * acc.getLink(GInd::getSiteMu(left, mu));
        
        // bottom left
        // 1x3
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(leftDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left2Dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left3Dn, nu))
                * acc.getLink(GInd::getSiteMu(left3Dn, mu))
                * acc.getLink(GInd::getSiteMu(threeDn, nu))
                * acc.getLink(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(dn, nu));
        // 3x1
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeftDn, nu))
                * acc.getLink(GInd::getSiteMu(threeLeftDn, mu))
                * acc.getLink(GInd::getSiteMu(twoLeftDn, mu))
                * acc.getLink(GInd::getSiteMu(leftDn, mu))
                * acc.getLink(GInd::getSiteMu(dn, nu));

        // bottom right
        // 1x3
        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeDn, nu))
                * acc.getLink(GInd::getSiteMu(threeDn, mu))
                * acc.getLink(GInd::getSiteMu(right3Dn, nu))
                * acc.getLink(GInd::getSiteMu(right2Dn, nu))
                * acc.getLink(GInd::getSiteMu(rightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));
        // 3x1
        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLink(GInd::getSiteMu(dn, mu))
                * acc.getLink(GInd::getSiteMu(rightDn, mu))
                * acc.getLink(GInd::getSiteMu(twoRightDn, mu))
                * acc.getLink(GInd::getSiteMu(threeRightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoRight, mu))
                * acc.getLinkDagger(GInd::getSiteMu(right, mu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));
        return temp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct squareClover3x3 { 
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    typedef GIndexer<All, HaloDepth> GInd;

    squareClover3x3(gaugeAccessor<floatT,comp> acc) : acc(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        gSite origin = site;
        gSite up = GInd::site_up(site, nu); //(0,1)
        gSite twoUp = GInd::site_up(up, nu); //(0,2)
        gSite threeUp = GInd::site_up(twoUp, nu); //(0,3)
        gSite dn = GInd::site_dn(site, nu); //(0,-1)
        gSite twoDn = GInd::site_dn(dn, nu); //(0,-2)
        gSite threeDn = GInd::site_dn(twoDn, nu); //(0,-3)

        gSite left = GInd::site_dn(site, mu); //(-1,0)
        gSite twoLeft = GInd::site_dn(left, mu); //(-2,0)
        gSite threeLeft = GInd::site_dn(twoLeft, mu); //(-3,0)
        gSite twoLeftUp = GInd::site_up(twoLeft, nu); //(-2,1)
        gSite twoLeft2Up = GInd::site_up(twoLeftUp, nu); //(-2,2)
        gSite twoLeft3Up = GInd::site_up(twoLeft2Up, nu); //(-2,3)
        gSite twoLeftDn = GInd::site_dn(twoLeft, nu); //(-2,-1)
        gSite twoLeft2Dn = GInd::site_dn(twoLeftDn, nu); //(-2,-2)
        gSite twoLeft3Dn = GInd::site_dn(twoLeft2Dn, nu); //(-2,-3)
        gSite leftUp = GInd::site_up(left, nu); //(-1,1)
        gSite leftDn = GInd::site_dn(left, nu); //(-1,-1)
        gSite left2Up = GInd::site_up(leftUp, nu); //(-1,2)
        gSite left2Dn = GInd::site_dn(leftDn, nu); //(-1,-2)
        gSite left3Up = GInd::site_up(left2Up, nu); //(-1,3)
        gSite left3Dn = GInd::site_dn(left2Dn, nu); //(-1,-3)
        gSite threeLeftUp = GInd::site_up(threeLeft, nu); //(-3,1)
        gSite threeLeft2Up = GInd::site_up(threeLeftUp, nu); //(-3,2)
        gSite threeLeft3Up = GInd::site_up(threeLeft2Up, nu); //(-3,3)
        gSite threeLeftDn = GInd::site_dn(threeLeft, nu); //(-3,-1)
        gSite threeLeft2Dn = GInd::site_dn(threeLeftDn, nu); //(-3,-2)
        gSite threeLeft3Dn = GInd::site_dn(threeLeft2Dn, nu); //(-3,-3)

        gSite right = GInd::site_up(site, mu); //(1,0)
        gSite twoRight = GInd::site_up(right, mu); //(2,0)
        gSite threeRight = GInd::site_up(twoRight, mu); //(3,0)
        gSite twoRightUp = GInd::site_up(twoRight, nu); //(2,1)
        gSite twoRight2Up = GInd::site_up(twoRightUp, nu); //(2,2)
        gSite twoRight3Up = GInd::site_up(twoRight3Up, nu); //(2,3)
        gSite twoRightDn = GInd::site_dn(twoRight, nu); //(2,-1)
        gSite twoRight2Dn = GInd::site_dn(twoRightDn, nu); //(2,-2)
        gSite twoRight3Dn = GInd::site_dn(twoRight2Dn, nu); //(2,-2)
        gSite rightUp = GInd::site_up(right, nu); //(1,1)
        gSite rightDn = GInd::site_dn(right, nu); //(1,-1)
        gSite right2Up = GInd::site_up(rightUp, nu); //(1,2)
        gSite right3Up = GInd::site_up(right2Up, nu); //(1,3)
        gSite right2Dn = GInd::site_dn(rightDn, nu); //(1,-2)
        gSite right3Dn = GInd::site_dn(right2Dn, nu); //(1,-3)
        gSite threeRightUp = GInd::site_up(threeRight, nu); //(3,1)
        gSite threeRight2Up = GInd::site_up(threeRightUp, nu); //(3,2)
        gSite threeRightDn = GInd::site_dn(threeRight, nu); //(3,-1)
        gSite threeRight2Dn = GInd::site_dn(threeRightDn, nu); //(3,-2)
        gSite threeRight3Dn = GInd::site_dn(threeRight2Dn, nu); //(3,-3)

        GSU3<floatT> temp;

        // top right 
        // 3x3
        temp = acc.getLink(GInd::getSiteMu(origin, mu))
               * acc.getLink(GInd::getSiteMu(right, mu))
               * acc.getLink(GInd::getSiteMu(twoRight, mu))
               * acc.getLink(GInd::getSiteMu(threeRight, nu))
               * acc.getLink(GInd::getSiteMu(threeRightUp, nu))
               * acc.getLink(GInd::getSiteMu(threeRight2Up, nu))
               * acc.getLinkDagger(GInd::getSiteMu(twoRight3Up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(right3Up, mu))
               * acc.getLinkDagger(GInd::getSiteMu(threeUp, mu))
               * acc.getLinkDagger(GInd::getSiteMu(twoUp, nu))
               * acc.getLinkDagger(GInd::getSiteMu(up, nu))
               * acc.getLinkDagger(GInd::getSiteMu(origin, nu));
      
//EDIT
        // top left
        // 3x3
        temp += acc.getLink(GInd::getSiteMu(origin, nu))
                * acc.getLink(GInd::getSiteMu(up, nu))
                * acc.getLink(GInd::getSiteMu(twoUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(left3Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft3Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft3Up, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft2Up, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeftUp, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft, nu))
                * acc.getLink(GInd::getSiteMu(threeLeft, mu))
                * acc.getLink(GInd::getSiteMu(twoLeft, mu))
                * acc.getLink(GInd::getSiteMu(left, mu));
        
        // bottom left
        // 3x3
        temp += acc.getLinkDagger(GInd::getSiteMu(left, mu))
                * acc.getLinkDagger(GInd::getSiteMu(twoLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft, mu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeftDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft2Dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeLeft3Dn, nu))
                * acc.getLink(GInd::getSiteMu(threeLeft3Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoLeft3Dn, mu))
                * acc.getLink(GInd::getSiteMu(left3Dn, mu))
                * acc.getLink(GInd::getSiteMu(threeDn, nu))
                * acc.getLink(GInd::getSiteMu(twoDn, nu))
                * acc.getLink(GInd::getSiteMu(dn, nu));

        // bottom right
        // 3x3
        temp += acc.getLinkDagger(GInd::getSiteMu(dn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(threeDn, nu))
                * acc.getLink(GInd::getSiteMu(threeDn, mu))
                * acc.getLink(GInd::getSiteMu(right3Dn, mu))
                * acc.getLink(GInd::getSiteMu(twoRight3Dn, mu))
                * acc.getLink(GInd::getSiteMu(threeRight3Dn, nu))
                * acc.getLink(GInd::getSiteMu(threeRight2Dn, nu))
                * acc.getLink(GInd::getSiteMu(threeRightDn, nu))
                * acc.getLinkDagger(GInd::getSiteMu(twoRight, mu))
                * acc.getLinkDagger(GInd::getSiteMu(right, mu))
                * acc.getLinkDagger(GInd::getSiteMu(origin, mu));
        return temp;
    }
};

/*FmunuKernel(gaugeAccessor<floatT> , GSU3<floatT> & ,  gSite ,  int ,  int )
	computes the tracless clover given by
		a^2 * F_{mu,nu} = -i * 1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
	with
		Q_{mu,nu} = U_(mu,nu)(x) + U_(nu,-mu)(x) + U_(-mu,-nu)(x) + U_(-nu,mu)(x)
	where U denotes the Link variables.

*/
template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct FieldStrengthTensor {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT,comp> acc;
    plaqClover<floatT,HaloDepth,onDevice,comp> plClov;
    typedef GIndexer<All, HaloDepth> GInd;

    FieldStrengthTensor(gaugeAccessor<floatT,comp> acc) : acc(acc),
    plClov(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        //define a unitary matrix for the addition in the end
        GSU3<floatT> unityGSU3 = gsu3_one<floatT>();

        //define a temporary GSU3 for the Fmunu computation
        GSU3<floatT> Fmunu;

        //define a temporary GSU3 for the Qmunu computations
        GSU3<floatT> Qmunu;

        Qmunu = plClov(site, mu, nu);

        // compute F_{mu,nu} = -i*1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
        Fmunu = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu)); // this is faster...

        // return tracless F_{mu,nu}
        return Fmunu - 1. / 3. * tr_c(Fmunu) * unityGSU3;

    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct FieldStrengthTensor_imp {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> acc;
    plaqClover<floatT,HaloDepth,onDevice,comp> plClov;
    rectClover<floatT,HaloDepth,onDevice,comp> rcClov;
    typedef GIndexer<All, HaloDepth> GInd;

    FieldStrengthTensor_imp(gaugeAccessor<floatT,comp> acc) : acc(acc),
    plClov(acc), rcClov(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        //define a unitary matrix for the addition in the end
        GSU3<floatT> unityGSU3 = gsu3_one<floatT>();

        //define a temporary GSU3 for the Fmunu computation
        GSU3<floatT> Fmunu;
        GSU3<floatT> Fmunu_plaq;
        GSU3<floatT> Fmunu_rect;

        //define a temporary GSU3 for the Qmunu computations
        GSU3<floatT> Qmunu;

        Qmunu = plClov(site, mu, nu);
        // compute F_{mu,nu} = -i*1/8 * (Q_{mu,nu} - Q_{nu,mu}) - 1/3 tr(F_mu_nu)*I
        Fmunu_plaq = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu)); // this is faster...

        Qmunu = rcClov(site, mu, nu);
        Fmunu_rect = (GCOMPLEX(floatT)(0, -1)) / ((floatT) 16) * (Qmunu - dagger(Qmunu));


        Fmunu = floatT(5./3.) * Fmunu_plaq - floatT(1./3.) * Fmunu_rect;

        // return tracless F_{mu,nu}
        return Fmunu - floatT(1./3.) * tr_c(Fmunu) * unityGSU3;

    }
};

template<class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct FieldStrengthTensor_imp_imp {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> acc;
    plaqClover<floatT,HaloDepth,onDevice,comp> plClov;
    rectClover<floatT,HaloDepth,onDevice,comp> rcClov;
    squareClover2x2<floatT,HaloDepth,onDevice,comp> sqClov2x2;
    rectClover1x3<floatT,HaloDepth,onDevice,comp> rcClov1x3;
    squareClover3x3<floatT,HaloDepth,onDevice,comp> sqClov3x3;
    typedef GIndexer<All, HaloDepth> GInd;

    FieldStrengthTensor_imp_imp(gaugeAccessor<floatT,comp> acc) : acc(acc),
    plClov(acc), rcClov(acc), sqClov2x2(acc), rcClov1x3(acc), sqClov3x3(acc) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSite site, int mu, int nu) {
        //define a unitary matrix for the addition in the end
        GSU3<floatT> unityGSU3 = gsu3_one<floatT>();

        //define a temporary GSU3 for the Fmunu computation
        GSU3<floatT> Fmunu;

        //set the coefficients from the paper arXiv:hep-lat/0203008, below the Eq.(32)
        floatT k5 = 0.0; //3x3
        floatT k1 = 19.0/9.0 - 55.0 * k5; //1x1
        floatT k2 = 1.0/36.0 - 16.0 * k5; //2x2
        floatT k3 = 64.0 * k5 - 32.0/45.0; //1x2
        floatT k4 = 1.0/15.0 - 6.0 * k5; //1x3
        //in the case of rectangular loops m!=n we average the contribution of the loops in each direction. //between Eq.(12) and Eq.(13) in the paper
        k3*=0.5;
        k4*=0.5;
        
        //q(x) = g^2/(32pi^2) epsilon_{mu nu rho sigma} Tr{F_{mu nu}(x) F_{rho sigma}(x)}
        //define a temporary GSU3 for the Qmunu computations
        GSU3<floatT> Qmunu;
        Fmunu=0.0;

        if (k1 != 0.0){ //1x1
          Qmunu = k1*plClov(site, mu, nu);
          Fmunu += (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu)); 
        }
        if (k2 != 0.0){ //2x2
          Qmunu = k2*sqClov2x2(site, mu, nu);
          Fmunu += (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu));
        }
        if (k3 != 0.0){ //1x2
          Qmunu = k3*rcClov(site, mu, nu);
          Fmunu += (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu));
        }
        if (k4 != 0.0){ //1x3
          Qmunu = k4*rcClov1x3(site, mu, nu);
          Fmunu += (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu));
        }
        if (k5 != 0.0){ //3x3
          Qmunu = k5*sqClov3x3(site, mu, nu);
          Fmunu += (GCOMPLEX(floatT)(0, -1)) / ((floatT) 8) * (Qmunu - dagger(Qmunu));
        }

        return Fmunu - floatT(1./3.) * tr_c(Fmunu) * unityGSU3;
    }
};
#endif //FIELDSTRENGTHTENSOR_H
