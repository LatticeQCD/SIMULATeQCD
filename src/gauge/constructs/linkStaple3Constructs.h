//
// Created by Jishnu on 07/08/18.
//

#ifndef LINKSTAPLE3CONSTRUCTS_H
#define LINKSTAPLE3CONSTRUCTS_H



#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"


template<class floatT,size_t HaloDepth, CompressionType comp>
  DEVICE GSU3<floatT> inline linkStaple3Up(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;
    //int nu =  (mu + nu_h)%4 ;
    //gSite origin = site;
    gSite up = GInd::site_up(site, nu);
    gSite right = GInd::site_up(site, mu);
    //Staple up
   /*
    nu
    ^
    |
    |
     --> mu

    ^ --->
    |    |
    *    v



    */

    temp =gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));


    return temp;

}

template<class floatT,size_t HaloDepth,CompressionType comp>
  DEVICE GSU3<floatT> inline linkStaple3Dn(gaugeAccessor<floatT,comp> gAcc, gSite site, int mu, int nu) {
    typedef GIndexer<All,HaloDepth> GInd;

    GSU3<floatT> temp;
    //int nu =  (mu + nu_h)%4 ;
    gSite dn = GInd::site_dn(site, nu);
    gSite right = GInd::site_up(site, mu);
    gSite rightDn = GInd::site_dn(right, nu);
    //Staple down
    /*


    *    ^
    |    |
    v---->

    */

    temp = gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), mu, nu);


    return temp;
}
#endif //3LINKSTAPLECONSTRUCTS_H
