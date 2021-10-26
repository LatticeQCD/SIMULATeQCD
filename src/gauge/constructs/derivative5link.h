//created by Dennis on 02.11.18

#ifndef DERIVATIVE_5LINK_H
#define DERIVATIVE_5LINK_H

#include "../../base/indexer/BulkIndexer.h"
#include "../../base/math/gaugeAccessor.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
__device__ GSU3<floatT> linkDerivative5(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {

    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp;

    //Force at position 3
    //term 1
    temp=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches D3 (4716) - nu-rho exchanged - fixed +

    //term 3
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho)); //matches A3 (4536) - nu-rho exchanged - fixed +

    //term 5
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches B3 (4595) - nu-rho exchanged - fixed

    //term 7
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));//matches C3 (4655) - nu-rho exchanged - fixed

    //force at position 2
    //term 9
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //wrong ... corrected and matches D4 (4725) - nu-rho exchanged - fixed

    //term 11
    temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu)) //matches B2 (4581)  - nu-rho exchanged - fixed
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));

    //term 13
    temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
            *finAccessor.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho),rho))
            *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); // wrong .. corrected matches B4  - nu-rho exchanged - fixed

    //term 15
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches D2  - nu-rho exchanged - fixed

    //force at position 1
    //term 17
    temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches C5

    //term 19
    temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho)); //matches D1

    //term 21
    temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho)); //matches D5

    //term 23
    temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches C1

    //force at position 4
    //term 25
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches C4 - nu-rho exchanged - fixed

    //term 27
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), nu)); //matches A2 - nu-rho exchanged - fixed

    //term 29
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), nu)); //matches A4 - nu-rho exchanged - fixed

    //term 31
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,rho), nu))
            *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //wrong matches C2 - nu-rho exchanged - fixed

    //force at position 5
    //term 33
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), nu)); //matches A5

    //term 35
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
            *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches B1

    //term 37
    temp+=gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), nu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
            *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), nu)); //matches B5

    //term 39
    temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
            *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches A1

    return temp;
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_1(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_3(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));
};
    
template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_5(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_7(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_9(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_11(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu)) //matches B2 (4581)  - nu-rho exchanged - fixed
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_13(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *finAccessor.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_15(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_17(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_19(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));
};


template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_21(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_23(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_25(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_27(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_29(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_31(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,rho), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(site,nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_33(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_35(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                *finAccessor.getLink(GInd::getSiteMu(site,nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_37(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), nu));
};

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
    __device__ GSU3<floatT> linkDerivative5_39(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, int nu, int rho) {
    typedef GIndexer<All,HaloDepth> GInd;
    return gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                *finAccessor.getLink(GInd::getSiteMu(site,nu));
};

#endif // DERIVATIVE_5LINK_H