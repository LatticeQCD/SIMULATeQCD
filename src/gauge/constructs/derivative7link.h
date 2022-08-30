#ifndef DERIVATIVE_7LINK_H
#define DERIVATIVE_7LINK_H

#include "../../base/indexer/BulkIndexer.h"
#include "../../base/math/gaugeAccessor.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R18, CompressionType compForce=R18>
__host__ __device__ GSU3<floatT> linkDerivative7(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site,
                                        int mu, int nu, int rho, int sigma, int TermCheck = -1, int SubTermCheck = -1) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp = gsu3_zero<floatT>();

    //terms with force @ 1
    if ( TermCheck==1 || TermCheck < 0 ) {

        //1.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,rho,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //was wrong - matches I1
        }

        //1.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches I7
        }

        //1.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,sigma,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_dn(site,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu( GInd::site_dn(site,rho),rho)); //matches K1
        }

        //1.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches J1
        }

        //1.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_dn(site,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu( GInd::site_dn(site,rho),rho)); //matches K7
        }

        //1.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches J7
        }

        //1.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches L1
        }

        //1.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); // was wrong - matches L7
        }
    } // if (TermCheck==1 || TermCheck<0)

    //terms with force @ 2
    if (TermCheck==2 || TermCheck<0) {

        //2.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches K2 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches G2 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches K6 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches L2 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches G6 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches H2 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches L6 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }

        //2.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches H6 sigma=nu(BI), nu=rho(BI), rho=sigma(BI)
        }
    } // if (TermCheck==2 || TermCheck<0)

    //terms with force @ 3
    if(TermCheck==3 || TermCheck<0) {

        //3.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,rho,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches L3 sigma=nu(BI)
        }

        //3.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches J3 sigma=nu(BI)
        }

        //3.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches H3 sigma=nu(BI)
        }

        //3.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches L5 sigma=nu(BI)
        }

        //3.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches H5 sigma=nu(BI)
        }

        //3.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches F3 sigma=nu(BI)
        }

        //3.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,sigma,nu,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches J5 sigma=nu(BI)
        }

        //3.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches F5 sigma=nu(BI)
        }
    } // if (TermCheck==3 || TermCheck<0)

    //terms with force @ 4
    if(TermCheck == 4 || TermCheck < 0  ) {

        //4.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,nu,rho,sigma),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches L4 sigma=nu(BI)
        }

        //4.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //was wrong - matches I4 sigma=nu(BI)
        }

        //4.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,sigma,rho),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches J4 sigma=nu(BI)
        }

        //4.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //was wrong - matches K4 sigma=nu(BI)
        }

        //4.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches H4 sigma=nu(BI)
        }

        //4.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches E4 sigma=nu(BI)
        }

        //4.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,nu)); //matches F4 sigma=nu(BI)
        }

        //4.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches G4 sigma=nu(BI)
        }
    } // if (TermCheck==4 || TermCheck<0)

    //terms with force @ 5
    if (TermCheck == 5 || TermCheck < 0 ) {

        //5.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up(site,rho,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches K3 sigma=nu(BI)
        }

        //5.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches K5 sigma=nu(BI)
        }

        //5.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,sigma,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches I3 sigma=nu(BI)
        }

        //5.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches G3 sigma=nu(BI)
        }

        //5.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches I5 sigma=nu(BI)
        }

        //5.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches G5 sigma=nu(BI)
        }

        //5.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches E3 sigma=nu(BI)
        }

        //5.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches E5 sigma=nu(BI)
        }
    } // if (TermCheck==5 || TermCheck<0)

    //terms with force @ 6
    if (TermCheck==6 || TermCheck < 0 ) {
        //6.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches I2 nu=rho(BI)
        }

        //6.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches I6 nu=rho(BI)
        }

        //6.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches E2 nu=rho(BI)
        }

        //6.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches J2 nu=rho(BI)
        }

        //6.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches E6 nu=rho(BI)
        }

        //6.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(site,rho)); //matches J6 nu=rho(BI)
        }

        //6.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches F2 nu=rho(BI)
        }

        //6.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); //matches F6 nu=rho(BI)
        }
    } // if (TermCheck==6 || TermCheck<0)

    //terms with force @ 7
    if (TermCheck == 7 || TermCheck < 0 ) {

        //7.1
        if ( SubTermCheck == 1 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,mu,rho,sigma),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches E1
        }

        //7.2
        if ( SubTermCheck == 2 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches E7
        }

        //7.3
        if ( SubTermCheck == 3 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches G1
        }

        //7.4
        if ( SubTermCheck == 4 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches F1
        }

        //7.5
        if ( SubTermCheck == 5 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,sigma,nu,rho),nu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //was wrong - matches G7
        }

        //7.6
        if ( SubTermCheck == 6 || SubTermCheck < 0 ) {
            temp+=gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches F7
        }

        //7.7
        if ( SubTermCheck == 7 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                    *finAccessor.getLink(GInd::getSiteMu(site,nu)); //matches H1
        }

        //7.8
        if ( SubTermCheck == 8 || SubTermCheck < 0 ) {
            temp+=gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),nu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                    *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),nu)); //matches H7
        }
    } // if (TermCheck==7 || TermCheck<0)

return temp;
}

#endif // DERIVATIVE_7LINK_H
