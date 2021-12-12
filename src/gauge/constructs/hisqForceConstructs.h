#ifndef HISQFORCE_H
#define HISQFORCE_H

#include "../../modules/HISQ/smearParameters.h"
#include "derivative3link.h"
#include "derivative5link.h"
#include "derivative7link.h"
#include "derivativeLepagelink.h"

template<class floatT,size_t HaloDepth,CompressionType compIn=R14, CompressionType compForce=R18>
    __host__ __device__ GSU3<floatT> smearingForce(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor,
						                           gSite site, int mu, SmearingParameters<floatT> _smearparam, int TermCheck = -1,
						                           int SubTermCheck = -1, bool doL1 = true, bool doL3 = true, bool doL5 = true,
						                           bool doL7 = true, bool doLLp = true) {
    typedef GIndexer<All,HaloDepth> GInd;

    floatT c1 =_smearparam._c_1;
    floatT c3 =_smearparam._c_3;
    floatT c5 =_smearparam._c_5;
    floatT c7 =_smearparam._c_7;
    floatT c_lp =_smearparam._c_lp;
    bool isLvl1 = (c_lp == 0.0 ? true : false);

    GSU3<floatT> temp= gsu3_zero<floatT>();
    GSU3<floatT> derivative_single_link = gsu3_zero<floatT>();
    GSU3<floatT> derivative_staple3 = gsu3_zero<floatT>();
    GSU3<floatT> derivative_staple5 = gsu3_zero<floatT>();
    GSU3<floatT> derivative_staple7 = gsu3_zero<floatT>();
    GSU3<floatT> derivative_staple_lp = gsu3_zero<floatT>();

    derivative_single_link= finAccessor.getLink(GInd::getSiteMu(site, mu));

    for (int nu_h = 1; nu_h < 4; nu_h++) {
      int nu = (mu + nu_h)%4;
      derivative_staple3 += linkDerivative3<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu, nu);
      if (!isLvl1) {
	derivative_staple_lp+=linkDerivativeLepage<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu, nu);
      }

      for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
      int sigma = 6 - mu - nu - rho;
	derivative_staple5 += linkDerivative5<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu, nu, rho);
	derivative_staple7 += linkDerivative7<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu, nu, rho, sigma, TermCheck, SubTermCheck);
      }
    }

    if (!doL1) {
      c1 = 0;
    }
    if (!doL3) {
      c3 = 0;
    }
    if (!doL5) {
      c5 = 0;
    }
    if (!doL7) {
      c7 = 0;
    }
    if (!doLLp) {
      c_lp = 0;
    }

    temp=c1*derivative_single_link- c3* derivative_staple3+ c5*derivative_staple5 - c7*derivative_staple7 + c_lp*derivative_staple_lp;
    return temp;
};
     
template<class floatT,size_t HaloDepth,CompressionType compIn=R14, CompressionType compForce=R18>
  __host__ __device__ GSU3<floatT> threeLinkContribution(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor,
						 gSite site, int mu, SmearingParameters<floatT> _smearparam) {
    typedef GIndexer<All,HaloDepth> GInd;
    floatT c1 =_smearparam._c_1;
    floatT c3 =_smearparam._c_3;
    GSU3<floatT> derivative_staple3 = gsu3_zero<floatT>();

   
    for (int nu_h = 1; nu_h < 4; nu_h++) {
      int nu = (mu + nu_h)%4;
      derivative_staple3 += linkDerivative3<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu, nu);
    }
    return c1*finAccessor.getLink(GInd::getSiteMu(site,mu))-c3*derivative_staple3;
};

template<class floatT, size_t HaloDepth, CompressionType compIn=R14, CompressionType compForce=R18> __host__ __device__ GSU3<floatT> lepagelinkContribution(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> finAccessor, gSite site, int mu, floatT c_lp) {
    GSU3<floatT> derivative_lp = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu+nu_h)%4;
        derivative_lp += linkDerivativeLepage<floatT,HaloDepth,compIn,compForce>(gAcc, finAccessor, site, mu ,nu);
    }
    return c_lp*derivative_lp;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7, int Term, int SubTerm) {
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;
          sevenlinkCont += linkDerivative7<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho, sigma,Term, SubTerm);
        }
    }
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5, int part) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
         
          if (part == 20) {
              
              fivelinkCont += linkDerivative5_17<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_5<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_15<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_1<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_23<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho);
          }

          if (part == 30) {
              fivelinkCont += linkDerivative5_19<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_7<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_9<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_13<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho)+linkDerivative5_21<floatT,HaloDepth,comp,R18>(gAcc, finAccessor, site, mu, nu, rho);
          }
        }
    }
    return finAccessor.getLink(GInd::getSiteMu(site,mu))+c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_11(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
         
              
          fivelinkCont += (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
              *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
              +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
              *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
              *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
                           )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
              *finAccessor.getLink(GInd::getSiteMu(site,nu));
        }
    }
    return c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_12(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;

          fivelinkCont += (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
              *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
              +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
              *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
                           )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), nu));
              }
    }
    return c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_13(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;

          fivelinkCont += (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
              +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), nu))
              *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,nu), rho))
                           )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
              *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), nu));

              }
    }
    return c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_14(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;

          fivelinkCont +=(gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                          *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
                          *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,rho), nu))
                          *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
                          +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                          *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
                          *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
                          *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
                          )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,nu));
        }
    }
    return c5*fivelinkCont;
};


template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_20(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
         
          fivelinkCont += 
  
              ((finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu), rho))
                        
                +
                gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), mu))
                )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu), nu))
               
               
               
               
               +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu), rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
                
                
                
                
                +gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu), nu))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), mu))
               
               
               
  
               +(gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), rho))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho), nu))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho), mu))
                 
                 +finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu), rho))
                 )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho), nu))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,rho));
          
        }
    }
    return c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> fiveLinkContribution_30(gaugeAccessor<floatT, comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c5) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> fivelinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          
          fivelinkCont += 
              ((finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu), mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), rho))
              

                +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho), mu))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), nu))
              


               +(gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu), nu))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho), rho))
                 *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), nu))
              
              

                 +gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                 *finAccessor.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                 *gAcc.getLink(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
                 )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho), mu))
              

               +(finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu), nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu), mu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), rho))
              
                 +gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn(site,mu,rho), rho))
                 *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_up_dn_dn(site,mu,nu,rho), nu))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), mu))
                 )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho), nu))
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho), rho));
        }
    }
    return c5*fivelinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_1(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=((finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,rho,sigma),nu))
            +
            finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),nu))
                            )
           *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
           +
           (finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),nu))
            +
            finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
            *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
            *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),nu))
            )
           *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                           )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,rho))
              
              +
              ((finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,sigma,rho),nu))
                +
                finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),nu))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu( GInd::site_dn(site,rho),sigma))
               +
               (finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,mu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),nu))
                +
                finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),nu))
                )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho)); 
              
              
        }
    }
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_2(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),nu))
               
              
              +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,rho))
              
              +
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
               
              
              +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho));
              
        }
    }
    
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_3(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                  *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,rho,sigma),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                  +
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                  *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                  )
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                           
               +
               gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,rho),nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),nu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                 )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
              *gAcc.getLinkDagger(GInd::getSiteMu(site,nu))
              
              +
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                  *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                  
                  +
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                  *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                  )
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
               +
               (gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                  *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,sigma,nu,rho),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                  +
                  
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                  *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                  )
                )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu));
          
        }
    }
    
    return -c7*sevenlinkCont;
};



template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_4(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=

              (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                 *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,nu,rho,sigma),mu))
                   *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
                   +
                   gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),mu))
                   *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
                   )
                 
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),rho))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
                 *((gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),mu))
                    *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                    +
                    gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                    *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,sigma,rho),mu))
                    *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                    )
                   *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
                   )
                 
                 )
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,nu))
              
              +
              
              (gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
                 *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),mu))
                   *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
                   +
                   gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),mu))
                   *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
                   )
                
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))
                 *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),mu))
                   *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                   +
                   gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                   *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),mu))
                   *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                   )
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
                 )
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),nu));
              }
    }
    
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_5(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=
              (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,nu,rho),sigma))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_up(site,rho,sigma),nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,nu,rho,sigma),sigma))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),nu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                 )
               +
               gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,rho,sigma,nu),nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,rho),sigma))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,rho,nu,sigma),nu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,rho,sigma),sigma))
                 )
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,rho))
              
              +
              (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),nu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),sigma))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_up_dn(site,sigma,rho),nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),sigma))
               +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,nu,rho,sigma),sigma))
                 *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),nu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                 )
               +
               gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),nu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
               *(gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,sigma,nu,rho),nu))
                 *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,rho),sigma))
                 +
                 gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),sigma))
                 *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn_dn(site,nu,rho,sigma),nu))
                 *gAcc.getLink(GInd::getSiteMu(GInd::site_dn_dn(site,rho,sigma),sigma))
                 )
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho));
              
              
              }
    }
    
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_6(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,nu),rho))
               *finAccessor.getLink(GInd::getSiteMu(GInd::site_up(site,rho),nu))
               +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                +
               gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,nu),rho))
               *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,rho,nu),nu))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(site,rho))

              +

              ((gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,sigma),sigma))
                +
                gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,sigma),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,nu,rho),rho))
               *finAccessor.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),nu))
               +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),sigma))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,nu),nu))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),sigma))
                +
                gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,sigma),sigma))
                *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),nu))
                *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,sigma),sigma))
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),rho))
               *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site,nu,rho),nu))
               )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site,rho),rho));
              }
    }
    
    return -c7*sevenlinkCont;
};

template<class floatT, size_t HaloDepth, CompressionType comp> __host__ __device__ GSU3<floatT> sevenLinkContribution_7(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT> finAccessor, gSite site, int mu, floatT c7) {
    typedef GIndexer<All, HaloDepth> GInd;
    GSU3<floatT> sevenlinkCont = gsu3_zero<floatT>();
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu + nu_h)%4;
        for (int rho_h = 0; rho_h < 2; rho_h++) {
          int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
          int sigma = 6 - mu - nu - rho;

          sevenlinkCont +=
    
              ((gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up(site,mu,rho,sigma),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site,mu,nu,rho),sigma))
                  +
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,nu,rho,sigma),sigma))
                  )
                )
               *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site,mu,nu),rho))
                            
               +
               (gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,sigma,rho),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),sigma))
                 
                  
                  +
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,nu,rho,sigma),sigma))
                  )
                )
               *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,nu,rho),rho))
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site,nu),mu))
              *finAccessor.getLink(GInd::getSiteMu(site,nu))
              
              +
              ((gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),rho))
                *(gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn(site,mu,rho),sigma))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,sigma,nu,rho),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),sigma))
                  +
                  gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,rho,sigma),sigma))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn_dn(site,mu,rho,sigma,nu),sigma))
                  )
                )
              *gAcc.getLink(GInd::getSiteMu(GInd::site_up_dn_dn(site,mu,nu,rho),rho))

               +
               (gAcc.getLink(GInd::getSiteMu(GInd::site_up(site,mu),rho))
                *(gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,sigma),sigma))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLink(GInd::getSiteMu(GInd::site_up_up_dn_dn(site,mu,rho,sigma,nu),sigma))
                  +
                  gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site,mu,rho),sigma))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up_dn(site,mu,rho,sigma,nu),nu))
                  *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_dn(site,mu,rho,nu),sigma))
                  )
                )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(site,mu,nu),rho))
               
               )
              *gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),mu))
              *finAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site,nu),nu));
    
              }
    }
    
    return -c7*sevenlinkCont;
};


#endif //HISQFORCE_H
