#ifndef FAT7LINKCONSTRUCTS_H
#define FAT7LINKCONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"
#include "../../modules/HISQ/smearParameters.h"


template<class floatT, size_t HaloDepth, CompressionType comp>
    HOST_DEVICE GSU3<floatT> inline naikLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {

    typedef GIndexer<All,HaloDepth> GInd;

    gSite origin = GInd::getSite(siteMu.isite);
    int mu = siteMu.mu;
    gSite down = GInd::site_dn(origin,mu);
    gSite up = GInd::site_up(origin,mu);
    
    return gAcc.getLink(GInd::getSiteMu(down,mu))*gAcc.getLink(GInd::getSiteMu(origin,mu))*gAcc.getLink(GInd::getSiteMu(up,mu));
}

template<class floatT,size_t HaloDepth,CompressionType comp>
  HOST_DEVICE GSU3<floatT> inline threeLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp = gsu3_zero<floatT>();
    gSite site = GInd::getSite(siteMu.isite);
    gSite site_save = site;
    int mu = siteMu.mu;
    
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu+nu_h)%4;
        temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
        site = site_save;
        temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), mu, nu);
        site = site_save;
    }
    return temp;
}

template<class floatT, size_t HaloDepth, CompressionType comp>
    HOST_DEVICE GSU3<floatT> inline lepageLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp = gsu3_zero<floatT>();
    gSite site = GInd::getSite(siteMu.isite);
    gSite site_save = site;
    int mu = siteMu.mu;
    
    for (int nu_h = 1; nu_h < 4; nu_h++) {
        int nu = (mu+nu_h)%4;
        temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, nu, mu, Back(nu), Back(nu));
        site = site_save;
        temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), Back(nu), mu, nu, nu);
        site = site_save;
    }
    return temp;
}


template<class floatT, size_t HaloDepth, CompressionType comp, int partNumber>
    HOST_DEVICE GSU3<floatT> inline fiveLinkStaple(gaugeAccessor<floatT, comp> gAcc, gSiteMu siteMu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp = gsu3_zero<floatT>();
    gSite site = GInd::getSite(siteMu.isite);
    gSite site_save = site;
    int mu = siteMu.mu;
    switch (partNumber) {
    case 1:
        
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                
                temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, rho, mu, Back(rho), Back(nu));
                site = site_save;
             
            }
        }
        return temp;
    case 2:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                               
                temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), Back(rho), mu, rho, nu);
                site = site_save;
               
            }
        }
        return temp;
    case 3:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                
                temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, Back(rho), mu, rho, Back(nu));
                site = site_save;
            }
        }
        return temp;
    case 4:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;

                temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), rho, mu, Back(rho), nu);
                site = site_save;
            }
        }
        return temp;
    default:
        return gsu3_zero<floatT>();
        
    }
}

template<class floatT, size_t HaloDepth, CompressionType comp, int partNumber>
    HOST_DEVICE GSU3<floatT> inline sevenLinkStaple(gaugeAccessor<floatT, comp> gAcc, gSiteMu siteMu) {
    typedef GIndexer<All,HaloDepth> GInd;
    GSU3<floatT> temp = gsu3_zero<floatT>();
    gSite site = GInd::getSite(siteMu.isite);
    gSite site_save = site;
    int mu = siteMu.mu;

    switch(partNumber) {
    case 1:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, rho, sigma, mu, Back(sigma), Back(rho), Back(nu));
                site = site_save;
            }
        }
        return temp;
    case 2:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), Back(rho), Back(sigma), mu, sigma, rho, nu);
                site = site_save;
            }
        }
        return temp;
    case 3:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,Back(nu),rho,sigma,mu,Back(sigma),Back(rho),nu);
                site = site_save;
            }
        }
        return temp;
    case 4:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,nu,Back(rho),sigma,mu,Back(sigma),rho,Back(nu));
                site = site_save;
            }
        }
        return temp;
    case 5:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,nu,rho,Back(sigma),mu,sigma,Back(rho),Back(nu));
                site = site_save;
            }
        }
        return temp;
    case 6:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,Back(nu),Back(rho),sigma,mu,Back(sigma),rho,nu);
                site = site_save;
            }
        }
        return temp;
    case 7:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,Back(nu),rho,Back(sigma),mu,sigma,Back(rho),nu);
                site = site_save;
            }
        }
        return temp;
    case 8:
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu + nu_h)%4;
            for (int rho_h = 0; rho_h < 2; rho_h++) {
                int rho =  (((mu+nu)%2)*((40*(mu+nu) - 6*mu*nu - 18*(mu*mu+nu*nu) + 2*(mu*mu*mu+nu*nu*nu))/12 +rho_h) + ((mu+nu+1)%2)*(mu+1+2*rho_h))%4;
                int sigma = 6 - mu - nu - rho;
                
                temp += gAcc.template getLinkPath<All,HaloDepth>(site,nu,Back(rho),Back(sigma),mu,sigma,rho,Back(nu));
                site = site_save;
            }
        }
        return temp;
    default:
        return gsu3_zero<floatT>();
    }
}

#endif //FAT7LINKCONSTRUCTS_H

