#ifndef FAT7LINKCONSTRUCTS_H
#define FAT7LINKCONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"
#include "../../modules/HISQ/smearParameters.h"
#include <assert.h>


template<class floatT, size_t HaloDepth, CompressionType comp>
__host__ __device__ GSU3<floatT> inline naikLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {

  typedef GIndexer<All,HaloDepth> GInd;

  gSite origin = GInd::getSite(siteMu.isite);
  int mu = siteMu.mu;
  gSite down = GInd::site_dn(origin,mu);
  gSite up = GInd::site_up(origin,mu);
    
  return gAcc.getLink(GInd::getSiteMu(down,mu))*gAcc.getLink(GInd::getSiteMu(origin,mu))*gAcc.getLink(GInd::getSiteMu(up,mu));
}

template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline threeLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu, int excluded_dir1 = -1, int excluded_dir2 = -1) {
  typedef GIndexer<All,HaloDepth> GInd;
  GSU3<floatT> temp = gsu3_zero<floatT>();
  gSite site = GInd::getSite(siteMu.isite);
  gSite site_save = site;
  int mu = siteMu.mu;
  if(mu == excluded_dir1 || mu == excluded_dir2) return temp*(floatT)NAN;
  if(excluded_dir1 == excluded_dir2) return temp*(floatT)NAN;

  int check_count = 0;
    
  for (int nu_h = 1; nu_h < 4; nu_h++) {
    int nu = (mu+nu_h)%4;
    if(nu == excluded_dir1 || nu == excluded_dir2) continue;
    check_count += 1;
    temp += gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
    site = site_save;
    temp += gAcc.template getLinkPath<All, HaloDepth>(site, Back(nu), mu, nu);
    site = site_save;
  }

  if(excluded_dir1 >= 0 || excluded_dir2 >= 0) assert(check_count == 1); //loop length is 1 in the case of use in hyp smearing
  if(excluded_dir1 >= 0 || excluded_dir2 >= 0) assert(excluded_dir1 >= 0 && excluded_dir2 >= 0); //only one excluded dir not supported

  return temp;
}


//It is assumed that the exluded directions in gaAcc_0, gaAcc_1, gaAcc_2 go in asending order (10 < 20 < 21 < 30 < 31 < 32)
// example:  suppose the exluded directions are 20, 21, 32, (for Vtilde_{i, mu; 2}), then gAcc_0 = _gauge_lvl1_20, gAcc_1 = _gauge_lvl1_21, gAcc_2 = _gauge_lvl1_32
// in other words, the primary excluded direction in this example is 2, so the extra excluded directions go in asending order:  0, 1, 3.
template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline hypThreeLinkStaple_second_level(gaugeAccessor<floatT,comp> gAcc_0, gaugeAccessor<floatT,comp> gAcc_1, gaugeAccessor<floatT,comp> gAcc_2, gSiteMu siteMu, int excluded_dir, gaugeAccessor<floatT,comp> temp_gAcc_mu_excluded_dir, gaugeAccessor<floatT,comp> temp_gAcc_nu_excluded_dir) {

  typedef GIndexer<All,HaloDepth> GInd;
  GSU3<floatT> temp = gsu3_zero<floatT>();
  int mu = siteMu.mu;
  if(mu==excluded_dir) return temp*(floatT)NAN;
  gSite origin = GInd::getSite(siteMu.isite);
  gSite upMu = GInd::site_up(origin,mu);

  //ex = 0, then mu=1,2,3, => 10, 20, 30 (0, 1, 2) = (mu-1, mu-1, mu-1)
  //ex = 1, then mu=0,2,3, => 10, 21, 31 (0, 1, 2) = (mu, mu-1, mu-1)
  //ex = 2, then mu=0,1,3, => 20, 21, 32 (0, 1, 2) = (mu, mu, mu-1)
  //ex = 3, then mu=0,1,2, => 30, 31, 32 (0, 1, 2) = (mu, mu, mu)
  //=>gAcc_idx = mu > excluded_dir ? mu - 1 : mu

  int gAcc_idx = mu > excluded_dir ? mu - 1 : mu;
  assert(gAcc_idx < 3);
  if(gAcc_idx == 0)temp_gAcc_mu_excluded_dir=gAcc_0;
  else if(gAcc_idx == 1)temp_gAcc_mu_excluded_dir=gAcc_1;
  else if(gAcc_idx == 2)temp_gAcc_mu_excluded_dir=gAcc_2;
  else assert(0);

  int nu_add = 0;
  for (int nu_h = 0; nu_h < 3; nu_h++) {

    if(nu_h == excluded_dir) nu_add = 1;
    int nu = nu_h + nu_add; // we want nu here to go in ascending order; if excluded_dir == 1, then nu should go 0, 2, 3
    if(nu == mu) continue;
    //if(nu == excluded_dir) assert(0); //shouldn't occur

    gSite downNu = GInd::site_dn(origin,nu);
    gSite upNu = GInd::site_up(origin,nu);
    gSite upMudownNu = GInd::site_dn(upMu, nu);
    assert(gAcc_idx != nu_h);

    if(nu_h == 0)temp_gAcc_nu_excluded_dir=gAcc_0;
    else if(nu_h == 1)temp_gAcc_nu_excluded_dir=gAcc_1;
    else if(nu_h == 2)temp_gAcc_nu_excluded_dir=gAcc_2;
      
    // nu > 0
    temp += temp_gAcc_mu_excluded_dir.getLink(GInd::getSiteMu(origin,nu))*temp_gAcc_nu_excluded_dir.getLink(GInd::getSiteMu(upNu,mu))*temp_gAcc_mu_excluded_dir.getLinkDagger(GInd::getSiteMu(upMu,nu));
    // nu < 0
    temp += temp_gAcc_mu_excluded_dir.getLinkDagger(GInd::getSiteMu(downNu,nu))*temp_gAcc_nu_excluded_dir.getLink(GInd::getSiteMu(downNu,mu))*temp_gAcc_mu_excluded_dir.getLink(GInd::getSiteMu(upMudownNu,nu));
  }
  return temp;
}

template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline hypThreeLinkStaple_third_level(gaugeAccessor<floatT,comp> gAcc_0, gaugeAccessor<floatT,comp> gAcc_1, gaugeAccessor<floatT,comp> gAcc_2, gaugeAccessor<floatT,comp> gAcc_3, gSiteMu siteMu, gaugeAccessor<floatT,comp> temp_gAcc_mu, gaugeAccessor<floatT,comp> temp_gAcc_nu) {

  typedef GIndexer<All,HaloDepth> GInd;
  GSU3<floatT> temp = gsu3_zero<floatT>();
  GSU3<floatT> temp_chk = gsu3_zero<floatT>();
  int mu = siteMu.mu;
  gSite origin = GInd::getSite(siteMu.isite);
  gSite upMu = GInd::site_up(origin,mu);

  if(mu == 0)temp_gAcc_mu=gAcc_0;
  else if(mu == 1)temp_gAcc_mu=gAcc_1;
  else if(mu == 2)temp_gAcc_mu=gAcc_2;
  else if(mu == 3)temp_gAcc_mu=gAcc_3;
 
  for (int nu_h = 1; nu_h < 4; nu_h++) {
    int nu = (mu+nu_h)%4;
    gSite downNu = GInd::site_dn(origin,nu);
    gSite upNu = GInd::site_up(origin,nu);
    gSite upMudownNu = GInd::site_dn(upMu, nu);

    if(nu == 0)temp_gAcc_nu=gAcc_0;
    else if(nu == 1)temp_gAcc_nu=gAcc_1;
    else if(nu == 2)temp_gAcc_nu=gAcc_2;
    else if(nu == 3)temp_gAcc_nu=gAcc_3;
    
    // nu > 0
    temp += temp_gAcc_mu.getLink(GInd::getSiteMu(origin,nu))*temp_gAcc_nu.getLink(GInd::getSiteMu(upNu,mu))*temp_gAcc_mu.getLinkDagger(GInd::getSiteMu(upMu,nu));
    // nu < 0
    temp += temp_gAcc_mu.getLinkDagger(GInd::getSiteMu(downNu,nu))*temp_gAcc_nu.getLink(GInd::getSiteMu(downNu,mu))*temp_gAcc_mu.getLink(GInd::getSiteMu(upMudownNu,nu));
  }
  assert(!(temp==temp_chk));
  return temp;
}

//project to su3
template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline su3unitarize(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {
  typedef GIndexer<All,HaloDepth> GInd;
  GSU3<floatT> temp = gsu3_zero<floatT>();
  int mu = siteMu.mu;
  gSite origin = GInd::getSite(siteMu.isite);
  temp += gAcc.getLink(GInd::getSiteMu(origin,mu));
  //temp += gAcc.template getLink<All, HaloDepth>(siteMu);
  temp.su3unitarize_hits(9, 0.);
  //temp.su3unitarize();
  return temp;
}



template<class floatT, size_t HaloDepth, CompressionType comp>
__host__ __device__ GSU3<floatT> inline lepageLinkStaple(gaugeAccessor<floatT,comp> gAcc, gSiteMu siteMu) {
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
__host__ __device__ GSU3<floatT> inline fiveLinkStaple(gaugeAccessor<floatT, comp> gAcc, gSiteMu siteMu) {
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
__host__ __device__ GSU3<floatT> inline sevenLinkStaple(gaugeAccessor<floatT, comp> gAcc, gSiteMu siteMu) {
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

