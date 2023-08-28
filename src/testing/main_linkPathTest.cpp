/*
 * main_LinkPathTest.cpp
 *
 */

#include "../simulateqcd.h"

#define PREC double
#define USE_GPU true

template<class floatT, size_t HaloDepth>
struct ThreeLinkExplicit {

    SU3Accessor<floatT> SU3Accessor;

    ThreeLinkExplicit(Gaugefield<floatT, true, HaloDepth> &gaugeIn) : SU3Accessor(gaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;

        SU3<floatT> temp=su3_zero<floatT>();
        gSite site = GInd::getSite(siteMu.isite);

        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (siteMu.mu+nu_h)%4;
            gSite siteUp = GInd::site_up(site,nu);
            gSite siteRight = GInd::site_up(site,siteMu.mu);
            gSite siteDn = GInd::site_dn(site,nu);
            gSite siteDnRight = GInd::site_up_dn(site,siteMu.mu,nu);

            temp+=SU3Accessor.getLink(GInd::getSiteMu(site, nu))
      	          *SU3Accessor.getLink(GInd::getSiteMu(siteUp, siteMu.mu))
      	          *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteRight, nu));

            temp+=SU3Accessor.getLinkDagger(GInd::getSiteMu(siteDn, nu))
      	          *SU3Accessor.getLink(GInd::getSiteMu(siteDn, siteMu.mu))
      	          *SU3Accessor.getLink(GInd::getSiteMu(siteDnRight, nu));
        }
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct ThreeLinkPath {

    SU3Accessor<floatT> SU3Accessor;

    ThreeLinkPath(Gaugefield<floatT, true, HaloDepth> &gaugeIn) : SU3Accessor(gaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;

        SU3<floatT> temp=su3_zero<floatT>();

        gSite site = GInd::getSite(siteMu.isite);
        gSite cp = site;
        int mu = siteMu.mu;
        for (int nu_h = 1; nu_h < 4; nu_h++) {
            int nu = (mu+nu_h)%4;
            site = cp;
            temp+=SU3Accessor.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
            site = cp;
            temp+=SU3Accessor.template getLinkPath<All, HaloDepth>(site, Back(nu), mu, nu);
        }
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct TestLinks {
    SU3Accessor<floatT> firstAccessor;
    SU3Accessor<floatT> secondAccessor;

    TestLinks(Gaugefield<floatT, true, HaloDepth> &gaugeFirst, Gaugefield<floatT, true, HaloDepth> &gaugeSecond) :
      firstAccessor(gaugeFirst.getAccessor()), secondAccessor(gaugeSecond.getAccessor()) {}

    __host__ __device__ int operator() (gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        int tmp = 0;
        for (int mu = 0; mu < 4; mu++) {
            tmp += (firstAccessor.getLink(GInd::getSiteMu(site, mu)) == secondAccessor.getLink(GInd::getSiteMu(site, mu)) ? 0 : 1);
        }
        return tmp;
    }
};

template<class floatT, size_t HaloDepth>
struct FiveLinkExplicit {

    SU3Accessor<floatT> SU3Accessor;

    FiveLinkExplicit(Gaugefield<floatT, true, HaloDepth> &gaugeIn) : SU3Accessor(gaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;

        SU3<floatT> temp=su3_zero<floatT>();
        gSite site = GInd::getSite(siteMu.isite);

        for (int nu_h = 1; nu_h < 3; nu_h++) {
            int nu = (siteMu.mu+nu_h)%4;
            int rho = (siteMu.mu+nu+1)%4;
            gSite siteUp = GInd::site_up(site,nu);
            gSite siteRight = GInd::site_up(site,siteMu.mu);
            gSite siteDn = GInd::site_dn(site,nu);
            gSite siteDnRight = GInd::site_up_dn(site,siteMu.mu,nu);
            gSite siteUpRho = GInd::site_up_up(site,nu,rho);
            gSite siteDnRho = GInd::site_up_dn(site,rho,nu);
            gSite siteUpRight = GInd::site_up_up(site,nu,siteMu.mu);
            gSite siteUpMinusRho = GInd::site_up_dn(site,nu,rho);
            gSite siteUpRightMinusRho = GInd::site_up_up_dn(site,nu,siteMu.mu,rho);
            gSite siteDnMinusRho = GInd::site_dn_dn(site,nu,rho);
            gSite siteDnRightMinusRho = GInd::site_up_dn_dn(site,siteMu.mu,nu,rho);

            temp+=SU3Accessor.getLink(GInd::getSiteMu(site, nu))
      	          *SU3Accessor.getLink(GInd::getSiteMu(siteUp, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteUpRho, siteMu.mu))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteUpRight, rho))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteRight, nu));

            temp+=SU3Accessor.getLink(GInd::getSiteMu(site, nu))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteUpMinusRho, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteUpMinusRho, siteMu.mu))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteUpRightMinusRho, rho))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteRight, nu));

            temp+=SU3Accessor.getLinkDagger(GInd::getSiteMu(siteDn, nu))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDn, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDnRho, siteMu.mu))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteDnRight, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDnRight, nu));

            temp+=SU3Accessor.getLinkDagger(GInd::getSiteMu(siteDn, nu))
                  *SU3Accessor.getLinkDagger(GInd::getSiteMu(siteDnMinusRho, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDnMinusRho, siteMu.mu))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDnRightMinusRho, rho))
                  *SU3Accessor.getLink(GInd::getSiteMu(siteDnRight, nu));
        }
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct FiveLinkPath {
    SU3Accessor<floatT> SU3Accessor;

    FiveLinkPath(Gaugefield<floatT, true, HaloDepth> &GaugeIn) : SU3Accessor(GaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        SU3<floatT> temp = su3_zero<floatT>();
        typedef GIndexer<All,HaloDepth> GInd;
        gSite site = GInd::getSite(siteMu.isite);
        gSite cp = site;
        for (int nu_h = 1; nu_h < 3; nu_h++) {
            int nu = (siteMu.mu+nu_h)%4;
            int rho = (siteMu.mu+nu+1)%4;
            site = cp;
            temp += SU3Accessor.template getLinkPath<All,HaloDepth>(site,nu,rho,siteMu.mu,Back(rho),Back(nu));
            site = cp;
            temp += SU3Accessor.template getLinkPath<All,HaloDepth>(site,nu,Back(rho),siteMu.mu,rho,Back(nu));
            site = cp;
            temp += SU3Accessor.template getLinkPath<All,HaloDepth>(site,Back(nu),rho,siteMu.mu,Back(rho),nu);
            site = cp;
            temp += SU3Accessor.template getLinkPath<All,HaloDepth>(site,Back(nu),Back(rho),siteMu.mu,rho,nu);
        }
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct SevenLinkExplicit {
    SU3Accessor<floatT> SU3Accessor;

    SevenLinkExplicit(Gaugefield<floatT,true,HaloDepth> &GaugeIn) : SU3Accessor(GaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        SU3<floatT> temp = su3_zero<floatT>();

        typedef GIndexer<All,HaloDepth> GInd;
        gSite site = GInd::getSite(siteMu.isite);
        int mu = siteMu.mu;
        for(int nu_h=1; nu_h<4; nu_h++) {
            int nu = (siteMu.mu+nu_h)%4;
            int rho = (siteMu.mu+nu_h+1)%4;
            int sigma = (siteMu.mu+nu_h+2)%4;

            temp += SU3Accessor.getLink(GInd::getSiteMu(site, nu))
                    *SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(site, nu), rho))
   	                *SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up_up(site, nu, rho), sigma))
   	                *SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up_up_up(site, nu, rho, sigma), mu))
   	                *SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up_up(site, nu, rho, mu), sigma))
   	                *SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site, nu, mu), rho))
   	                *SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up(site, mu), nu));
    	}
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct SevenLinkPath {
    SU3Accessor<floatT> SU3Accessor;

    SevenLinkPath(Gaugefield<floatT,true,HaloDepth> &GaugeIn) : SU3Accessor(GaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        SU3<floatT> temp = su3_zero<floatT>();

        typedef GIndexer<All,HaloDepth> GInd;
        gSite site = GInd::getSite(siteMu.isite);
        gSite cp = site;
        for(int nu_h=1; nu_h<4; nu_h++) {
            int nu = (siteMu.mu+nu_h)%4;
            int rho = (siteMu.mu+nu_h+1)%4;
            int sigma = (siteMu.mu+nu_h+2)%4;
            site = cp;
            temp+= SU3Accessor.template getLinkPath<All,HaloDepth>(site,nu,rho,sigma,siteMu.mu,Back(sigma),Back(rho),Back(nu));
        }
        return temp;
    }
};

template<class floatT, size_t HaloDepth>
bool testSmearLink(CommunicationBase &commBase) {
    typedef GIndexer<All,HaloDepth> GInd;
    LatticeContainer<true,floatT> redBase(commBase);
    Gaugefield<floatT, true, HaloDepth> gauge(commBase);
    Gaugefield<floatT, true, HaloDepth> gaugeSmearOld(commBase);
    Gaugefield<floatT, true, HaloDepth> gaugeSmearNew(commBase);
    Gaugefield<floatT, false, HaloDepth> gaugeSmearOld_Host(commBase);
    Gaugefield<floatT, false, HaloDepth> gaugeSmearNew_Host(commBase);
    gauge.readconf_nersc("../test_conf/gauge12750");

    gaugeSmearOld.iterateOverBulkAllMu(ThreeLinkExplicit<floatT, HaloDepth>(gauge));

    gaugeSmearNew.iterateOverBulkAllMu(ThreeLinkPath<floatT, HaloDepth>(gauge));
    gaugeSmearOld_Host=gaugeSmearOld;
    gaugeSmearNew_Host=gaugeSmearNew;
    floatT test = 0.0;
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;

    redBase.adjustSize(elems);
    redBase.template iterateOverBulk<All, HaloDepth>(TestLinks<floatT, HaloDepth>(gaugeSmearOld, gaugeSmearNew));

    redBase.reduce(test, elems);
    gSite site3 = GInd::getSite(1,1,1,1);

    SU3<PREC> test1 = gaugeSmearOld_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));
    SU3<PREC> test2 = gaugeSmearNew_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));

    rootLogger.info(test1.getLink00() ,  " " ,  test1.getLink01() ,  " " ,  test1.getLink10());
    rootLogger.info(test2.getLink00() ,  " " ,  test2.getLink01() ,  " " ,  test2.getLink10());

    if (test > 0.0) return false;

    gaugeSmearOld.iterateOverBulkAllMu(FiveLinkExplicit<floatT, HaloDepth>(gauge));
    gaugeSmearNew.iterateOverBulkAllMu(FiveLinkPath<floatT, HaloDepth>(gauge));
    gaugeSmearOld_Host=gaugeSmearOld;
    gaugeSmearNew_Host=gaugeSmearNew;

    redBase.template iterateOverBulk<All, HaloDepth>(TestLinks<floatT, HaloDepth>(gaugeSmearOld, gaugeSmearNew));
    redBase.reduce(test,elems);

    test1 = gaugeSmearOld_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));
    test2 = gaugeSmearNew_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));

    rootLogger.info(test1.getLink00() ,  " " ,  test1.getLink01() ,  " " ,  test1.getLink10());
    rootLogger.info(test2.getLink00() ,  " " ,  test2.getLink01() ,  " " ,  test2.getLink10());

    if (test > 0.0) return false;

    gaugeSmearOld.iterateOverBulkAllMu(SevenLinkExplicit<floatT, HaloDepth>(gauge));
    gaugeSmearNew.iterateOverBulkAllMu(SevenLinkPath<floatT, HaloDepth>(gauge));
    gaugeSmearOld_Host=gaugeSmearOld;
    gaugeSmearNew_Host=gaugeSmearNew;

    redBase.template iterateOverBulk<All, HaloDepth>(TestLinks<floatT, HaloDepth>(gaugeSmearOld, gaugeSmearNew));
    redBase.reduce(test,elems);

    test1 = gaugeSmearOld_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));
    test2 = gaugeSmearNew_Host.getAccessor().getLink(GInd::getSiteMu(site3, 0));

    rootLogger.info(test1.getLink00() ,  " " ,  test1.getLink01() ,  " " ,  test1.getLink10());
    rootLogger.info(test2.getLink00() ,  " " ,  test2.getLink01() ,  " " ,  test2.getLink10());

    if (test > 0.0) return false;
    else return true;
}


int main(int argc, char **argv) {

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/linkPathTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 0;

    initIndexer(HaloDepth,param, commBase);

    stdLogger.setVerbosity(INFO);
    rootLogger.setVerbosity(INFO);
    rootLogger.info("Testing LinkPath Feature for smearing constructs");
    if (testSmearLink<PREC,HaloDepth>(commBase)) {
        rootLogger.info("Test " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    } else {
        rootLogger.error("Test failed!");
        return -1;
    }

    return 0;
}

