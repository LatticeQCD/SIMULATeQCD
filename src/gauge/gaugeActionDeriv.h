//
// Created by Lukas Mazur on 06.07.18.
//

#pragma once
#include "../define.h"
#include "../base/math/complex.h"
#include "../base/gutils.h"
#include "../base/math/su3array.h"
#include "../base/math/su3.h"
#include "constructs/plaqConstructs.h"
#include "gaugefield.h"



template<class floatT,size_t HaloDepth>
__host__ __device__ SU3<floatT> inline gaugeActionDerivPlaq(SU3Accessor<floatT> gAcc, gSite site, int mu) {
    SU3<floatT> result = su3_zero<floatT>();
    SU3<floatT> tmp = su3_zero<floatT>();

    for (int nu_aux = 1; nu_aux < 4; ++nu_aux) {
        const int nu = (mu + nu_aux) % 4;
        tmp = Plaq_P<floatT,HaloDepth>(gAcc, site, mu, nu) + dagger(Plaq_Q<floatT,HaloDepth>(gAcc, site, mu, nu));

        result += tmp;

    }
    result.TA();
    return result;
}

template<class floatT,size_t HaloDepth>
__host__ __device__ SU3<floatT> inline gaugeActionDerivRect(SU3Accessor<floatT> gAcc, gSite site, int mu) {
    typedef GIndexer<All,HaloDepth> GInd;
    SU3<floatT> result = su3_zero<floatT>();
    SU3<floatT> tmp = su3_zero<floatT>();

    SU3<floatT> P, Q, R, S;
    SU3<floatT> Pmu, Pnu, Qmu, Qnu;
    SU3<floatT> LinkNu, LinkDNu;
    SU3<floatT> LinkNu_nu, LinkDNu_nu;
    SU3<floatT> LinkMu, LinkDMu;

    for (int nu_aux = 1; nu_aux < 4; ++nu_aux) {
        const int nu = (mu + nu_aux) % 4;
        P = Plaq_P<floatT,HaloDepth>(gAcc, site, mu, nu);
        Q = Plaq_Q<floatT,HaloDepth>(gAcc, site, mu, nu);
        R = Plaq_R<floatT,HaloDepth>(gAcc, site, mu, nu);
        S = Plaq_S<floatT,HaloDepth>(gAcc, site, mu, nu);

        Pmu = Plaq_P<floatT,HaloDepth>(gAcc, GInd::site_up(site, mu), mu, nu);
        Pnu = Plaq_P<floatT,HaloDepth>(gAcc, GInd::site_up(site, nu), mu, nu);
        Qmu = Plaq_Q<floatT,HaloDepth>(gAcc, GInd::site_up(site, mu), mu, nu);
        Qnu = Plaq_Q<floatT,HaloDepth>(gAcc, GInd::site_dn(site, nu), mu, nu);

        LinkMu = gAcc.getLink(GInd::getSiteMu(site, mu));
        LinkDMu = dagger(LinkMu);
        LinkNu = gAcc.getLink(GInd::getSiteMu(site, nu));
        LinkDNu = dagger(LinkNu);

        LinkNu_nu = gAcc.getLink(GInd::getSiteMu(GInd::site_dn(site, nu), nu));
        LinkDNu_nu = dagger(LinkNu_nu);

        tmp = P * S - R * Q
              + LinkMu * Pmu * LinkDMu * P
              - Q * LinkMu * Qmu * LinkDMu
              - LinkDNu_nu * Qnu * LinkNu_nu * Q
              + P * LinkNu * Pnu * LinkDNu;

        result += tmp;


    }
    result.TA();

    return result;
}

template<class floatT,size_t HaloDepth>
__host__ __device__ SU3<floatT> inline symanzikGaugeActionDeriv(SU3Accessor<floatT> latacc, gSite s, int mu) {
    typedef GIndexer<All,HaloDepth> GInd;
   // SU3<floatT> tmp = (5. / 3.) * gaugeActionDerivPlaq<floatT,HaloDepth>(gAcc, site, mu) -
     //                  (1. / 12.) * gaugeActionDerivRect<floatT,HaloDepth>(gAcc, site, mu);
    //return tmp;

    //const gSite s(GInd::getSite(site.isite));
    //int mu = site.mu;

    const floatT g_c1 = -5.0/3.0;
    const floatT g_c2 = 1.0/12.0;

    SU3<floatT> m_0, m_res1, staple;

    staple = su3_zero<floatT>();

    for (int nu_aux = 1; nu_aux < 4; nu_aux++)
    {
        const int nu = (mu + nu_aux) % 4;
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_2dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2dn(s, nu), mu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_2dn(s, nu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));
        staple += g_c2 * m_0;

        // m_0 , m_res1 frei
        m_0 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up(s, nu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));
        staple += g_c2 * m_0;


        //m_res1, m_0 frei
        m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_2up(s, mu), nu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), mu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));

        // m_0 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up_dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));

        m_0 += m_res1;

        m_res1 = g_c2 * latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), mu)) * m_0;
        staple +=  m_res1;

        // m_0, m_res1 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, nu, mu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, mu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

        // m_res1 frei
        m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
        m_res1 = m_res1 * latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
        m_res1*= (g_c1 * latacc.getLinkDagger(GInd::getSiteMu(s, nu)) + g_c2 * m_0);
        staple += m_res1;

        // m_0 , m_res1 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), mu));
        m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), nu));
        m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

        // m_res1 frei
        m_res1 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
        m_res1 = m_res1* latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
        m_res1*= (g_c1 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu)) + g_c2 * m_0);
        staple += m_res1;


    } //   nu_aux


    m_res1 = floatT(-1.0)*latacc.getLink(GInd::getSiteMu(s, mu)) * staple;

    m_res1.TA();

    return m_res1;
}


//up to an additional factor of -beta/3 identical to symanikGaugeActionDeriv but faster
template<class floatT, size_t HaloDepth, CompressionType comp=R18>
__host__ __device__ SU3<floatT> inline gauge_force(SU3Accessor<floatT,comp> latacc, gSiteMu site, floatT beta){

    typedef GIndexer<All,HaloDepth> GInd;

    const gSite s(GInd::getSite(site.isite));
    int mu = site.mu;

    const floatT g_c1 = -5.0/3.0;
    const floatT g_c2 = 1.0/12.0;

    //CAVE: In contrast to std. textbook definitions of the gauge action: We use a definition inherited from MILC!
    //      Therefore we find an additional factor of 3/5 in r_1!
    const COMPLEX(floatT) r_1 = COMPLEX(floatT)(beta / 3.0, 0.0)* 3.0/5.0; //


    SU3<floatT> m_0, m_res1, staple;

    staple = su3_zero<floatT>();

    for (int nu_aux = 1; nu_aux < 4; nu_aux++)
      {
        const int nu = (mu + nu_aux) % 4;
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_2dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2dn(s, nu), mu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_2dn(s, nu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));
        staple += g_c2 * m_0;

        // m_0 , m_res1 frei
        m_0 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up(s, nu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));
        staple += g_c2 * m_0;


        //m_res1, m_0 frei
        m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_2up(s, mu), nu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), mu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
        m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));

        // m_0 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up_dn(s, mu, nu), nu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));

        m_0 += m_res1;

        m_res1 = g_c2 * latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), mu)) * m_0;
        staple +=  m_res1;

        // m_0, m_res1 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, nu, mu), mu));
        m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, mu), nu));
        m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

        // m_res1 frei
        m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
        m_res1 = m_res1 * latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
        m_res1*= (g_c1 * latacc.getLinkDagger(GInd::getSiteMu(s, nu)) + g_c2 * m_0);
        staple += m_res1;

        // m_0 , m_res1 frei
        m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), mu));
        m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), nu));
        m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

        // m_res1 frei
        m_res1 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
        m_res1 = m_res1* latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
        m_res1*= (g_c1 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu)) + g_c2 * m_0);
        staple += m_res1;


      } //   nu_aux


    m_res1 = r_1 * latacc.getLink(GInd::getSiteMu(s, mu)) * staple;

    m_res1.TA();

    return m_res1;




}



