#pragma once

#include "../../gauge/gaugefield.h"
#include "smearParameters.h"
#include <type_traits>

// Recursive Fat7 path definitions shared by production smearing and its test.
// The path arithmetic is taken unchanged from main_hisqSmearingTest.cpp.
namespace hisq_smearing {

template <class floatT, size_t HaloDepth, CompressionType comp, class Site = gSite>
__host__ __device__
    Site
    shift_site(Site site, int dir, int sign)
{
    using GInd = GIndexer<All, HaloDepth>;

    if (sign > 0)
        return GInd::site_up(site, dir);
    else
        return GInd::site_dn(site, dir);
}

template <class floatT, size_t HaloDepth, CompressionType comp, class Site = gSite>
__host__ __device__
    SU3<siteReal_t<floatT, Site>>
    dress_link(
        SU3Accessor<floatT, comp> gAcc,
        Site site,
        int mu,
        int dir,
        int sign,
        const SU3<siteReal_t<floatT, Site>> &inner)
{
    using GInd = GIndexer<All, HaloDepth>;

    if (sign > 0)
    {

        // x --U_dir(x)--> x+dir
        // inner goes x+dir -> x+dir+mu
        // U_dir^\dagger(x+mu) closes the staple.

        Site upMu = GInd::site_up(site, mu);

        SU3<siteReal_t<floatT, Site>> left =
            gAcc.getLink(GInd::getSiteMu(site, dir));

        SU3<siteReal_t<floatT, Site>> right =
            gAcc.getLinkDagger(GInd::getSiteMu(upMu, dir));

        return left * inner * right;
    }
    else
    {

        // x --U_dir^\dagger(x-dir)--> x-dir
        // inner goes x-dir -> x-dir+mu
        // U_dir(x-dir+mu) closes the staple.

        Site downDir = GInd::site_dn(site, dir);
        Site downDirUpMu = GInd::site_up(downDir, mu);

        SU3<siteReal_t<floatT, Site>> left =
            gAcc.getLinkDagger(GInd::getSiteMu(downDir, dir));

        SU3<siteReal_t<floatT, Site>> right =
            gAcc.getLink(GInd::getSiteMu(downDirUpMu, dir));

        return left * inner * right;
    }
}

template <class floatT, size_t HaloDepth, CompressionType comp>
struct RecursiveFat7Lvl1
{

    SU3Accessor<floatT, comp> gAcc;

    floatT c1;
    floatT c3;
    floatT c5;
    floatT c7;

    RecursiveFat7Lvl1(
        SU3Accessor<floatT, comp> acc,
        const SmearingParameters<floatT> &p)
        : gAcc(acc),
          c1(p._c_1),
          c3(p._c_3),
          c5(p._c_5),
          c7(p._c_7)
    {
    }

    // Lane sites (laneSite.h) only for R18 input.
    template <class SiteMu, class = std::enable_if_t<comp == R18 || !isLaneSite<SiteMu>>>
    __host__ __device__
        SU3<siteReal_t<floatT, SiteMu>>
        operator()(SiteMu siteMu)
    {
        using GInd = GIndexer<All, HaloDepth>;
        using Real = siteReal_t<floatT, SiteMu>;

        const int mu = siteMu.mu;
        const auto x = GInd::siteOf(siteMu);

        // 1-link term
        SU3<Real> result =
            c1 * gAcc.getLink(GInd::getSiteMu(x, mu));

        // Outer transverse direction: nu
        for (int nu = 0; nu < 4; ++nu)
        {

            if (nu == mu)
                continue;

            for (int sNu = -1; sNu <= 1; sNu += 2)
            {

                const auto y =
                    shift_site<floatT, HaloDepth, comp>(x, nu, sNu);

                // If we stop here, dressing by nu generates the
                // ordinary 3-link contribution.
                SU3<Real> innerNu =
                    c3 * gAcc.getLink(GInd::getSiteMu(y, mu));

                // Next transverse direction: rho
                for (int rho = 0; rho < 4; ++rho)
                {

                    if (rho == mu || rho == nu)
                        continue;

                    for (int sRho = -1; sRho <= 1; sRho += 2)
                    {

                        const auto z =
                            shift_site<floatT, HaloDepth, comp>(
                                y, rho, sRho);

                        // Dressing this by rho and then nu
                        // generates the 5-link contribution.
                        SU3<Real> innerRho =
                            c5 * gAcc.getLink(
                                     GInd::getSiteMu(z, mu));

                        // Last transverse direction: sigma
                        for (int sigma = 0; sigma < 4; ++sigma)
                        {

                            if (sigma == mu ||
                                sigma == nu ||
                                sigma == rho)
                                continue;

                            for (int sSigma = -1;
                                 sSigma <= 1;
                                 sSigma += 2)
                            {

                                const auto w =
                                    shift_site<
                                        floatT,
                                        HaloDepth,
                                        comp>(
                                        z,
                                        sigma,
                                        sSigma);

                                // Central link of the 7-link path.
                                SU3<Real> leaf =
                                    c7 * gAcc.getLink(
                                             GInd::getSiteMu(w, mu));

                                innerRho +=
                                    dress_link<
                                        floatT,
                                        HaloDepth,
                                        comp>(
                                        gAcc,
                                        z,
                                        mu,
                                        sigma,
                                        sSigma,
                                        leaf);
                            }
                        }

                        innerNu +=
                            dress_link<
                                floatT,
                                HaloDepth,
                                comp>(
                                gAcc,
                                y,
                                mu,
                                rho,
                                sRho,
                                innerRho);
                    }
                }

                result +=
                    dress_link<
                        floatT,
                        HaloDepth,
                        comp>(
                        gAcc,
                        x,
                        mu,
                        nu,
                        sNu,
                        innerNu);
            }
        }

        return result;
    }
};

template <class floatT, size_t HaloDepth, CompressionType comp>
struct RecursiveFat7Lvl2
{
    SU3Accessor<floatT, comp> gAcc;

    floatT c1;
    floatT c3;
    floatT c5;
    floatT c7;
    floatT cLp;

    RecursiveFat7Lvl2(
        SU3Accessor<floatT, comp> acc,
        const SmearingParameters<floatT> &p)
        : gAcc(acc),
          c1(p._c_1),
          c3(p._c_3),
          c5(p._c_5),
          c7(p._c_7),
          cLp(p._c_lp)
    {
    }

    // Lane sites (laneSite.h) only for R18 input.
    template <class SiteMu, class = std::enable_if_t<comp == R18 || !isLaneSite<SiteMu>>>
    __host__ __device__
        SU3<siteReal_t<floatT, SiteMu>>
        operator()(SiteMu siteMu)
    {
        using GInd = GIndexer<All, HaloDepth>;
        using Real = siteReal_t<floatT, SiteMu>;

        const int mu = siteMu.mu;
        const auto x = GInd::siteOf(siteMu);

        // ========================================================
        // 1-link
        // ========================================================

        SU3<Real> result =
            c1 * gAcc.getLink(
                     GInd::getSiteMu(x, mu));

        // ========================================================
        // Outer transverse direction nu
        //
        // This outer dressing is shared by:
        //
        //   3-link
        //   Lepage
        //   5-link
        //   7-link
        //
        // ========================================================

        for (int nu = 0; nu < 4; ++nu)
        {
            if (nu == mu)
                continue;

            for (int sNu = -1; sNu <= 1; sNu += 2)
            {
                const auto y =
                    shift_site<floatT, HaloDepth, comp>(
                        x,
                        nu,
                        sNu);

                // ------------------------------------------------
                // 3-link contribution
                //
                // After the final outer nu dressing:
                //
                //   nu, mu, -nu
                //
                // ------------------------------------------------

                SU3<Real> innerNu =
                    c3 * gAcc.getLink(
                             GInd::getSiteMu(y, mu));

                // ------------------------------------------------
                // Lepage contribution
                //
                // Positive orientation:
                //
                //   nu, nu, mu, -nu, -nu
                //
                // Negative orientation:
                //
                //  -nu, -nu, mu, +nu, +nu
                //
                // One inner nu dressing is constructed here;
                // the second nu dressing is the common outer
                // dressing below.
                // ------------------------------------------------

                const auto y2 =
                    shift_site<floatT, HaloDepth, comp>(
                        y,
                        nu,
                        sNu);

                SU3<Real> lepageCenter =
                    cLp * gAcc.getLink(
                              GInd::getSiteMu(y2, mu));

                innerNu +=
                    dress_link<
                        floatT,
                        HaloDepth,
                        comp>(
                        gAcc,
                        y,
                        mu,
                        nu,
                        sNu,
                        lepageCenter);

                // =================================================
                // rho level:
                // generates 5-link + contains 7-link subtree
                // =================================================

                for (int rho = 0; rho < 4; ++rho)
                {
                    if (rho == mu || rho == nu)
                        continue;

                    for (int sRho = -1;
                         sRho <= 1;
                         sRho += 2)
                    {
                        const auto z =
                            shift_site<
                                floatT,
                                HaloDepth,
                                comp>(
                                y,
                                rho,
                                sRho);

                        // -----------------------------------------
                        // 5-link center
                        // -----------------------------------------

                        SU3<Real> innerRho =
                            c5 * gAcc.getLink(
                                     GInd::getSiteMu(z, mu));

                        // =========================================
                        // sigma level:
                        // generates 7-link contribution
                        // =========================================

                        for (int sigma = 0;
                             sigma < 4;
                             ++sigma)
                        {
                            if (sigma == mu ||
                                sigma == nu ||
                                sigma == rho)
                                continue;

                            for (int sSigma = -1;
                                 sSigma <= 1;
                                 sSigma += 2)
                            {
                                const auto w =
                                    shift_site<
                                        floatT,
                                        HaloDepth,
                                        comp>(
                                        z,
                                        sigma,
                                        sSigma);

                                SU3<Real> leaf =
                                    c7 * gAcc.getLink(
                                             GInd::getSiteMu(
                                                 w,
                                                 mu));

                                innerRho +=
                                    dress_link<
                                        floatT,
                                        HaloDepth,
                                        comp>(
                                        gAcc,
                                        z,
                                        mu,
                                        sigma,
                                        sSigma,
                                        leaf);
                            }
                        }

                        // Dress the 5/7 subtree with rho.

                        innerNu +=
                            dress_link<
                                floatT,
                                HaloDepth,
                                comp>(
                                gAcc,
                                y,
                                mu,
                                rho,
                                sRho,
                                innerRho);
                    }
                }

                // =================================================
                // Common outer nu dressing.
                //
                // This is where the saving happens:
                //
                // one pair of multiplications dresses the SUM of
                //
                //   c3 contribution
                // + Lepage contribution
                // + all 5-link contributions
                // + all 7-link contributions
                //
                // =================================================

                result +=
                    dress_link<
                        floatT,
                        HaloDepth,
                        comp>(
                        gAcc,
                        x,
                        mu,
                        nu,
                        sNu,
                        innerNu);
            }
        }

        return result;
    }
};

} // namespace hisq_smearing
