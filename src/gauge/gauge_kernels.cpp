template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct plaquetteKernel{

    gaugeAccessor<floatT,comp> gAcc;

    plaquetteKernel(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> temp;

        floatT result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                GSU3<floatT> tmp = gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
                result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct plaquetteKernelSS{

    gaugeAccessor<floatT,comp> gAcc;

    plaquetteKernelSS(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> temp;

        floatT result = 0;
        for (int nu = 1; nu < 3; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                GSU3<floatT> tmp = gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
                result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct plaquetteKernel_double{

    gaugeAccessor<floatT,comp> gAcc;

    plaquetteKernel_double(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ double operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        double result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                GSU3<double> tmp = gAcc.template getLink<double>(GInd::getSiteMu(site, mu));
                             tmp*= gAcc.template getLink<double>(GInd::getSiteMu(GInd::site_up(site, mu),nu));
                             tmp*= gAcc.template getLinkDagger<double>(GInd::getSiteMu(GInd::site_up(site, nu),mu));
                             tmp*= gAcc.template getLinkDagger<double>(GInd::getSiteMu(site, nu));
                result+=tr_d(tmp);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct UtauMinusUsigmaKernel{
    gaugeAccessor<floatT,comp> gAcc;

    UtauMinusUsigmaKernel(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> temp;

        floatT result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                GSU3<floatT> tmp = gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
                if ( mu == 0 ) {
                    result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);
                } else {
                    result -= tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);
                }
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct cloverKernel{

    gaugeAccessor<floatT,comp> gAcc;
    FieldStrengthTensor<floatT,HaloDepth,onDevice,comp> FT;

    cloverKernel(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()), FT(gAcc){ }

    __device__ __host__ floatT operator()(gSite site) {

        GSU3<floatT> Fmunu;

        floatT result = 0;

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                Fmunu = FT( site, mu, nu);
                result += tr_d(Fmunu * Fmunu);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct rectangleKernel{

    gaugeAccessor<floatT,comp> gAcc;

    rectangleKernel(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ floatT operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> temp;

        floatT result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                temp = gAcc.getLink(GInd::getSiteMu(GInd::site_up(site, mu), mu) )
                        * gAcc.getLink(GInd::getSiteMu(GInd::site_2up(site, mu), nu) )
                        * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(site, mu, nu), mu))
                        * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site, nu), mu))
                        * gAcc.getLinkDagger(GInd::getSiteMu(site, nu));
                temp += gAcc.getLink(GInd::getSiteMu(GInd::site_up(site, mu), nu) )
                         * gAcc.getLink(GInd::getSiteMu(GInd::site_up_up(site, mu, nu), nu) )
                         * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_2up(site, nu), mu))
                         * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site, nu), nu))
                         * gAcc.getLinkDagger(GInd::getSiteMu(site, nu));

                result += tr_d(gAcc.getLink(GInd::getSiteMu(site, mu)), temp);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct rectangleKernel_double{

    gaugeAccessor<floatT,comp> gAcc;

    rectangleKernel_double(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ double operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<double> temp;
        GSU3<double> temp2;

        double result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                    temp =  gAcc.template getLink<double>(GInd::getSiteMu(GInd::site_up(site, mu), mu) );
                    temp *= gAcc.template getLink<double>(GInd::getSiteMu(GInd::site_2up(site, mu), nu) );
                    temp *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(GInd::site_up_up(site, mu, nu), mu));
                    temp *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(GInd::site_up(site, nu), mu));
                    temp *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(site, nu));
                    
                    temp2 =  gAcc.template getLink<double>(GInd::getSiteMu(GInd::site_up(site, mu), nu) );
                    temp2 *= gAcc.template getLink<double>(GInd::getSiteMu(GInd::site_up_up(site, mu, nu), nu) );
                    temp2 *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(GInd::site_2up(site, nu), mu));
                    temp2 *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(GInd::site_up(site, nu), nu));
                    temp2 *= gAcc.template getLinkDagger<double>(GInd::getSiteMu(site, nu));

                    temp +=  temp2;

                    temp2 = gAcc.getLink(GInd::getSiteMu(site, mu));

                    result += tr_d(temp2, temp);
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
struct gaugeActKernel_double{

    gaugeAccessor<floatT,comp> gAcc;

    gaugeActKernel_double(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()){ }

    __device__ __host__ double operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<double> m_0;
        GSU3<double> m_3;

        const double g1_r = 5.0/3.0;
        const double g2_r = -1.0/12.0;

        double result = 0;
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                m_0 = g1_r * gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site, mu),nu ) ) *    // m1
                        dagger ( gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site , nu),mu ) ) ); // m2

                //
                //      m2
                //    +----+
                //    |    |
                //  m3|    |
                //    V    |m1
                //         |
                //         |
                //    e    |
                //

                m_0 += g2_r *  gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site , mu),nu ) ) *    // m1
                       gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up_up(site , mu, nu),nu ) ) *         // m1
                       dagger ( gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site , nu),nu ) ) *   // m3
                                gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_2up(site , nu),mu ) )    // m2
                              );

                //
                //         m3
                //    <---------+
                //              |
                //              |m2
                //    e    -----+
                //           m1
                //

                m_0 += g2_r * gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site , mu),mu ) ) *    // m1
                       gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_2up(site , mu),nu ) ) *          // m2
                       dagger ( gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up(site , nu),mu ) ) *  // m3
                                gAcc.template getLink<double>( GInd::getSiteMu  ( GInd::site_up_up(site , mu, nu),mu ) ) // m3
                              );

                //
                //    |
                //  m1|
                //    |
                //    e---->
                //      m2
                //

                m_3 = dagger ( gAcc.template getLink<double>( GInd::getSiteMu  ( site,nu ) ) ) *  // m1
                      gAcc.template getLink<double>( GInd::getSiteMu  ( site,mu ) );              // m2

                result += tr_d ( m_3, m_0 );
            }
        }
        return result;
    }
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct count_faulty_links {
    gaugeAccessor<floatT,comp> gL;
    gaugeAccessor<floatT,comp> gR;
    floatT tol;
    count_faulty_links(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeR, floatT tolerance=1e-6) : gL(GaugeL.getAccessor()), gR(GaugeR.getAccessor()), tol(tolerance) {}

    __host__ __device__ int operator() (gSite site) {
        int sum = 0;
        for (int mu = 0; mu < 4; mu++) {
            gSiteMu siteMu = GIndexer<All,HaloDepth>::getSiteMu(site,mu);
            GSU3<floatT> a = gL.getLink(siteMu);
            GSU3<floatT> b = gR.getLink(siteMu);
            if (!compareGSU3(a, b, tol)) {
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    GCOMPLEX(floatT) diff = a(i, j) - b(i, j);
                    floatT diff_abs = fabs(diff.cREAL);
                    if (diff_abs > tol){
                        printf("Link at site (%i %i %i %i) mu=%i, Matrix-Element (%i,%i) differ by %.4e \n", siteMu.coord.x, siteMu.coord.y, siteMu.coord.z, siteMu.coord.t, mu, i,j, diff_abs);
                    }
                }
                sum++;
            }
        }
        return sum;
    }
};
