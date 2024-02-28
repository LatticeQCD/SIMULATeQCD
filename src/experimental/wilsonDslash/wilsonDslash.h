
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct WilsonDslashKernel {

    //! The functor has to know about all the elements that it needs for computation.
    //! However, it does not need the Spinor, where the result should go (SpinorOut).
    SU3Accessor<floatT> gAcc;
    SpinorColorAcc<floatT> spinorIn;
    floatT _kappa;
    floatT _c_sw;
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;

    //! Use the constructor to initialize the members
    WilsonDslashKernel(
            Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12> &spinorIn,
            Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
            floatT kappa, floatT c_sw
        ) :
      gAcc(gauge.getAccessor()),
      spinorIn(spinorIn.getAccessor()),
      _kappa(kappa), _c_sw(c_sw), 
      FT(gauge.getAccessor())
  {}

    /*! This is the operator() overload that is called to perform the Dslash. This has to have the following design: It
     * takes a gSite, and it returns the object that we want to write. In this case, we want to return a Vect3<floatT>
     * to store it in another spinor.
     */
    __device__ __host__ inline auto operator()(gSite site) const
    {
      //! We need an indexer to access elements. As the indexer knows about the lattice layout, we do not have to
      //! care about even/odd here explicitly. All that is done by the indexer.
      typedef GIndexer<LatLayout, HaloDepthSpin > GInd;

      /// Define temporary spinor that's 0 everywhere
      ColorVect<floatT> Dirac_psi;

      FourMatrix<floatT> I=FourMatrix<floatT>::identity();
      FourMatrix<floatT> G[4];
      for(int mu=0;mu<4;mu++){
        G[mu]=FourMatrix<floatT>::gamma(mu);
      }
      /// loop through all 4 directions and add result to current site
      for (int mu = 0; mu < 4; mu++) {
        FourMatrix<floatT> P_plus = (I+G[mu]);   
        FourMatrix<floatT> P_minus = (I-G[mu]);

        SU3<floatT> first_term=(gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))));
        SU3<floatT> second_term=gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
        /*
        COMPLEX(floatT) result(0.0,0.0);
        for(int i = 0 ; i < 3 ; i++){
          for(int j = 0 ; j < 3 ; j++){
            result+=first_term(i,j);

            printf("%d %d %lf %lf\n",i,j,result.cREAL,result.cIMAG);
          }
          }*/
  
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
            //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,P_plus.A[i][j].cREAL,P_minus.A[i][j].cREAL);
//          }
//          printf("\n");
//        }

        //floatT bla = 0;
        /* for(int i = 0 ; i < 4 ; i++){ */
        /*   for(int j = 0 ; j < 4 ; j++){ */
        /*      // bla += FourMatrix<floatT>::gamma(0).A[i][j].cIMAG; */
        /*        printf("(%.1f,%.1f) ", FourMatrix<floatT>::gamma(1).A[i][j].cREAL, FourMatrix<floatT>::gamma(1).A[i][j].cIMAG ); */
        /*   } */
        /*  // printf("\n"); */
        /* } */


        //! transport spinor psi(x+mu) to psi(x) with link
//        COMPLEX(floatT) result(0.0,0.0);
//        for(int i = 0 ; i < 4 ; i++){
//          result+=re_dot_prod( (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)))[i],spinorIn.getColorVect(GInd::site_up(site, mu))[i]);
//
//          printf("%d %lf %lf\n",i,result.cREAL,result.cIMAG);
//        }
        Dirac_psi = Dirac_psi - 0.5 * _kappa * ( first_term * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) )
          //! transport spinor psi(x-mu) to psi(x) with link dagger
          + second_term * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) ) );
      }
      //mass term
      floatT M = 1.0; // 1/sqrt(kappa) psi
      Dirac_psi = Dirac_psi + M * spinorIn.getColorVect(site);

      ColorVect<floatT> Clover;
      for(int mu = 0 ; mu < 4 ; mu++){
        for(int nu = mu+1 ; nu < 4 ; nu++){
//          if(mu==nu) continue;
             SU3<floatT> Fmunu = FT(site,mu,nu);
          /*
             COMPLEX(floatT) result(0.0,0.0);
             for(int i = 0 ; i < 3 ; i++){
             for(int j = 0 ; j < 3 ; j++){
             result+=Fmunu(i,j);
             }
             }
             printf("%d %d %d %d %lf %lf\n",mu,nu,result.cREAL,result.cIMAG);
             */
          Clover = Clover - _kappa * (_c_sw/2.0) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
          //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,(G[mu]*G[nu]).A[i][j].cREAL,(G[mu]*G[nu]).A[i][j].cIMAG);
//          }
//          printf("\n");
//        }
/*          COMPLEX(floatT) result(0.0,0.0);
          for(int i = 0 ; i < 4 ; i++){
            result+=re_dot_prod( ((G[mu]*G[nu])[i],  ((G[mu]*G[nu])[i]);

            printf("%d %lf %lf\n",i, result.cREAL,result.cIMAG);
          }
          */
        }
      }
      Dirac_psi = Dirac_psi + Clover;

      /*
      COMPLEX(floatT) res(0.0,0.0);
      for(int i = 0 ; i < 4 ; i++){
        res+= re_dot_prod(Dirac_psi[i], Dirac_psi[i]);
      }
 //     printf("Dirac %lf %lf\n",res.cREAL,res.cIMAG);
 //     */
      return convertColorVectToVect12(Dirac_psi);
    }
};

//! Abstract base class for all kinds of Dslash operators that shall enter the inversion
template<typename floatT, bool onDevice, size_t HaloDepth, typename SpinorLHS_t, typename SpinorRHS_t>
class WilsonDslash {
  private:
    Gaugefield<floatT, true, HaloDepth>& _gauge;
    floatT _kappa, _c_sw;
  public:
    //! This shall be a simple call of the Dslash without involving a constant
    WilsonDslash(Gaugefield<floatT, true, HaloDepth>& gauge, floatT kappa, floatT c_sw) : _gauge(gauge),_kappa(kappa),_c_sw(c_sw) {}

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    void apply(SpinorRHS_t & lhs, SpinorRHS_t & rhs, bool update = true){
        auto kernel = WilsonDslashKernel<floatT, All, All, HaloDepth, HaloDepth>(rhs, _gauge, _kappa, _c_sw);
         lhs.template iterateOverBulk(kernel);
 
    };
};




template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct WilsonDslashKernelEven {

    //! The functor has to know about all the elements that it needs for computation.
    //! However, it does not need the Spinor, where the result should go (SpinorOut).
    SU3Accessor<floatT> gAcc;
    SpinorColorAcc<floatT> spinorIn;
    floatT _kappa;
    floatT _c_sw;
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;

    //! Use the constructor to initialize the members
    WilsonDslashKernelEven(
            Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12> &spinorIn,
            Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
            floatT kappa, floatT c_sw
        ) :
      gAcc(gauge.getAccessor()),
      spinorIn(spinorIn.getAccessor()),
      _kappa(kappa), _c_sw(c_sw), 
      FT(gauge.getAccessor())
  {}

    /*! This is the operator() overload that is called to perform the Dslash. This has to have the following design: It
     * takes a gSite, and it returns the object that we want to write. In this case, we want to return a Vect3<floatT>
     * to store it in another spinor.
     */
    __device__ __host__ inline auto operator()(gSite site) const
    {
      //! We need an indexer to access elements. As the indexer knows about the lattice layout, we do not have to
      //! care about even/odd here explicitly. All that is done by the indexer.
      typedef GIndexer<LatLayout, HaloDepthSpin > GInd;

      /// Define temporary spinor that's 0 everywhere
      ColorVect<floatT> Dirac_psi;
      ColorVect<floatT> Delta_oe;
      ColorVect<floatT> Delta_eo;

      FourMatrix<floatT> I=FourMatrix<floatT>::identity();
      FourMatrix<floatT> G[4];
      for(int mu=0;mu<4;mu++){
        G[mu]=FourMatrix<floatT>::gamma(mu);
      }
      /// loop through all 4 directions and add result to current site
      for (int mu = 0; mu < 4; mu++) {
        FourMatrix<floatT> P_plus = (I+G[mu]);   
        FourMatrix<floatT> P_minus = (I-G[mu]);

        SU3<floatT> first_term=(gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))));
        SU3<floatT> second_term=gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)));
        /*
        COMPLEX(floatT) result(0.0,0.0);
        for(int i = 0 ; i < 3 ; i++){
          for(int j = 0 ; j < 3 ; j++){
            result+=first_term(i,j);

            printf("%d %d %lf %lf\n",i,j,result.cREAL,result.cIMAG);
          }
          }*/
  
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
            //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,P_plus.A[i][j].cREAL,P_minus.A[i][j].cREAL);
//          }
//          printf("\n");
//        }

        //floatT bla = 0;
        /* for(int i = 0 ; i < 4 ; i++){ */
        /*   for(int j = 0 ; j < 4 ; j++){ */
        /*      // bla += FourMatrix<floatT>::gamma(0).A[i][j].cIMAG; */
        /*        printf("(%.1f,%.1f) ", FourMatrix<floatT>::gamma(1).A[i][j].cREAL, FourMatrix<floatT>::gamma(1).A[i][j].cIMAG ); */
        /*   } */
        /*  // printf("\n"); */
        /* } */


        //! transport spinor psi(x+mu) to psi(x) with link
//        COMPLEX(floatT) result(0.0,0.0);
//        for(int i = 0 ; i < 4 ; i++){
//          result+=re_dot_prod( (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)))[i],spinorIn.getColorVect(GInd::site_up(site, mu))[i]);
//
//          printf("%d %lf %lf\n",i,result.cREAL,result.cIMAG);
//        }
        if (LatLayout == Odd){        
        Delta_oe = Delta_oe + 0.5 * ( first_term * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) ) + second_term * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) ) );
        }
        Dirac_psi = Dirac_psi - 0.5 * _kappa * ( first_term * (P_minus * spinorIn.getColorVect(GInd::site_up(site, mu)) )
          //! transport spinor psi(x-mu) to psi(x) with link dagger
          + second_term * (P_plus * spinorIn.getColorVect(GInd::site_dn(site, mu)) ) );
      }

      
      //mass term
      floatT M = 1.0; // 1/sqrt(kappa) psi
      Dirac_psi = Dirac_psi + M * spinorIn.getColorVect(site);

      ColorVect<floatT> Clover;
      ColorVect<floatT> Aoo;
      for(int mu = 0 ; mu < 4 ; mu++){
        for(int nu = mu+1 ; nu < 4 ; nu++){
//          if(mu==nu) continue;
             SU3<floatT> Fmunu = FT(site,mu,nu);
          /*
             COMPLEX(floatT) result(0.0,0.0);
             for(int i = 0 ; i < 3 ; i++){
             for(int j = 0 ; j < 3 ; j++){
             result+=Fmunu(i,j);
             }
             }
             printf("%d %d %d %d %lf %lf\n",mu,nu,result.cREAL,result.cIMAG);
             */
          Clover = Clover - _kappa * (_c_sw/2.0) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
          // Aoo=1 - i c_sw * kappa/2 * \sum_mu \sum_nu sigma_munu * F_munu
          // sigma_munu = i/4[Gmu,Gnu]=i/4(GmuGnu-GnuGmu)=i/2(GmuGnu)
          // Aoo=1 + c_sw * kappa/4 * \sum_mu \smu_nu GmuGnu * F_munu
          // Aoo=1 + c_sw * kappa/2 * \sum_mu \smu_{nu >mu} GmuGnu * F_munu
          if (LatLayout == Odd){        
            Aoo = Aoo + _kappa * (_c_sw/2.0) * ( Fmunu * ((G[mu]*G[nu]) * spinorIn.getColorVect(site) ) );
          }
//        for(int i = 0 ; i < 4 ; ++i){
//          for(int j = 0 ; j < 4 ; ++j){
          //printf("(%lf,%lf) ",FourMatrix<floatT>::gamma(3).A[i][j].cREAL,FourMatrix<floatT>::gamma(3).A[i][j].cIMAG);
//            printf("%d %d (%lf,%lf) ",i,j,(G[mu]*G[nu]).A[i][j].cREAL,(G[mu]*G[nu]).A[i][j].cIMAG);
//          }
//          printf("\n");
//        }
/*          COMPLEX(floatT) result(0.0,0.0);
          for(int i = 0 ; i < 4 ; i++){
            result+=re_dot_prod( ((G[mu]*G[nu])[i],  ((G[mu]*G[nu])[i]);

            printf("%d %lf %lf\n",i, result.cREAL,result.cIMAG);
          }
          */
        }
      }
      Dirac_psi = Dirac_psi + Clover;

      /*
      COMPLEX(floatT) res(0.0,0.0);
      for(int i = 0 ; i < 4 ; i++){
        res+= re_dot_prod(Dirac_psi[i], Dirac_psi[i]);
      }
 //     printf("Dirac %lf %lf\n",res.cREAL,res.cIMAG);
 //     */
      return convertColorVectToVect12(Dirac_psi);
    }
};

//! Abstract base class for all kinds of Dslash operators that shall enter the inversion
template<typename floatT, bool onDevice, size_t HaloDepth, typename SpinorLHS_t, typename SpinorRHS_t>
class WilsonDslashEven {
  private:
    Gaugefield<floatT, true, HaloDepth>& _gauge;
    floatT _kappa, _c_sw;
  public:
    //! This shall be a simple call of the Dslash without involving a constant
    WilsonDslashEven(Gaugefield<floatT, true, HaloDepth>& gauge, floatT kappa, floatT c_sw) : _gauge(gauge),_kappa(kappa),_c_sw(c_sw) {}

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    void apply(SpinorRHS_t & lhs, SpinorRHS_t & rhs, bool update = true){
        auto kernel = WilsonDslashKernelEven<floatT, Even, Even, HaloDepth, HaloDepth>(rhs, _gauge, _kappa, _c_sw);
         lhs.template iterateOverBulk(kernel);
 
    };
};
