#include "integrator.h"


// inline void WaitEnter() { rootLogger.warn("Press Enter to continue..."); while (std::cin.get()!='\n'); }

template<class floatT, size_t HaloDepth, CompressionType comp=R18>
struct do_evolve_Q
{
    do_evolve_Q(gaugeAccessor<floatT, comp> gAcc,gaugeAccessor<floatT> pAccessor,floatT stepsize) : _stepsize(stepsize),
    _pAccessor(pAccessor), _gAcc(gAcc){}

    
    double _stepsize;
    gaugeAccessor<floatT> _pAccessor;
    gaugeAccessor<floatT, comp> _gAcc;

    __device__ __host__ GSU3<floatT> operator()(gSiteMu site){
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<double> temp;

        temp= su3_exp<double>(GCOMPLEX(double)(0.0,1.0)*_stepsize*_pAccessor.template getLink<double>(site)) 
        *_gAcc.template getLink<double>(site);

        temp.su3unitarize();

        return temp;
    }
};

template<class floatT, size_t HaloDepth>
struct do_evolve_P
{
    do_evolve_P(gaugeAccessor<floatT> ipdotAccessor,gaugeAccessor<floatT> pAccessor,floatT stepsize) : _stepsize(stepsize),
    _pAccessor(pAccessor), _ipdotAccessor(ipdotAccessor){}

    floatT _stepsize;
    gaugeAccessor<floatT> _pAccessor;
    gaugeAccessor<floatT> _ipdotAccessor;

    __device__ __host__ GSU3<floatT> operator()(gSiteMu site){
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<double> temp;

        temp = _pAccessor.template getLink<double>(site);
        temp -= GCOMPLEX(double)(0.0,1.0)*_stepsize *_ipdotAccessor.template getLink<double>(site);

        return temp;
    }
};

template<class floatT, size_t HaloDepth, CompressionType comp=R18>
struct get_gauge_Force
{
    gaugeAccessor<floatT, comp> _gAcc;
    floatT _beta;

    get_gauge_Force(gaugeAccessor<floatT, comp> gAcc, floatT beta) : _gAcc(gAcc), _beta(beta){}

    __device__ __host__ GSU3<floatT> operator()(gSiteMu siteM){
        typedef GIndexer<All,HaloDepth> GInd;
        gSite site(GInd::getSite(siteM.isite));

        GSU3<floatT> temp;
        // temp =- _beta/3.0*symanzikGaugeActionDeriv<floatT,HaloDepth>(_gAcc, site, siteM.mu);
        temp = gauge_force<floatT,HaloDepth,comp>(_gAcc, siteM, _beta);

        return temp;
    }
};


//only for testing 
template<class floatT, size_t HaloDepth>
struct get_mom_tr
{
    gaugeAccessor<floatT> _pAccessor;
    get_mom_tr(gaugeAccessor<floatT> pAccessor): _pAccessor(pAccessor){}

    __device__ __host__ floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;

        floatT ret = 0.0;

        for (int mu = 0; mu < 4; mu++) {
            ret += abs(tr_c(_pAccessor.getLink(GInd::getSiteMu(site, mu))));
        }
        return ret;
    }
};

// this is called from outside, append switch cases if other integration schemes are added
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::integrate(){

    switch(_rhmc_param.integrator())
    {
        case 0:
            SWleapfrog();
            break;

        case 1:
            PQPQP2MN();
            break;

        default:
            rootLogger.error("Unkown integration method!");
    }
}


// Sexton-Weingarten integration scheme
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::SWleapfrog(){

    floatT ieps, iepsh, steph_1f, step_1f, sw_step, sw_steph;

    floatT chmp0=_rhmc_param.mu_f();

    ieps = _rhmc_param.step_size();
    iepsh = 0.5 * ieps;

    step_1f = _rhmc_param.step_size()/_rhmc_param.no_step_1f();
    steph_1f = 0.5* step_1f;

    sw_step = step_1f/_rhmc_param.no_sw();
    sw_steph = 0.5 *sw_step;

    //==================================================//
    // Perform the first half step                      //
    //==================================================//

    updateP_fermforce( iepsh, _phi_2f, true);
    updateP_fermforce( steph_1f, _phi_1f, false);
    
    updateP_gaugeforce( sw_steph );


    rootLogger.info("Done first Leapfrog step");

    //==================================================//
    // Perform the next ( _no_md - 1 ) steps            //
    //==================================================//


    for (int md=1; md<_rhmc_param.no_md(); md++)                  // start loop over steps of 2f
    {
        for (int step=1; step<=_rhmc_param.no_step_1f();step++)   // start loop over steps of 1f
        {
            for (int sw=1; sw<=_rhmc_param.no_sw(); sw++)         // start loop over steps of gauge part
            {
                evolveQ( sw_step );
                updateP_gaugeforce( sw_step );
            }// end loop over steps of gauge part
           if (_rhmc_param.mu_f() !=0){
            _smearing.SmearAll(chmp0);
           }
           else {
               _smearing.SmearAll(); 
           }
            // update P using only the 1f part of the force
            rootLogger.info("strange force:");
            updateP_fermforce( step_1f, _phi_1f, false); 
           
        }// end loop over steps of 1f
        rootLogger.info("light force:");
        // update P using only the 2f part of the force
          updateP_fermforce( ieps, _phi_2f, true); 
       
    }  


    //==================================================//
    // Perform the last half step                       //
    //==================================================// 

    // bring P steph_1f away from the end of the trajectory for 1f part of the force

    for (int step=1; step<_rhmc_param.no_step_1f(); step++)
    {
        for (int sw = 1; sw<=_rhmc_param.no_sw(); sw++)
        {
            evolveQ( sw_step );
            updateP_gaugeforce( sw_step );
        }
        if (_rhmc_param.mu_f() != 0){
             _smearing.SmearAll(chmp0);
        }
        else {
            _smearing.SmearAll();
        }
        updateP_fermforce( step_1f, _phi_1f, false );
    }

    // bring P sw_steph away from the end of the trajectory for gauge part of the force

    for (int sw=1; sw<_rhmc_param.no_sw(); sw++)
    {
        evolveQ( sw_step );
        updateP_gaugeforce( sw_step );
    }
    

    // bring Q to the end of the trajectory

    evolveQ( sw_step );
   if (_rhmc_param.mu_f() != 0){
             _smearing.SmearAll(chmp0);
        }
        else {
            _smearing.SmearAll();
        }

    // bring P to the end of the trajectory by updating with all the forces
    updateP_fermforce( steph_1f, _phi_1f, false); 
    updateP_fermforce( steph_1f, _phi_1f, true); 
    updateP_gaugeforce( sw_steph );
}
// 2MN PQPQP Omelyan integrator on all scales 
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::PQPQP2MN(){

  double ieps, ieps3, ieps6, ieps23;
  double step_1f, step3_1f, step6_1f, step23_1f;
  double sw_step, sw_steph, sw_step3, sw_step6, sw_step23;
  double tauP1, forceP1;
  double tauP2, forceP2;
  double tauQ, tauG, forceG;
  double chmp0;
  double shadow1, shadow2;
  
  ieps = _rhmc_param.step_size();
  ieps3 = ieps/3.0;
  ieps6 = ieps/6.0;
  ieps23 = 2.0*ieps/3.0;
  
  step_1f = _rhmc_param.step_size()/_rhmc_param.no_step_1f()/2.0;
  step3_1f = step_1f/3.0;
  step6_1f = step_1f/6.0;
  step23_1f = 2.0*step_1f/3.0;
  
  sw_step = step_1f/_rhmc_param.no_sw()/2.0;
  sw_steph = sw_step/2.0;
  sw_step3 = sw_step/3.0;
  sw_step6 = sw_step/6.0;
  sw_step23 = 2.0*sw_step/3.0;

  chmp0 = _rhmc_param.mu_f();

  //==================================================//
  // Perform the first half step                      //
  //==================================================//
  
  updateP_fermforce( ieps6, _phi_2f, true);
  forceP2 = forceinfo2();
  tauP2=ieps6;

  updateP_fermforce( step6_1f, _phi_1f, false);    
  forceP1 = forceinfo2();
  tauP1=step6_1f;
  
  updateP_gaugeforce( sw_step6);
  forceG = forceinfo2();
  tauG=sw_step6;

  //calculation of the 2nd order Poisson brackets
  //see 1801.06412
  shadow1=forceP2+(forceP1/pow(2*_rhmc_param.no_step_1f(),2))+(forceG/pow(4*_rhmc_param.no_step_1f()*_rhmc_param.no_sw(),2));

  rootLogger.info( "Done initial P integration step");

  tauQ=0.0;
  //==================================================//
  // Perform the next ( _no_md - 1 ) steps            //
  //==================================================//
  for (int md=1; md< _rhmc_param.no_md(); md++)                  // start loop over steps of 2f
    {
      for (int step=1; step <= _rhmc_param.no_step_1f(); step++)   // start loop over steps of 1f
        {
	  for (int sw=1; sw <= _rhmc_param.no_sw(); sw++)         // start loop over steps of gauge part
            {
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
            }// end loop over steps of gauge part

	  forceinfo();

	  _smearing.SmearAll(chmp0);
	  // update P using only the 1f part of the force
	  rootLogger.info( "strange force:");
	  updateP_fermforce( step23_1f, _phi_1f, false); tauP1+=step23_1f;

	  for (int sw=1; sw <= _rhmc_param.no_sw(); sw++)         // start loop over steps of gauge part
            {
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
            }// end loop over steps of gauge part

	  forceinfo();

	  _smearing.SmearAll(chmp0);
	  // update P using only the 1f part of the force
	  rootLogger.info("strange force:");
	  updateP_fermforce( step3_1f, _phi_1f, false); tauP1+=step3_1f;
        }// end loop over steps of 1f

      rootLogger.info("light force:");
      // update P using only the 2f part of the force
      updateP_fermforce( ieps23, _phi_2f, true); tauP2+=ieps23;

      for (int step=1; step <= _rhmc_param.no_step_1f(); step++)   // start loop over steps of 1f
        {
	  for (int sw=1; sw <= _rhmc_param.no_sw(); sw++)         // start loop over steps of gauge part
            {
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
            }// end loop over steps of gauge part

	  forceinfo();

	  _smearing.SmearAll(chmp0);
	  // update P using only the 1f part of the force
	  rootLogger.info("strange force:");
	  updateP_fermforce( step23_1f, _phi_1f, false); tauP1+=step23_1f;

	  for (int sw=1; sw <= _rhmc_param.no_sw(); sw++)         // start loop over steps of gauge part
            {
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	      evolveQ( sw_steph );             tauQ += sw_steph;
	      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
            }// end loop over steps of gauge part

	  forceinfo();

	  _smearing.SmearAll(chmp0);
	  // update P using only the 1f part of the force
	  rootLogger.info("strange force:");
	  updateP_fermforce( step3_1f, _phi_1f, false); tauP1+=step3_1f;
        }// end loop over steps of 1f

      rootLogger.info("light force:");
      // update P using only the 2f part of the force
      updateP_fermforce( ieps3, _phi_2f, true); tauP2+=ieps3;
    }  

  //==================================================//
  // Perform the last half step                       //
  //==================================================// 
  // bring P steph_1f away from the end of the trajectory for 1f part of the force
  for (int step=1; step <= _rhmc_param.no_step_1f(); step++)
    {
      for (int sw = 1; sw <= _rhmc_param.no_sw(); sw++)
        {
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
        }

      forceinfo();
      
      _smearing.SmearAll(chmp0);
      rootLogger.info("strange force:");
      updateP_fermforce( step23_1f, _phi_1f, false); tauP1 += step23_1f;

      for (int sw = 1; sw <= _rhmc_param.no_sw(); sw++)
        {
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
        }

      forceinfo();

      _smearing.SmearAll(chmp0);
      rootLogger.info("strange force:");
      updateP_fermforce( step3_1f, _phi_1f, false); tauP1+=step3_1f;
    }
    
  rootLogger.info("light forces:");
  // update P using only the 2f part of the force
  updateP_fermforce( ieps23, _phi_2f, true); tauP2+=ieps23;

  for (int step=1; step < _rhmc_param.no_step_1f(); step++)
    {
      for (int sw = 1; sw <= _rhmc_param.no_sw(); sw++)
        {
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
        }

      forceinfo();
      
      _smearing.SmearAll(chmp0);
      rootLogger.info("strange force:");
      updateP_fermforce( step23_1f, _phi_1f, false); tauP1+=step23_1f;

      for (int sw = 1; sw <= _rhmc_param.no_sw(); sw++)
        {
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
	  evolveQ( sw_steph );             tauQ += sw_steph;
	  updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
        }

      forceinfo();

      _smearing.SmearAll(chmp0);
      rootLogger.info("strange force:");
      updateP_fermforce( step3_1f, _phi_1f, false); tauP1+=step3_1f;
    }

  // bring P sw_steph away from the end of the trajectory for gauge part of the force
  for (int sw=1; sw <= _rhmc_param.no_sw(); sw++)
    {
      evolveQ( sw_steph );             tauQ += sw_steph;
      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
      evolveQ( sw_steph );             tauQ += sw_steph;
      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
    }

  forceinfo();

  _smearing.SmearAll(chmp0);
  updateP_fermforce( step23_1f, _phi_1f, false); tauP1+=step23_1f;

  for (int sw=1; sw < _rhmc_param.no_sw(); sw++)
    {
      evolveQ( sw_steph );             tauQ += sw_steph;
      updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
      evolveQ( sw_steph );             tauQ += sw_steph;
      updateP_gaugeforce( sw_step3 );  tauG += sw_step3;
    }

  forceinfo();

  // bring Q to the end of the trajectory
  evolveQ( sw_steph );             tauQ += sw_steph;
  updateP_gaugeforce( sw_step23 ); tauG += sw_step23;
  evolveQ( sw_steph );             tauQ += sw_steph;

  // bring P to the end of the trajectory by updating with all the forces
  updateP_gaugeforce( sw_step6);
  forceG=forceinfo2();
  tauG+=sw_step6;

  _smearing.SmearAll(chmp0);
  updateP_fermforce( step6_1f, _phi_1f, false);
  forceP1=forceinfo2();
  tauP1+=step6_1f;

  updateP_fermforce( ieps6, _phi_2f, true);
  forceP2=forceinfo2();
  tauP2+=ieps6;

  rootLogger.info( "End of trajectory: ");
  rootLogger.info(" ... tauQ  = ", tauQ);
  rootLogger.info(" ... tauG  = ",tauG);
  rootLogger.info(" ... tauP1 = ",tauP1);
  rootLogger.info(" ... tauP2 = ",tauP2);
    
  //calculation of the 2nd order Poisson brackets
  //see 1801.06412
  shadow2=forceP2+(forceP1/pow(2*_rhmc_param.no_step_1f(),2))+(forceG/pow(4*_rhmc_param.no_step_1f()*_rhmc_param.no_sw(),2));
  //Hamiltonian violations to 2nd order
  rootLogger.info("Delta Shadow = ",pow(ieps,2)*(shadow2 - shadow1)/72.0);
}


//update P with the gauge force
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::updateP_gaugeforce(floatT stepsize){

    ipdot.iterateOverBulkAllMu(get_gauge_Force<floatT,HaloDepth,R18>(gAcc, _rhmc_param.beta()));

    evolveP(stepsize);
}

//update P with the fermion force
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::updateP_fermforce(floatT stepsize, 
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, bool light/* std::vector<floatT> rat_coeff*/){

  
    ip_dot_f2_hisq.updateForce(phi,ipdot,light);

    forceinfo();

    evolveP(stepsize);
}

template<class floatT, size_t HaloDepth>
struct trace
{

    trace(gaugeAccessor<floatT> ipdotAccessor) : _ipdotAccessor(ipdotAccessor){}

    gaugeAccessor<floatT> _ipdotAccessor;
    

    __device__ __host__ floatT operator()(gSite site){
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> temp;

        floatT ret =0.0;

        for(int mu=0; mu<4; mu++)
        {

        temp= _ipdotAccessor.getLink(GInd::getSiteMu(site, mu));

        ret += -2.0 * tr_d(temp,temp);

        }


        return ret;
    }
};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::forceinfo(){

    typedef GIndexer<All,HaloDepth> GInd;
    LatticeContainer<onDevice,floatT> force_tr(_p.getComm(), "forcetr");
    force_tr.adjustSize(GInd::getLatData().vol4);

    force_tr.template iterateOverBulk<All, HaloDepth>(trace<floatT, HaloDepth>(ipdotAccessor));

    floatT thing;
    force_tr.reduce(thing, GInd::getLatData().vol4);

    thing = thing /(4* GInd::getLatData().globvol4);



    rootLogger.info("Average force = " ,  thing);


}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
floatT integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::forceinfo2(){

    typedef GIndexer<All,HaloDepth> GInd;
    LatticeContainer<onDevice,floatT> force_tr(_p.getComm(), "forcetr");
    force_tr.adjustSize(GInd::getLatData().vol4);

    force_tr.template iterateOverBulk<All, HaloDepth>(trace<floatT, HaloDepth>(ipdotAccessor));

    floatT thing;
    force_tr.reduce(thing, GInd::getLatData().vol4);

    return thing;

}


// update the gauge field
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::evolveQ(floatT stepsize){

    _gaugeField.iterateOverBulkAllMu(do_evolve_Q<floatT, HaloDepth, R18>(gAcc, pAccessor, stepsize));
    _gaugeField.updateAll();
}

//helper function, called in updateP_Xforce
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::evolveP(floatT stepsize){

    _p.iterateOverBulkAllMu(do_evolve_P<floatT, HaloDepth>(ipdotAccessor, pAccessor, stepsize));
}

//test if momenta are traceless, only used in tests
template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::check_traceless(){

    typedef GIndexer<All,HaloDepth> GInd;
    LatticeContainer<onDevice,floatT> redBase(_p.getComm());
    const size_t elems = GInd::getLatData().vol4;

    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(get_mom_tr<floatT, HaloDepth>(pAccessor));

    floatT momenta;

    redBase.reduce(momenta, elems);

    rootLogger.info("summed trace of momenta: " ,  momenta);
}


// this is called from outside, append switch cases if other integration schemes are added
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::integrate(){

    PureGaugeleapfrog();
}


// Leapfrogger for pure gauge HMC
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::PureGaugeleapfrog(){

    //==================================================//
    // Perform the first half step                      //
    //==================================================//

    updateP_gaugeforce( _rhmc_param.step_size()/2.0 );

    rootLogger.info("Done first Leapfrog step");

    //==================================================//
    // Perform the next ( _no_md - 1 ) steps            //
    //==================================================//

    for (int sw=1; sw<_rhmc_param.no_md(); sw++)
    {
        evolveQ( _rhmc_param.step_size() );
        updateP_gaugeforce( _rhmc_param.step_size() );
    }   

    //==================================================//
    // Perform the last half step                       //
    //==================================================// 

    evolveQ(_rhmc_param.step_size());
    updateP_gaugeforce( _rhmc_param.step_size()/2.0 );
}

//update P with the gauge force
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::updateP_gaugeforce(floatT stepsize){

    ipdot.iterateOverBulkAllMu(get_gauge_Force<floatT,HaloDepth,comp>(gAcc, _rhmc_param.beta()));
    evolveP(stepsize);
    // check_traceless();
}


// update the gauge field
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::evolveQ(floatT stepsize){

    _gaugeField.iterateOverBulkAllMu(do_evolve_Q<floatT, HaloDepth, comp>(gAcc, pAccessor, stepsize));
    _gaugeField.updateAll();
}

//helper function, called in updateP_Xforce
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::evolveP(floatT stepsize){

    _p.iterateOverBulkAllMu(do_evolve_P<floatT, HaloDepth>(ipdotAccessor, pAccessor, stepsize));
}

//test if momenta are traceless, only used in tests
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void pure_gauge_integrator<floatT, onDevice, HaloDepth, comp>::check_traceless(){

    typedef GIndexer<All,HaloDepth> GInd;
    LatticeContainer<onDevice,floatT> redBase(_p.getComm());
    const size_t elems = GInd::getLatData().vol4;

    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(get_mom_tr<floatT, HaloDepth>(pAccessor));

    floatT momenta;

    redBase.reduce(momenta, elems);

    rootLogger.info("summed trace of momenta: " ,  momenta);
}


// explicit instantiation
// template class integrator<float, All, 2>;
#define CLASS1_INIT(floatT, HALO, HALOSPIN)			\
template class integrator<floatT, true, All, HALO, HALOSPIN>;
#define CLASS2_INIT(floatT, HALO, comp) \
template class pure_gauge_integrator<floatT, true, HALO, comp>;

INIT_PHHS(CLASS1_INIT)
INIT_PHC(CLASS2_INIT)

