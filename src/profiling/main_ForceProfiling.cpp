#include "../define.h"
#include "../gauge/gaugefield.h"
#include "../spinor/spinorfield.h"
#include "../modules/HISQ/smearParameters.h"
#include "../modules/HISQ/hisqSmearing.h"
#include "../modules/HISQ/hisqForce.h"


#define PREC float
int main(int argc, char *argv[]) {

  stdLogger.setVerbosity(INFO);

  CommunicationBase commBase(&argc, &argv);
  RhmcParameters rhmc_param;

  rhmc_param.readfile(commBase,"../parameter/profiling/force_profiling.param", argc, argv);

  commBase.init(rhmc_param.nodeDim());
  
  const size_t HaloDepth = 2;
  const size_t HaloDepthSpin = 4;
  initIndexer(HaloDepth,rhmc_param,commBase);
  initIndexer(HaloDepthSpin,rhmc_param,commBase);
  RationalCoeff rat;

  rat.readfile(commBase, rhmc_param.rat_file(), argc, argv);
    
  Gaugefield<PREC, true, HaloDepth, R18> gauge(commBase);
  Gaugefield<PREC, true, HaloDepth> gaugeLvl2(commBase,"SHARED_GAUGELVL2");
  Gaugefield<PREC, true, HaloDepth,U3R14> gaugeNaik(commBase, "SHARED_GAUGENAIK");
  Gaugefield<PREC, true, HaloDepth> force(commBase);
  Gaugefield<PREC, false, HaloDepth> force_host(commBase);
  Spinorfield<PREC, true, Even, HaloDepthSpin> SpinorIn(commBase);
  Spinorfield<PREC, true, Even, HaloDepthSpin, 14> SpinorOutMulti(commBase,"SHARED_tmp");
  Spinorfield<PREC, true, Even, HaloDepthSpin> spinortmp(commBase);
  grnd_state<false> h_rand;
  grnd_state<true> d_rand;
  
  h_rand.make_rng_state(rhmc_param.seed());

  
  d_rand = h_rand;
  
  SpinorIn.gauss(d_rand.state);
  gauge.random(d_rand.state);
  
  
  AdvancedMultiShiftCG<PREC,12> CG;
  AdvancedMultiShiftCG<PREC,14> CG14;
  HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin,1> dslash(gaugeLvl2,gaugeNaik,0.0);
  HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 12> dslash_multi(gaugeLvl2,gaugeNaik,0.0);
  HisqSmearing<PREC, true, HaloDepth,R18> smearing(gauge,gaugeLvl2,gaugeNaik);
  HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18> ip_dot_f2_hisq(gauge,force,CG,dslash,dslash_multi,rhmc_param,rat,smearing);
 
  smearing.SmearAll();
 
  SimpleArray<PREC, 14> shifts(0.0);
  
  for (size_t i = 0; i <rat.r_2f_den.get().size(); ++i)
    
    {
      
      shifts[i] = rat.r_2f_den[i] + rhmc_param.m_ud()*rhmc_param.m_ud();;
      
    }

  CG14.invert(dslash,SpinorOutMulti,SpinorIn,shifts,rhmc_param.cgMax(), rhmc_param.residue());
  
  SpinorIn = (PREC)rat.r_2f_const() * SpinorIn;
  
  for (size_t i = 0; i < rat.r_2f_den.get().size(); ++i)
    
    {
      
      spinortmp.copyFromStackToStack(SpinorOutMulti, 0, i);
      
      SpinorIn = SpinorIn + (PREC)rat.r_2f_num[i]*spinortmp;
      
    }
  SpinorIn.updateAll();
  
  
  MemoryManagement::memorySummary();
  
  StopWatch<true> timer;
  timer.start();
  
  ip_dot_f2_hisq.updateForce(SpinorIn,force,true);
  
  timer.stop();
    
  force_host=force;
  
  GSU3<PREC> test1 = force_host.getAccessor().getLink(GIndexer<All,HaloDepth>::getSiteMu(0,0,0,0,0));
  
  rootLogger.info("Time: " ,  timer);
		       
  rootLogger.info("Force parallelGpu:");
  
  rootLogger.info(test1.getLink00(), test1.getLink01(), test1.getLink02(), test1.getLink10());
  
    
		       
    
  return 0;
}
