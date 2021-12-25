#include "hisqSmearing.h"
#include "staggeredPhases.h"


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, CompressionType compLvl1, CompressionType compLvl2, CompressionType compNaik>
void HisqSmearing<floatT, onDevice, HaloDepth, comp, compLvl1, compLvl2, compNaik>::SmearAll(floatT mu_f, bool multiplyPhase) {

    //_gauge_lvl1.iterateOverBulkAllMu(HisqSmearingStruct<floatT, HaloDepth,comp>(_gauge_base.getAccessor(), _Lvl1));
    _dummy.iterateOverBulkAllMu(staple3_lvl1);
    _gauge_lvl1 = _Lvl1._c_1 * _gauge_base + _Lvl1._c_3 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_1_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_2_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_3_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_4_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_5 * _dummy;


    
    _dummy.iterateOverBulkAllMu(staple7_1_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_2_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_3_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_4_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_5_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_6_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_7_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_8_lvl1);
    _gauge_lvl1 = _gauge_lvl1 + _Lvl1._c_7 * _dummy;

    
    _gauge_lvl1.iterateOverBulkAllMu(U3ProjectStruct<floatT, HaloDepth,compLvl1>(_gauge_lvl1.getAccessor()));
    _gauge_lvl1.updateAll();

    _dummy.iterateOverBulkAllMu(staple3_lvl2);
    _gauge_lvl2 = _Lvl2._c_1 * _gauge_lvl1 + _Lvl2._c_3 * _dummy;

    _dummy.iterateOverBulkAllMu(stapleLepage);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_lp * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_1_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_2_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_3_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple5_4_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_5 * _dummy;

    _dummy.iterateOverBulkAllMu(staple7_1_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_2_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_3_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_4_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_5_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_6_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_7_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    _dummy.iterateOverBulkAllMu(staple7_8_lvl2);
    _gauge_lvl2 = _gauge_lvl2 + _Lvl2._c_7 * _dummy;
    
    //_gauge_lvl2.iterateOverBulkAllMu(HisqSmearingStruct<floatT, HaloDepth,compLvl1>(_gauge_lvl1.getAccessor(), _Lvl2));
    
    //if (mu_f != 0){
    //     if (multiplyPhase){
    //    staggeredPhaseKernel<floatT,onDevice,HaloDepth,compLvl2> multPhase(_gauge_lvl2,mu_f);
    //    _gauge_lvl2.iterateOverBulkAllMu(multPhase); 
    //    _gauge_lvl2.updateAll();
    //    }
    //else {
    //   if (multiplyPhase) {
    //    staggeredPhaseKernel<floatT, onDevice, HaloDepth,compLvl2> multPhase(_gauge_lvl2);
    //    _gauge_lvl2.iterateOverBulkAllMu(multPhase);
    //    _gauge_lvl2.updateAll(); 
    //                     }
    //}    
    
   
    
    //if ( mu_f !=0 ) {
    //    if (multiplyPhase) {
    //    staggeredPhaseKernel<floatT, onDevice, HaloDepth,compLvl1> multPhase(_gauge_lvl1,mu_f);
    //    _gauge_lvl1.iterateOverBulkAllMu(multPhase);
    //    _gauge_lvl1.updateAll();
    //    }
    //}
    //else {
    //   if (multiplyPhase) {
    //    staggeredPhaseKernel<floatT, onDevice, HaloDepth,compLvl1> multPhase(_gauge_lvl1);
    //    _gauge_lvl1.iterateOverBulkAllMu(multPhase);
    //    _gauge_lvl1.updateAll(); 
    //                     }
    //}



    if (mu_f != 0) {
        if (multiplyPhase) {
        staggeredPhaseKernel<floatT,onDevice,HaloDepth,compLvl2> multPhase(_gauge_lvl2,mu_f);
        _gauge_lvl2.iterateOverBulkAllMu(multPhase);
                 }
    }
    else {
        if (multiplyPhase) {
        staggeredPhaseKernel<floatT,onDevice,HaloDepth,compLvl2> multPhase(_gauge_lvl2);
        _gauge_lvl2.iterateOverBulkAllMu(multPhase); 
        }
    }
    _gauge_lvl2.updateAll();
    
    if ( mu_f !=0 ) {
        if (multiplyPhase) {
        staggeredPhaseKernel<floatT, onDevice, HaloDepth,compLvl1> multPhase(_gauge_lvl1,mu_f);
        _gauge_lvl1.iterateOverBulkAllMu(multPhase);
        _gauge_lvl1.updateAll();
        }
    }
    else {
       if (multiplyPhase) {
        staggeredPhaseKernel<floatT, onDevice, HaloDepth,compLvl1> multPhase(_gauge_lvl1);
        _gauge_lvl1.iterateOverBulkAllMu(multPhase);
        _gauge_lvl1.updateAll(); 
                         }
    }
    
    //  _gauge_naik.iterateOverBulkAllMu(naiktermStruct<floatT, HaloDepth,compLvl1>(_gauge_lvl1.getAccessor()));
    _gauge_naik.iterateOverBulkAllMu(stapleNaik);
    
    _gauge_naik.updateAll();
}
#define CLASS_INIT(floatT,HALO) \
  template class HisqSmearing<floatT,true,HALO,R14,R18,R18,U3R14>;	\
  template class HisqSmearing<floatT,true,HALO,R18,R18,R18,R18>; \
  template class HisqSmearing<floatT,true,HALO,R18,R18,R18,U3R14>;


INIT_PH(CLASS_INIT)