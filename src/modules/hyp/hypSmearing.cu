#include "hypSmearing.h"

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void HypSmearing<floatT, onDevice, HaloDepth, comp>::SmearTest(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out) {

    _dummy.iterateOverBulkAllMu(staple3_lvl1_10);
    gauge_out = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(gauge_out);

}

// also call updateAll()
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void HypSmearing<floatT, onDevice, HaloDepth, comp>::Su3Unitarize(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out){

    HypStaple<floatT, HaloDepth, comp, 4> su_3_unitarize(gauge_out.getAccessor(), gauge_out.getAccessor(), gauge_out.getAccessor(), gauge_out.getAccessor());
    gauge_out.iterateOverBulkAllMu(su_3_unitarize);
    if(update_all)gauge_out.updateAll();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void HypSmearing<floatT, onDevice, HaloDepth, comp>::SmearAll(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out) {

    // create level 1 fields
    _dummy.iterateOverBulkAllMu(staple3_lvl1_10);
    _gauge_lvl1_10 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_10);

    _dummy.iterateOverBulkAllMu(staple3_lvl1_20);
    _gauge_lvl1_20 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_20);

    _dummy.iterateOverBulkAllMu(staple3_lvl1_30);
    _gauge_lvl1_30 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_30);

    _dummy.iterateOverBulkAllMu(staple3_lvl1_21);
    _gauge_lvl1_21 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_21);

    _dummy.iterateOverBulkAllMu(staple3_lvl1_31);
    _gauge_lvl1_31 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_31);

    _dummy.iterateOverBulkAllMu(staple3_lvl1_32);
    _gauge_lvl1_32 = (1-params.alpha_3) * _gauge_base + params.alpha_3/2 * _dummy;
    Su3Unitarize(_gauge_lvl1_32);

    // now that we have level 1 fields, create level 2 staples
    // note:  the order of the gauge fields goes in ascending order (10 < 20 < 30, 10 < 21 < 31, 20 < 21 < 32, 30 < 31 < 32)
    // this is ASSUMED by HypStaple<floatT, HaloDepth, comp, 2>; DO NOT change this order without also modifying HypStaple<floatT, HaloDepth, comp, 2> and threeLinkStaple_second_level
    HypStaple<floatT, HaloDepth, comp, 2> staple3_lvl2_0(_gauge_lvl1_10.getAccessor(), _gauge_lvl1_20.getAccessor(), _gauge_lvl1_30.getAccessor(), _dummy.getAccessor(), 0);
    HypStaple<floatT, HaloDepth, comp, 2> staple3_lvl2_1(_gauge_lvl1_10.getAccessor(), _gauge_lvl1_21.getAccessor(), _gauge_lvl1_31.getAccessor(), _dummy.getAccessor(), 1);
    HypStaple<floatT, HaloDepth, comp, 2> staple3_lvl2_2(_gauge_lvl1_20.getAccessor(), _gauge_lvl1_21.getAccessor(), _gauge_lvl1_32.getAccessor(), _dummy.getAccessor(), 2);
    HypStaple<floatT, HaloDepth, comp, 2> staple3_lvl2_3(_gauge_lvl1_30.getAccessor(), _gauge_lvl1_31.getAccessor(), _gauge_lvl1_32.getAccessor(), _dummy.getAccessor(), 3);

    //second level fields
    _dummy.iterateOverBulkAllMu(staple3_lvl2_0);
    _gauge_lvl2_0 = (1-params.alpha_2) * _gauge_base + params.alpha_2/4 * _dummy;
    Su3Unitarize(_gauge_lvl2_0);

    _dummy.iterateOverBulkAllMu(staple3_lvl2_1);
    _gauge_lvl2_1 = (1-params.alpha_2) * _gauge_base + params.alpha_2/4 * _dummy;
    Su3Unitarize(_gauge_lvl2_1);

    _dummy.iterateOverBulkAllMu(staple3_lvl2_2);
    _gauge_lvl2_2 = (1-params.alpha_2) * _gauge_base + params.alpha_2/4 * _dummy;
    Su3Unitarize(_gauge_lvl2_2);

    _dummy.iterateOverBulkAllMu(staple3_lvl2_3);
    _gauge_lvl2_3 = (1-params.alpha_2) * _gauge_base + params.alpha_2/4 * _dummy;
    Su3Unitarize(_gauge_lvl2_3);

    // now that we have level 2 fields, create level 3 staple
    HypStaple<floatT, HaloDepth, comp, 1> staple3_lvl3(_gauge_lvl2_0.getAccessor(), _gauge_lvl2_1.getAccessor(), _gauge_lvl2_2.getAccessor(), _gauge_lvl2_3.getAccessor());
 
    _dummy.iterateOverBulkAllMu(staple3_lvl3);

    // OLD VERSION, (MAYBE) DOES NOT WORK FOR SOME REASON
    //_gauge_lvl2_0 = (1-params.alpha_1) * _gauge_base + params.alpha_1/6 * _dummy; //reused _gauge_lvl2_0
    //Su3Unitarize(_gauge_lvl2_0);
    
    // NEW VERSION, USES EXTRA FIELD RATHER THAN REUSE _gauge_lvl2_0
    gauge_out = (1-params.alpha_1) * _gauge_base + params.alpha_1/6 * _dummy; //reused _gauge_lvl2_0
    Su3Unitarize(gauge_out);
    
}

#define CLASS_INIT(floatT,HALO) \
  template class HypSmearing<floatT,true,HALO,R18>;


INIT_PH(CLASS_INIT)
