/*
 * hisqSmearing.h
 *
 * The methods related to HISQ smearing, which give a better approach to the continuum limit.
 *
 */

#pragma once

#include "../../gauge/gaugefield.h"
#include "../../gauge/constructs/fat7LinkConstructs.h"
#include "../../gauge/constructs/projectU3Constructs.h"
#include "../../gauge/constructs/naikConstructs.h"
#include "smearParameters.h"

// Staples that appear in the fat link. Each staple object is one local staple.
template<class floatT, size_t HaloDepth, CompressionType comp, int linkNumber, int partNumber = 0>
class staple {
private:
    gaugeAccessor<floatT, comp> _gAcc;
public:
    staple(gaugeAccessor<floatT, comp> gAccIn) : _gAcc(gAccIn) {}
    __host__ __device__ GSU3<floatT> operator() (gSiteMu site) {
        switch (linkNumber) {
        case 3:
            return threeLinkStaple<floatT, HaloDepth, comp>(_gAcc, site);
        case 5:
            return fiveLinkStaple<floatT, HaloDepth, comp, partNumber>(_gAcc, site);
        case 7:
            return sevenLinkStaple<floatT, HaloDepth, comp, partNumber>(_gAcc, site);
        case -5:
            return lepageLinkStaple<floatT, HaloDepth, comp>(_gAcc, site);
        case -3:
            return naikLinkStaple<floatT, HaloDepth, comp>(_gAcc, site);
        default:
            return gsu3_zero<floatT>();
        }
    }
};

// Project back to U(3). Needed after first level of smearing.
template<class floatT,size_t HaloDepth, CompressionType comp>
struct U3ProjectStruct{
    gaugeAccessor<floatT,comp> gauge_acc;
    U3ProjectStruct(gaugeAccessor<floatT,comp> gaugeAcc_in): gauge_acc(gaugeAcc_in){}
    __device__ __host__ GSU3<floatT> operator()(gSiteMu site) {
        GSU3<floatT> temp;
        temp = projectU3<floatT, HaloDepth>(gauge_acc, site, site.mu);
        return temp;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R14, CompressionType compLvl1 = R18, CompressionType compLvl2 = R18, CompressionType compNaik = U3R14>
class HisqSmearing {
private:
    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge_base;
    Gaugefield<floatT, onDevice, HaloDepth, compLvl1> _gauge_lvl1;
    Gaugefield<floatT, onDevice, HaloDepth, compLvl2> &_gauge_lvl2;
    Gaugefield<floatT, onDevice, HaloDepth, compNaik> &_gauge_naik;
    Gaugefield<floatT, onDevice, HaloDepth> _dummy;
    SmearingParameters<floatT> _Lvl1 = getLevel1Params<floatT>();
    SmearingParameters<floatT> _Lvl2;

    staple<floatT, HaloDepth, comp, 3> staple3_lvl1;
    staple<floatT, HaloDepth, comp, 5,1> staple5_1_lvl1;
    staple<floatT, HaloDepth, comp, 5,2> staple5_2_lvl1;
    staple<floatT, HaloDepth, comp, 5,3> staple5_3_lvl1;
    staple<floatT, HaloDepth, comp, 5,4> staple5_4_lvl1;
    staple<floatT, HaloDepth, comp, 7,1> staple7_1_lvl1;
    staple<floatT, HaloDepth, comp, 7,2> staple7_2_lvl1;
    staple<floatT, HaloDepth, comp, 7,3> staple7_3_lvl1;
    staple<floatT, HaloDepth, comp, 7,4> staple7_4_lvl1;
    staple<floatT, HaloDepth, comp, 7,5> staple7_5_lvl1;
    staple<floatT, HaloDepth, comp, 7,6> staple7_6_lvl1;
    staple<floatT, HaloDepth, comp, 7,7> staple7_7_lvl1;
    staple<floatT, HaloDepth, comp, 7,8> staple7_8_lvl1;

    staple<floatT, HaloDepth, compLvl1, 3> staple3_lvl2;
    staple<floatT, HaloDepth, compLvl1, 5,1> staple5_1_lvl2;
    staple<floatT, HaloDepth, compLvl1, 5,2> staple5_2_lvl2;
    staple<floatT, HaloDepth, compLvl1, 5,3> staple5_3_lvl2;
    staple<floatT, HaloDepth, compLvl1, 5,4> staple5_4_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,1> staple7_1_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,2> staple7_2_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,3> staple7_3_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,4> staple7_4_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,5> staple7_5_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,6> staple7_6_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,7> staple7_7_lvl2;
    staple<floatT, HaloDepth, compLvl1, 7,8> staple7_8_lvl2;

    staple<floatT, HaloDepth, compLvl1, -3> stapleNaik;
    staple<floatT, HaloDepth, compLvl1, -5> stapleLepage;

 public:
    HisqSmearing(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_base,Gaugefield<floatT, onDevice, HaloDepth, compLvl2> &gauge_lvl2,
		         Gaugefield<floatT, onDevice, HaloDepth, compNaik> &gauge_naik, floatT naik_epsilon = 0.0)
        : _gauge_base(gauge_base),                                   // This is the to-be-updated gauge field
          _gauge_lvl1(gauge_base.getComm(), "SHARED_GAUGENAIK" ),
          _gauge_lvl2(gauge_lvl2),                                   // In RHMC initializer this is "smearW"
          _gauge_naik(gauge_naik),                                   // In RHMC initializer this is "smearX"
          _dummy(gauge_base.getComm(), "SHARED_DUMMY"),
          _Lvl2(getLevel2Params<floatT>(naik_epsilon)),
          staple3_lvl1(_gauge_base.getAccessor()),           // Level 1 smearing
          staple5_1_lvl1(_gauge_base.getAccessor()),
          staple5_2_lvl1(_gauge_base.getAccessor()),
          staple5_3_lvl1(_gauge_base.getAccessor()),
          staple5_4_lvl1(_gauge_base.getAccessor()),
          staple7_1_lvl1(_gauge_base.getAccessor()),
          staple7_2_lvl1(_gauge_base.getAccessor()),
          staple7_3_lvl1(_gauge_base.getAccessor()),
          staple7_4_lvl1(_gauge_base.getAccessor()),
          staple7_5_lvl1(_gauge_base.getAccessor()),
          staple7_6_lvl1(_gauge_base.getAccessor()),
          staple7_7_lvl1(_gauge_base.getAccessor()),
          staple7_8_lvl1(_gauge_base.getAccessor()),
          staple3_lvl2(_gauge_lvl1.getAccessor()),           // Level 2 smearing
          staple5_1_lvl2(_gauge_lvl1.getAccessor()),
          staple5_2_lvl2(_gauge_lvl1.getAccessor()),
          staple5_3_lvl2(_gauge_lvl1.getAccessor()),
          staple5_4_lvl2(_gauge_lvl1.getAccessor()),
          staple7_1_lvl2(_gauge_lvl1.getAccessor()),
          staple7_2_lvl2(_gauge_lvl1.getAccessor()),
          staple7_3_lvl2(_gauge_lvl1.getAccessor()),
          staple7_4_lvl2(_gauge_lvl1.getAccessor()),
          staple7_5_lvl2(_gauge_lvl1.getAccessor()),
          staple7_6_lvl2(_gauge_lvl1.getAccessor()),
          staple7_7_lvl2(_gauge_lvl1.getAccessor()),
          staple7_8_lvl2(_gauge_lvl1.getAccessor()),
          stapleLepage(_gauge_lvl1.getAccessor()),
          stapleNaik(_gauge_lvl1.getAccessor()) {}

    void SmearAll(floatT mu_f=0.0, bool multiplyPhase = true);

    // Will be used in the force calculation
    template<CompressionType comp_tmp>
    void SmearLvl1(Gaugefield<floatT, onDevice, HaloDepth, comp_tmp> &gauge_out) {

        _dummy.iterateOverBulkAllMu(staple3_lvl1);
        gauge_out = _Lvl1._c_1 * _gauge_base + _Lvl1._c_3 * _dummy;

        _dummy.iterateOverBulkAllMu(staple5_1_lvl1);
        gauge_out = gauge_out + _Lvl1._c_5 * _dummy;

        _dummy.iterateOverBulkAllMu(staple5_2_lvl1);
        gauge_out = gauge_out + _Lvl1._c_5 * _dummy;

        _dummy.iterateOverBulkAllMu(staple5_3_lvl1);
        gauge_out = gauge_out + _Lvl1._c_5 * _dummy;

        _dummy.iterateOverBulkAllMu(staple5_4_lvl1);
        gauge_out = gauge_out + _Lvl1._c_5 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_1_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_2_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_3_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_4_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_5_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_6_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_7_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;

        _dummy.iterateOverBulkAllMu(staple7_8_lvl1);
        gauge_out = gauge_out + _Lvl1._c_7 * _dummy;
        gauge_out.updateAll();
    }

    // Will be used in the force calculation
    template<CompressionType comp_tmp, CompressionType comp_tmp2>
    void ProjectU3(Gaugefield<floatT, onDevice, HaloDepth, comp_tmp> &gauge_in,Gaugefield<floatT, onDevice, HaloDepth, comp_tmp2> &gauge_out) {
        gauge_out.iterateOverBulkAllMu(U3ProjectStruct<floatT, HaloDepth,comp_tmp>(gauge_in.getAccessor()));
        gauge_out.updateAll();
    }
};
