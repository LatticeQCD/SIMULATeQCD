#ifndef HYPSMEARING_H
#define HYPSMEARING_H


#include "../../gauge/gaugefield.h"
#include "../../gauge/constructs/fat7LinkConstructs.h"
#include "hypParameters.h"
#include "../../gauge/gaugefield_device.cpp"

template<class floatT, size_t HaloDepth, CompressionType comp, int linkNumber>
class HypStaple {
private:
  gaugeAccessor<floatT, comp> _gAcc_0;
  gaugeAccessor<floatT, comp> _gAcc_1;
  gaugeAccessor<floatT, comp> _gAcc_2;
  gaugeAccessor<floatT, comp> _gAcc_3;
  gaugeAccessor<floatT, comp> _gAcc_temp1;
  gaugeAccessor<floatT, comp> _gAcc_temp2;
  int _excluded_dir1;
  int _excluded_dir2;

public:
  HypStaple(gaugeAccessor<floatT, comp> gAccIn_0, //this is really the only required arg, but I'm not sure what the default is for gAcc
	    gaugeAccessor<floatT, comp> gAccIn_1,
	    gaugeAccessor<floatT, comp> gAccIn_2,
	    gaugeAccessor<floatT, comp> gAccIn_3,
	    int excluded_dir1 = -1,
	    int excluded_dir2 = -1 ) : _gAcc_0(gAccIn_0),_gAcc_1(gAccIn_1), _gAcc_2(gAccIn_2), _gAcc_3(gAccIn_3),
				       _gAcc_temp1(gAccIn_0), _gAcc_temp2(gAccIn_0),
				       _excluded_dir1(excluded_dir1), _excluded_dir2(excluded_dir2){}
  __host__ __device__ GSU3<floatT> operator() (gSiteMu site) {
    switch (linkNumber) {
    case 1:
      return hypThreeLinkStaple_third_level<floatT, HaloDepth, comp>(_gAcc_0, _gAcc_1, _gAcc_2, _gAcc_3, site, _gAcc_temp1, _gAcc_temp2);
    case 2:
      return hypThreeLinkStaple_second_level<floatT, HaloDepth, comp>(_gAcc_0, _gAcc_1, _gAcc_2, site, _excluded_dir1, _gAcc_temp1, _gAcc_temp2);
    case 3:
      return hypThreeLinkStaple_first_level<floatT, HaloDepth, comp>(_gAcc_0, site, _excluded_dir1, _excluded_dir2);
    case 4:
      return su3unitarize_project<floatT, HaloDepth, comp>(_gAcc_0, _gAcc_1, site);
    default:
      return gsu3_zero<floatT>();
    }
  }

};

//template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R14, CompressionType compLvl1 = R18, CompressionType compLvl2 = R18>
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R18>
class HypSmearing {
private:

  Gaugefield<floatT, onDevice, HaloDepth> _dummy;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_10;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_20;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_30;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_21;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_31;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl1_32;

  bool update_all = false;

  Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge_base;
  //Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge_lvl2_0;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl2_0;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl2_1;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl2_2;
  Gaugefield<floatT, onDevice, HaloDepth> _gauge_lvl2_3;


  HypSmearingParameters<floatT> params;

  HypStaple<floatT, HaloDepth, comp, 4> su3_unitarize;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_10;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_20;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_30;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_21;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_31;
  HypStaple<floatT, HaloDepth, comp, 3> staple3_lvl1_32;

public:
  //HypSmearing(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_base, Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out)
  HypSmearing(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_base)
    : _gauge_base(gauge_base),
      _dummy(gauge_base.getComm(), "SHARED_DUMMY"),
      _gauge_lvl1_10(gauge_base.getComm(), "SHARED_GAUGE_LVL1_10"),
      _gauge_lvl1_20(gauge_base.getComm(), "SHARED_GAUGE_LVL1_20"),
      _gauge_lvl1_30(gauge_base.getComm(), "SHARED_GAUGE_LVL1_30"),
      _gauge_lvl1_21(gauge_base.getComm(), "SHARED_GAUGE_LVL1_21"),
      _gauge_lvl1_31(gauge_base.getComm(), "SHARED_GAUGE_LVL1_31"),
      _gauge_lvl1_32(gauge_base.getComm(), "SHARED_GAUGE_LVL1_32"),
      //_gauge_lvl2_0(gauge_out), //return value
      _gauge_lvl2_0(gauge_base.getComm(), "SHARED_GAUGE_LVL2_0"), //return value
      _gauge_lvl2_1(gauge_base.getComm(), "SHARED_GAUGE_LVL2_1"),
      _gauge_lvl2_2(gauge_base.getComm(), "SHARED_GAUGE_LVL2_2"),
      _gauge_lvl2_3(gauge_base.getComm(), "SHARED_GAUGE_LVL2_3"),
      su3_unitarize(_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor()),
      staple3_lvl1_10(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 1, 0),
      staple3_lvl1_20(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 2, 0),
      staple3_lvl1_30(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 3, 0),
      staple3_lvl1_21(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 2, 1),
      staple3_lvl1_31(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 3, 1),
      staple3_lvl1_32(_gauge_base.getAccessor(),_gauge_base.getAccessor(), _gauge_base.getAccessor(), _gauge_base.getAccessor(), 3, 2){
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    update_all = size == 1 || HaloDepth;
  }

  void SmearAll(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out);
  void SmearTest(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out);
  void Su3Unitarize(Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out, Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_base);



};
#endif //HYPSMEARING_H
