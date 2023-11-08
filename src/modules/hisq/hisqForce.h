/*
 * hisqForce.h
 *
 * D. Bollweg
 *
 * The methods related to the fictitious HISQ force, which drives the RHMC trajectory.
 *
 */

#include "../../gauge/constructs/hisqForceConstructs.h"
#include "../../gauge/constructs/derivativeProjectU3.h"
#include "../../gauge/constructs/naikDerivativeConstructs.h"
#include "hisqSmearing.h"
#include "../inverter/inverter.h"
#include "../dslash/dslash.h"
#include "../rhmc/rhmcParameters.h"

template <bool RunTesting> //template selector to change between rat approx used in RHMC and rat approx used in Testing
class RatDegreeSelector;

template <> class RatDegreeSelector<true> { public: static const int RatDegree = 14;};
template <> class RatDegreeSelector<false> { public: static const int RatDegree = 12;};

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting = false, const int rdeg = RatDegreeSelector<runTesting>::RatDegree>
struct tensor_product {
    Vect3arrayAcc<floatT> _x;
    Vect3arrayAcc<floatT> _y;
    SU3Accessor<floatT> gAccessor;
    SimpleArray<floatT,rdeg> _rat_num;
    tensor_product(Gaugefield<floatT, onDevice, HaloDepth> &gaugeIn, Vect3arrayAcc<floatT> x, Vect3arrayAcc<floatT> y, SimpleArray<floatT,rdeg> rat_num);
    __device__ __host__ SU3<floatT> operator()(gSiteMu site);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
class contribution_3link {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    SmearingParameters<floatT> _smParams = (lvl1 ? getLevel1Params<floatT>() : getLevel2Params<floatT>());
public:
    contribution_3link(Gaugefield<floatT, onDevice, HaloDepth,comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
class contribution_lepagelink {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c_lp = -1/8.0;
public:
    contribution_lepagelink(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
class contribution_5link {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c5=1/8./8.;
public:
    contribution_5link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
class contribution_7link {
private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c7 = 1/48./8.;
public:
    contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceInm);
    __host__ __device__ SU3<floatT> operator() (gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
class constructU3ProjForce {
private:
    SU3Accessor<floatT> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
public:
    constructU3ProjForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
class finalizeForce {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
public:
    finalizeForce(Gaugefield<floatT, onDevice, HaloDepth,comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
class constructNaikDerivForce {
private:
    SU3Accessor<floatT> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
public:
    constructNaikDerivForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp = R18, bool runTesting = false, const int rdeg = RatDegreeSelector<runTesting>::RatDegree>
class HisqForce {
private:

    Gaugefield<floatT, onDevice, HaloDepth> _GaugeU3P;
    Gaugefield<floatT, onDevice, HaloDepth> _GaugeLvl1;
    Gaugefield<floatT, onDevice, HaloDepth, comp> _TmpForce;     // One of its uses is for NaikForce
    Gaugefield<floatT, onDevice, HaloDepth, R18> &_GaugeBase;
    Gaugefield<floatT, onDevice,  HaloDepth, R18> _Dummy;
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, rdeg> _spinor_x;
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, rdeg> _spinor_y;


    constructU3ProjForce<floatT, onDevice, HaloDepth,comp> _createF2;

    finalizeForce<floatT, onDevice, HaloDepth,comp> _finalizeF3;

    constructNaikDerivForce<floatT, onDevice, HaloDepth,comp> _createNaikF1;

    //F1 part
    contribution_3link<floatT, onDevice, HaloDepth, comp, false> F1_create_3Link;

    // contribution_7link<floatT, onDevice, HaloDepth, comp, 1> F1_7link_part_1;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 2> F1_7link_part_2;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 3> F1_7link_part_3;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 4> F1_7link_part_4;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 5> F1_7link_part_5;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 6> F1_7link_part_6;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 7> F1_7link_part_7;

    contribution_5link<floatT, onDevice, HaloDepth, comp, 11> F1_5link_part11;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 12> F1_5link_part12;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 13> F1_5link_part13;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 14> F1_5link_part14;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 20> F1_5link_part20;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 30> F1_5link_part30;

    contribution_lepagelink<floatT, onDevice, HaloDepth, comp> F1_lepagelink;


    //F3 part
    contribution_3link<floatT, onDevice, HaloDepth, comp, true> F3_create_3Link;

    // contribution_7link<floatT, onDevice, HaloDepth, comp, 1> F3_7link_part_1;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 2> F3_7link_part_2;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 3> F3_7link_part_3;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 4> F3_7link_part_4;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 5> F3_7link_part_5;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 6> F3_7link_part_6;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 7> F3_7link_part_7;

    contribution_5link<floatT, onDevice, HaloDepth, comp, 11> F3_5link_part11;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 12> F3_5link_part12;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 13> F3_5link_part13;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 14> F3_5link_part14;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 20> F3_5link_part20;
    contribution_5link<floatT, onDevice, HaloDepth, comp, 30> F3_5link_part30;

    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &_smearing;
    AdvancedMultiShiftCG<floatT, rdeg> &_cg;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &_dslash;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &_dslash_multi;
    RhmcParameters _rhmc_param;
    RationalCoeff _rat;


public:
    // Initializer list is in cpp file.
    HisqForce(Gaugefield<floatT, onDevice, HaloDepth, R18> &GaugeBase,
   	    Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
   	    AdvancedMultiShiftCG<floatT, rdeg> &cg,
   	    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin,1> &dslash,
   	    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &dslash_multi,
   	    RhmcParameters &rhmc_param,
   	    RationalCoeff &rat,
   	    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &smearing);

    void make_f0(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn,
        Gaugefield<floatT,onDevice, HaloDepth, comp> &Force,
        Gaugefield<floatT,onDevice, HaloDepth, comp> &NaikForce,
        bool isLight);

    void updateForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force, bool isLight);

    void TestForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force, grnd_state<true> &d_rand);
};
