#ifndef DSLASH_H
#define DSLASH_H

#include "../inverter/inverter.h"

#define C_1000 (0.5)

//! Abstract base class for all kind of Dslash operators that shall enter the inversion
template<typename SpinorLHS_t, typename SpinorRHS_t>
class DSlash : public LinearOperator<SpinorRHS_t> {
public:

    //! This shall be a simple call of the DSlash without involving a constant
    virtual void Dslash(SpinorLHS_t &lhs, SpinorRHS_t &rhs, bool update = true);

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    virtual void applyMdaggM(SpinorRHS_t &, SpinorRHS_t &, bool update = true) = 0;
};

//! HisqDslash

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct HisqDslashFunctor {

    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    floatT _c_3000;

    template<bool onDevice, size_t NStacks>
    HisqDslashFunctor(
            Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, floatT c_3000) :
            _spinorIn(spinorIn.getAccessor()),
            _gAcc_smeared(gauge_smeared.getAccessor()),
            _gAcc_Naik(gauge_Naik.getAccessor()), _c_3000(c_3000) {}

    __device__ __host__ inline auto operator()(gSiteStack site) const;

    auto getAccessor() const {
        return *this;
    }

};

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct HisqMdaggMFunctor {
    gVect3arrayAcc<floatT> _spinorTmp;
    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    floatT _mass2;
    floatT _c_3000;

    template<bool onDevice, size_t NStacks>
    HisqMdaggMFunctor(
            Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> &spinorTmp,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik,
            floatT mass2, floatT c_3000) :
            _spinorIn(spinorIn.getAccessor()),
            _spinorTmp(spinorTmp.getAccessor()),
            _gAcc_smeared(gauge_smeared.getAccessor()),
            _gAcc_Naik(gauge_Naik.getAccessor()),
            _mass2(mass2),
            _c_3000(c_3000) {}

    __device__ __host__ inline auto operator()(gSiteStack site);
};

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 1>
class HisqDSlash : public DSlash<Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>,
        Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> > {

    using SpinorRHS_t = Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks>;
    using SpinorLHS_t = Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>;

    template<CompressionType comp>
    using Gauge_t = Gaugefield<floatT, onDevice, HaloDepthGauge, comp>;

    Gauge_t<R18> &_gauge_smeared;
    Gauge_t<U3R14> &_gauge_Naik;


    //! Optimization: The memory of this spinor may be shared. However, it must not share the Halo Buffers
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> _tmpSpin;

    double _mass;
    floatT _mass2;
    floatT _c_3000;

public:
    HisqDSlash(Gauge_t<R18> &gaugefield_smeared, Gauge_t<U3R14> &gaugefield_Naik, const double mass, floatT naik_epsilon = 0.0,
               std::string spinorName = "SHARED_HisqDSlashSpinor") :
            _gauge_smeared(gaugefield_smeared), _gauge_Naik(gaugefield_Naik),
    _tmpSpin(_gauge_smeared.getComm(), spinorName), _mass(mass), _mass2(mass * mass), _c_3000((-1./48.0)*(1.0+(double)naik_epsilon)) {}

    //! Does not use the mass
    virtual void Dslash(SpinorLHS_t &lhs, SpinorRHS_t &rhs, bool update = false);

    //! Includes the mass term
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, SpinorRHS_t &spinorIn, bool update = false);

    template<Layout LatLayout>
    HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>
    getFunctor(Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &rhs);

};


//! stdStagDlash

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct stdStagDslashFunctor {

    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R14> _gAcc;

    template<bool onDevice, size_t NStacks>
    stdStagDslashFunctor(
            Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R14> &gauge) :
            _spinorIn(spinorIn.getAccessor()),
            _gAcc(gauge.getAccessor()) {}


    __device__ __host__ inline auto operator()(gSiteStack site) const;


    auto getAccessor() const {
        return *this;
    }

};

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 1>
class stdStagDSlash
        : public DSlash<Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>,
                Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> > {

    using SpinorRHS_t = Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks>;
    using SpinorLHS_t = Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>;

    template<CompressionType comp>
    using Gauge_t = Gaugefield<floatT, onDevice, HaloDepthGauge, comp>;

    Gauge_t<R14> &_gauge;

    //! Optimization: The memory of this spinor may be shared. However, it must not share the Halo Buffers
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> _tmpSpin;
    floatT _mass;
    floatT _mass2;

public:

    stdStagDSlash(Gauge_t<R14> &gaugefield, const floatT mass) :
            _gauge(gaugefield), _tmpSpin(_gauge.getComm()), _mass(mass), _mass2(mass * mass) {

    }

    //! Does not use the mass
    virtual void Dslash(SpinorLHS_t &lhs, SpinorRHS_t &rhs, bool update = true);

    //! Includes the mass term
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, SpinorRHS_t &spinorIn, bool update = true);

    template<Layout LatLayout>
    stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>
    getFunctor(Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &rhs);

};

#endif
