/*
 * dslash.h
 *
 */

#pragma once

#include "../inverter/inverter.h"

#define C_1000 (0.5)

//! Abstract base class for all kinds of Dslash operators that shall enter the inversion
template<typename SpinorLHS_t, typename SpinorRHS_t>
class DSlash : public LinearOperator<SpinorRHS_t> {
public:

    //! This shall be a simple call of the DSlash without involving a constant
    virtual void Dslash(SpinorLHS_t &lhs, const SpinorRHS_t &rhs, bool update = true);

    //! This shall be a call of the M^\dagger M where M = m + D or similar
    virtual void applyMdaggM(SpinorRHS_t &, const SpinorRHS_t &, bool update = true) = 0;
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
            const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, floatT c_3000) :
            _spinorIn(spinorIn.getAccessor()),
            _gAcc_smeared(gauge_smeared.getAccessor()),
            _gAcc_Naik(gauge_Naik.getAccessor()), _c_3000(c_3000) {}

    __device__ __host__ inline auto operator()(gSiteStack site) const;

    __host__ __device__ void initialize(__attribute__((unused)) gSite site) {};

    auto getAccessor() const {
        return *this;
    }
};


template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
struct HisqDslashThreadRHSFunctor {

    gVect3arrayAcc<floatT> _spinorOut;
    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    floatT _c_3000;

    template<bool onDevice>
    HisqDslashThreadRHSFunctor(
        Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> &spinorOut,
        const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
        Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
        Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, floatT c_3000) :
        _spinorOut(spinorOut.getAccessor()), 
        _spinorIn(spinorIn.getAccessor()),
        _gAcc_smeared(gauge_smeared.getAccessor()),
        _gAcc_Naik(gauge_Naik.getAccessor()),
        _c_3000(c_3000) {}

    __device__ __host__ inline void operator()(gSite site);

    auto getAccessor() const {
        return *this;
    }  
};

template<bool onDevice, class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_cached>
struct HisqDslashStackedFunctor {

    gVect3arrayAcc<floatT> _spinorOut;
    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    
    

    floatT _c_3000;

    HisqDslashStackedFunctor(
        Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks*NStacks_cached> &spinorOut,
        const Spinorfield<floatT,onDevice, LatLayoutRHS, HaloDepthSpin, NStacks*NStacks_cached> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, floatT c_3000) :
            _spinorOut(spinorOut.getAccessor()),
            _spinorIn(spinorIn.getAccessor()),
            _gAcc_smeared(gauge_smeared.getAccessor()),
            _gAcc_Naik(gauge_Naik.getAccessor()), _c_3000(c_3000) {}

    __device__ __host__ inline void operator()(gSiteStack site);


    
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
            const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> &spinorTmp,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik,
            floatT mass2, floatT c_3000) :
            _spinorTmp(spinorTmp.getAccessor()),
            _spinorIn(spinorIn.getAccessor()),
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
    virtual void Dslash(SpinorLHS_t &lhs, const SpinorRHS_t &rhs, bool update = false);

    virtual void Dslash_threadRHS(SpinorLHS_t &lhs, const SpinorRHS_t &rhs, bool update = false);

    template<size_t NStacks_cached>
    void Dslash_stacked(Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks*NStacks_cached> &lhs, const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks*NStacks_cached>& rhs, bool update = false);

    //! Includes the mass term
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, const SpinorRHS_t &spinorIn, bool update = false);

    template<Layout LatLayout>
    HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>
    getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &rhs);


};

//! Dslash inverse

template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class HisqDSlashInverse {
private:
    // operators
    ConjugateGradient<floatT, NStacks> cg;
    HisqDSlash<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks> dslash_oe;
    HisqDSlash<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks> dslash_oe_inv;
    HisqDSlash<floatT, onDevice, Odd, HaloDepthGauge, HaloDepthSpin, NStacks> dslash_eo;
    floatT mass;

public:
    HisqDSlashInverse(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
                      Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik,
                      const double mass, const floatT naik_epsilon = 0.0) :
                      dslash_oe(gauge_smeared, gauge_Naik, 0.0, naik_epsilon),
                      dslash_oe_inv(gauge_smeared, gauge_Naik, mass, naik_epsilon),
                      dslash_eo(gauge_smeared, gauge_Naik, 0.0, naik_epsilon),
                      mass(mass) {}

    void apply_Dslash_inverse(SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks> &spinorOut,
                const SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks> &spinorIn,
                int cgMax, double residue) {
        // compute the inverse using
        // \chi_e = (1m^2 - D_{eo}D_{oe})^{-1} (m \eta_e - D_{eo} \eta_o)
        // \chi_o = \frac 1m (\eta_o - D_{oe}\chi_e)
        dslash_eo.Dslash(spinorOut.even, spinorIn.odd, true);
        spinorOut.even = spinorIn.even * mass - spinorOut.even;
        // invert in place is possible since the CG copies the input early on
        cg.invert_new(dslash_oe_inv, spinorOut.even, spinorOut.even, cgMax, residue); //! this takes up most of the computation time

        dslash_oe.Dslash(spinorOut.odd, spinorOut.even, false);
        spinorOut.odd = (static_cast<floatT>(1.) / mass)*(spinorIn.odd - spinorOut.odd);
        spinorOut.odd.updateAll();
    }
};


//! stdStagDlash

template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct stdStagDslashFunctor {

    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R14> _gAcc;

    template<bool onDevice, size_t NStacks>
    stdStagDslashFunctor(
            const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
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
    virtual void Dslash(SpinorLHS_t &lhs, const SpinorRHS_t &rhs, bool update = true);

    //! Includes the mass term
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, const SpinorRHS_t &spinorIn, bool update = true);

    template<Layout LatLayout>
    stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>
    getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &rhs);

};

