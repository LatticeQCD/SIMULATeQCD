#include "../../base/communication/siteComm.h"
#include "../../gauge/gaugefield.h"
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h" // has the C_1000

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
struct dDdmuFunctor {

    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    floatT _sign;
    floatT _pow_3;
    int _order;

    dDdmuFunctor(
            const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, int order, floatT naik_epsilon = 0.0) :
        _spinorIn(spinorIn.getAccessor()),
        _gAcc_smeared(gauge_smeared.getAccessor()),
        _gAcc_Naik(gauge_Naik.getAccessor())
    {
        _order = order;
        _sign = floatT(pow(-1.0, order));
        floatT c_3000 = (floatT)((-1./48.0)*(1.0+(double)naik_epsilon));
        _pow_3 = floatT(pow(3.0, order)) * c_3000;
    }

    auto getAccessor() const {
        return *this;
    }

    __device__ __host__ gVect3<floatT> operator()(gSiteStack site) const;
};
