/*
 * smearParameters.h
 *
 * J. Goswami
 *
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "../../define.h"
template <class floatT>
struct SmearingParameters{
    SmearingParameters(){}
    floatT _c_1;               // 1 link
    floatT _c_3;               // 3 link
    floatT _c_5;               // 5 link
    floatT _c_7;               // 7 link
    floatT _c_lp;              // 5 link Lepage
    SmearingParameters(floatT c_1, floatT c_3, floatT c_5, floatT c_7, floatT c_lp) :
        _c_1(c_1),
        _c_3(c_3),
        _c_5(c_5),
        _c_7(c_7),
        _c_lp(c_lp){}
};

template<class floatT>
inline SmearingParameters<floatT> getLevel1Params(){
    SmearingParameters<floatT> params_L1(1/8.,1/8./2.,1/8./8.,1/48./8.,0.0);
    return params_L1;
}

template<class floatT>
inline SmearingParameters<floatT> getLevel2Params(floatT naik_epsilon = 0.0){
    SmearingParameters<floatT> params_L2(1.0*(1+naik_epsilon/8),1/8./2.,1/8./8.,1/48./8.,-1/8.0);
    return params_L2;
}

template<class floatT>
floatT get_naik_epsilon_from_amc(floatT amc) {
    floatT amc_sqr = amc * amc;
    return amc_sqr * (-27./40. + amc_sqr * (327./1120. + amc_sqr * (-15607./268800. - amc_sqr * 73697./3942400.)));
}

#endif //PARAMETERS_H
