//
// Created by Jangho Kim on 2023.04.10.
//

#ifndef WEINBERG_H
#define WEINBERG_H

#include "../../base/latticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "fieldStrengthTensor.h"


/// Don't forget to exchange Halos before computing one of these observables!
template<class floatT, bool onDevice, size_t HaloDepth>
class Weinberg {
protected:
    LatticeContainer<onDevice,floatT> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;
private:

    gMemoryPtr<true> DevMemPointer;
    gMemoryPtr<false> HostMemPointer;

    bool recompute;

    typedef GIndexer<All,HaloDepth> GInd;
public:
    Weinberg(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield)
            : _redBase(gaugefield.getComm()),
              _gauge(gaugefield),
              DevMemPointer(MemoryManagement::getMemAt<true>("wbDevMem")),
              HostMemPointer(MemoryManagement::getMemAt<false>("wbHostMem")),
              recompute(true){
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    ~Weinberg() {}

    // That method will just fill _redBase with weinberg operator.
    template<bool onDeviceRet, bool improved=false, bool improved_O6=false>
    MemoryAccessor WBField();

    void dontRecomputeField(){recompute = false;}
    void recomputeField(){recompute = true;}

    template<bool improved = false, bool improved_O6 = false>
    floatT WB();

    template<bool improved = false, bool improved_O6 = false>
    void WBTimeSlices(std::vector<floatT> &result);
};



/*WBDensKernel(SU3Accessor<floatT> gAcc, floatT * WBDensArray)
	compute the Weinberg Operator Wb(x) given by
		WB(x)= 8/3 \sum_{x} Re Tr I [ \sum_{rho} ( F_(0,rho) * F_(1,rho) - F_(1,rho) * F_(0,rho)] * [F_(2,rho) * F_(3,rho) - F_(3,rho) * F_(2,rho) ) 
    +  \sum_{rho} ( F_(0,rho) * F_(2,rho) - F_(2,rho) * F_(0,rho)] * [F_(3,rho) * F_(1,rho) - F_(1,rho) * F_(3,rho) ) 
    +  \sum_{rho} ( F_(0,rho) * F_(3,rho) - F_(3,rho) * F_(0,rho)] * [F_(1,rho) * F_(2,rho) - F_(2,rho) * F_(1,rho) ) ]
 */
template<class floatT, size_t HaloDepth, bool onDevice>
struct WBDens {
    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> gAcc;
    FieldStrengthTensor<floatT,HaloDepth,onDevice,R18> FT;

    WBDens(SU3Accessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define SU3 matrices for the Fmunu computation
        SU3<floatT> Fmu1rho;
        SU3<floatT> Fnu1rho;
        SU3<floatT> Fmunu;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the Weinberg Operator
        int mu1=0;
        for(int nu1=1;nu1<4;++nu1){
            for(int rho=0; rho < 4; rho++){
                if((rho != mu1) && (rho != nu1)){
                    Fmu1rho = FT(site, mu1, rho);
                    Fnu1rho = FT(site, nu1, rho);
                    int mu2=nu1%3; mu2++;
                    int nu2=mu2%3; nu2++;
                    Fmunu = FT(site, mu2, nu2);
                    tmp += tr_d((COMPLEX(floatT)(0, 1)) * ((floatT) 8) / ((floatT) 3) * (Fmu1rho * Fnu1rho - Fnu1rho*Fmu1rho) * Fmunu);
                }
            }
        }
        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice>
struct WBDens_imp {
    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> gAcc;
    FieldStrengthTensor_imp<floatT,HaloDepth,onDevice,R18> FT;

    WBDens_imp(SU3Accessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define SU3 matrices for the Fmunu computation
        SU3<floatT> Fmu1rho;
        SU3<floatT> Fnu1rho;
        SU3<floatT> Fmunu;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the Weinberg Operator
        int mu1=0;
        for(int nu1=1;nu1<4;++nu1){
            for(int rho=0; rho < 4; rho++){
                if((rho != mu1) && (rho != nu1)){
                    Fmu1rho = FT(site, mu1, rho);
                    Fnu1rho = FT(site, nu1, rho);
                    int mu2=nu1%3; mu2++;
                    int nu2=mu2%3; nu2++;
                    Fmunu = FT(site, mu2, nu2);
                    tmp += tr_d((COMPLEX(floatT)(0, 1)) * ((floatT) 8) / ((floatT) 3) * (Fmu1rho * Fnu1rho - Fnu1rho*Fmu1rho) * Fmunu);
                }
            }
        }
        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice>
struct WBDens_imp_imp {
    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> gAcc;
    FieldStrengthTensor_imp_imp<floatT,HaloDepth,onDevice,R18> FT;

    WBDens_imp_imp(SU3Accessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define SU3 matrices for the Fmunu computation
        SU3<floatT> Fmu1rho;
        SU3<floatT> Fnu1rho;
        SU3<floatT> Fmunu;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the Weinberg Operator
        int mu1=0;
        for(int nu1=1;nu1<4;++nu1){
            for(int rho=0; rho < 4; rho++){
                if((rho != mu1) && (rho != nu1)){
                    Fmu1rho = FT(site, mu1, rho);
                    Fnu1rho = FT(site, nu1, rho);
                    int mu2=nu1%3; mu2++;
                    int nu2=mu2%3; nu2++;
                    Fmunu = FT(site, mu2, nu2);
                    tmp += tr_d((COMPLEX(floatT)(0, 1)) * ((floatT) 8) / ((floatT) 3) * (Fmu1rho * Fnu1rho - Fnu1rho*Fmu1rho) * Fmunu);
                }
            }
        }
        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, bool improved, bool improved_O6>
struct WBDensKernel {
    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> gAcc;

    WBDens<floatT,HaloDepth,onDevice> WbDens;
    WBDens_imp<floatT,HaloDepth,onDevice> WbDens_imp;
    WBDens_imp_imp<floatT,HaloDepth,onDevice> WbDens_imp_imp;
    typedef GIndexer<All, HaloDepth> GInd;

    WBDensKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge) : gAcc(gauge.getAccessor()),
    WbDens(gAcc), WbDens_imp(gAcc), WbDens_imp_imp(gAcc) {}

    __device__ __host__ inline floatT operator()(gSite site) {
        floatT result;
        if (improved_O6) {
            result = WbDens_imp_imp(site);
        } 
        else if (improved) {
            result = WbDens_imp(site);
        } 
        else {
            result = WbDens(site);
        }

        return result;
    }

};

#endif //TOPOLOGY_H
