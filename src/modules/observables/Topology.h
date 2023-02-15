//
// Created by Lukas Mazur on 22.06.18.
//

#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "../../base/LatticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "FieldStrengthTensor.h"


/// Don't forget to exchange Halos before computing one of these observables!
template<class floatT, bool onDevice, size_t HaloDepth>
class Topology {
protected:
    LatticeContainer<onDevice,floatT> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;
private:

    gMemoryPtr<true> DevMemPointer;
    gMemoryPtr<false> HostMemPointer;

    bool recompute;

    typedef GIndexer<All,HaloDepth> GInd;
public:
    Topology(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield)
            : _redBase(gaugefield.getComm()),
              _gauge(gaugefield),
              DevMemPointer(MemoryManagement::getMemAt<true>("topDevMem")),
              HostMemPointer(MemoryManagement::getMemAt<false>("topHostMem")),
              recompute(true){
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    ~Topology() {}

    // That method will just fill _redBase with top. charge density.
    template<bool onDeviceRet, bool improved=false, bool improved_O6=false>
    MemoryAccessor topChargeField();

    void dontRecomputeField(){recompute = false;}
    void recomputeField(){recompute = true;}

    template<bool improved = false, bool improved_O6 = false>
    floatT topCharge();

    template<bool improved = false, bool improved_O6 = false>
    void topChargeTimeSlices(std::vector<floatT> &result);
};



/*topChargeDensKernel(gaugeAccessor<floatT> gAcc, floatT * topChargeDensArray)
	compute the topological charge density q(x) given by
		q(x)= 1/4pi^2 * tr(  F_(3,0) * F_(1,2)
					  	   + F_(3,1) * F_(2,0)
					  	   + F_(3,2) * F_(0,1) )
 */
template<class floatT, size_t HaloDepth, bool onDevice>
struct topChargeDens {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gAcc;
    FieldStrengthTensor<floatT,HaloDepth,onDevice,R18> FT;

    topChargeDens(gaugeAccessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define GSU3 matrices for the Fmunu computation
        GSU3<floatT> Fmunu;
        GSU3<floatT> Frhodelta;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the topological charge density
        Fmunu = FT(site, 3, 0);
        Frhodelta = FT(site, 1, 2);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 1);
        Frhodelta = FT(site, 2, 0);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 2);
        Frhodelta = FT(site, 0, 1);

        //multiply with 1/4*pi^2
        tmp += tr_d(Fmunu * Frhodelta);
        tmp /= (4. * M_PI * M_PI);

        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice>
struct topChargeDens_imp {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gAcc;
    FieldStrengthTensor_imp<floatT,HaloDepth,onDevice,R18> FT;

    topChargeDens_imp(gaugeAccessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define GSU3 matrices for the Fmunu computation
        GSU3<floatT> Fmunu;
        GSU3<floatT> Frhodelta;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the topological charge density
        Fmunu = FT(site, 3, 0);
        Frhodelta = FT(site, 1, 2);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 1);
        Frhodelta = FT(site, 2, 0);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 2);
        Frhodelta = FT(site, 0, 1);

        //multiply with 1/4*pi^2
        tmp += tr_d(Fmunu * Frhodelta);
        tmp /= (4. * M_PI * M_PI);

        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice>
struct topChargeDens_imp_imp {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gAcc;
    FieldStrengthTensor_imp_imp<floatT,HaloDepth,onDevice,R18> FT;

    topChargeDens_imp_imp(gaugeAccessor<floatT> gAcc) : gAcc(gAcc), FT(gAcc) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        // define GSU3 matrices for the Fmunu computation
        GSU3<floatT> Fmunu;
        GSU3<floatT> Frhodelta;

        // define a tmp complex number for the trace computation
        floatT tmp = 0.;

        //compute the topological charge density
        Fmunu = FT(site, 3, 0);
        Frhodelta = FT(site, 1, 2);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 1);
        Frhodelta = FT(site, 2, 0);
        tmp += tr_d(Fmunu * Frhodelta);

        Fmunu = FT(site, 3, 2);
        Frhodelta = FT(site, 0, 1);

        ////multiply with 1/4*pi^2
        //multiply with 1/32*pi^2
        tmp += tr_d(Fmunu * Frhodelta);
        tmp /= (4. * M_PI * M_PI);
        //tmp /= (32. * M_PI * M_PI);

        return tmp;
    }
};

template<class floatT, size_t HaloDepth, bool onDevice, bool improved, bool improved_O6>
struct topChargeDensKernel {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gAcc;

    topChargeDens<floatT,HaloDepth,onDevice> topChDens;
    topChargeDens_imp<floatT,HaloDepth,onDevice> topChDens_imp;
    topChargeDens_imp_imp<floatT,HaloDepth,onDevice> topChDens_imp_imp;
    typedef GIndexer<All, HaloDepth> GInd;

    topChargeDensKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge) : gAcc(gauge.getAccessor()),
    topChDens(gAcc), topChDens_imp(gAcc), topChDens_imp_imp(gAcc) {}

    __device__ __host__ inline floatT operator()(gSite site) {
        floatT result;
        if (improved_O6) {
            result = topChDens_imp_imp(site);
        } 
        else if (improved) {
            result = topChDens_imp(site);
        } 
        else {
            result = topChDens(site);
        }

        return result;
    }

};

#endif //TOPOLOGY_H
