//
// Created by Hai-Tao Shu on 11.11.2020
//
// see PolyakovLoopCorrelator.cpp for details
#include "WilsonLineCorrelator.h"
#include "../gaugeFixing/PolyakovLoopCorrelator.cpp"

template<class floatT, bool onDevice, size_t HaloDepth>
struct ResetWilsonLineKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor _wline;
    ResetWilsonLineKernel(Gaugefield<floatT,onDevice,HaloDepth> &_gauge,
                    MemoryAccessor wline) :
            gaugeAccessor(_gauge.getAccessor()), _wline(wline) {}
    __device__ __host__ void operator()(gSite site) {
        
        int id = site.isite;
        _wline.setValue<GSU3<floatT>>(id, gsu3_one<floatT>());

    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct MeasureWilsonLineKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor _wline;
    int _tau;
    int _dtau;
    MeasureWilsonLineKernel(Gaugefield<floatT,onDevice,HaloDepth> &_gauge,
                    MemoryAccessor wline, int tau, int dtau) :
            gaugeAccessor(_gauge.getAccessor()), _wline(wline), _tau(tau), _dtau(dtau) {}
    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        const int Nt = (int)GInd::getLatData().globLT;
        const int Nx = (int)GInd::getLatData().globLX;
        const int Ny = (int)GInd::getLatData().globLY;
        const int Nz = (int)GInd::getLatData().globLZ;

        sitexyzt coord = site.coord;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],(_tau+_dtau)%Nt);

        int id = site.isite;
        GSU3<floatT> temp;
        _wline.getValue<GSU3<floatT>>(id, temp);
        temp *= gaugeAccessor.getLink(GInd::getSiteMu(newSite, 3));
        _wline.setValue<GSU3<floatT>>(id,temp);
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
void WilsonLineCorrelator<floatT,onDevice,HaloDepth>::WLCtoArrays(std::vector<floatT> &vec_wlca_full,
                                                                  std::vector<floatT> &vec_wlc1_full,
                                                                  std::vector<floatT> &vec_wlc8_full,
                                                                  std::vector<int> &vec_factor,
                                                                  std::vector<int> &vec_weight,
                                                                  bool fastCorr) {

    typedef GIndexer<All,HaloDepth> GInd;
    int Nt=GInd::getLatData().globLT;
    int psite,qnorm,g;
    typedef gMemoryPtr<true>  MemTypeGPU;
    typedef gMemoryPtr<false> MemTypeCPU;
    ReadIndexSpatial<HaloDepth> calcReadIndexSpatial;

    for (int tau=0;tau<Nt;tau++)
    {
        rootLogger.info("measuring wilson line corr at tau: " ,  tau);
        MemTypeGPU PtrPloopGPU = MemoryManagement::getMemAt<true>("ploopGPU");
        PtrPloopGPU->template adjustSize<GSU3<floatT>>(GInd::getLatData().globvol3);
        MemoryAccessor _wlineGPU(PtrPloopGPU->getPointer());
    
        iterateFunctorNoReturn<onDevice>(ResetWilsonLineKernel<floatT,onDevice,HaloDepth>(_gauge, _wlineGPU), calcReadIndexSpatial, _elems);
        for (int dtau=0;dtau<Nt;dtau++)
        {
            std::vector<floatT> vec_wlca(this->distmax,0);
            std::vector<floatT> vec_wlc1(this->distmax,0);
            std::vector<floatT> vec_wlc8(this->distmax,0);
    
    
            MemTypeGPU PtrwlcaoffGPU = MemoryManagement::getMemAt<true>("wlcaoffGPU");
            MemTypeGPU Ptrwlc1offGPU = MemoryManagement::getMemAt<true>("wlc1offGPU");
            MemTypeGPU Ptrwlc8offGPU = MemoryManagement::getMemAt<true>("wlc8offGPU");
            MemTypeGPU PtrwlcaonGPU  = MemoryManagement::getMemAt<true>("wlcaonGPU");
            MemTypeGPU Ptrwlc1onGPU  = MemoryManagement::getMemAt<true>("wlc1onGPU");
            MemTypeGPU Ptrwlc8onGPU  = MemoryManagement::getMemAt<true>("wlc8onGPU");
            PtrwlcaoffGPU->template adjustSize<floatT>(this->pvol3);
            Ptrwlc1offGPU->template adjustSize<floatT>(this->pvol3);
            Ptrwlc8offGPU->template adjustSize<floatT>(this->pvol3);
            PtrwlcaonGPU->template  adjustSize<floatT>(this->RSonmax);
            Ptrwlc1onGPU->template  adjustSize<floatT>(this->RSonmax);
            Ptrwlc8onGPU->template  adjustSize<floatT>(this->RSonmax);
            MemoryAccessor _wlcaoffGPU (PtrwlcaoffGPU->getPointer());
            MemoryAccessor _wlc1offGPU (Ptrwlc1offGPU->getPointer());
            MemoryAccessor _wlc8offGPU (Ptrwlc8offGPU->getPointer());
            MemoryAccessor _wlcaonGPU  (PtrwlcaonGPU->getPointer());
            MemoryAccessor _wlc1onGPU  (Ptrwlc1onGPU->getPointer());
            MemoryAccessor _wlc8onGPU  (Ptrwlc8onGPU->getPointer());
    
            iterateFunctorNoReturn<onDevice>(MeasureWilsonLineKernel<floatT,onDevice,HaloDepth>(_gauge, _wlineGPU, tau, dtau), calcReadIndexSpatial, _elems);
    
            PassIndex passReadIndex;
    
            if (fastCorr) {
                iterateFunctorNoReturn<onDevice>(
                        PloopCorrOffAxisKernel<floatT,HaloDepth>(_wlineGPU,_wlcaoffGPU,_wlc1offGPU,_wlc8offGPU),passReadIndex,this->pvol3 );
                iterateFunctorNoReturn<onDevice>(
                        PloopCorrOnAxisKernel<floatT,HaloDepth> (_wlineGPU,_wlcaoffGPU,_wlc1offGPU,_wlc8offGPU,
                                                                 _wlcaonGPU,_wlc1onGPU,_wlc8onGPU ),            passReadIndex,this->RSonmax);
            } else {
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrAVG<floatT>>(_wlineGPU,_wlineGPU,_wlcaoffGPU),
                        passReadIndex,this->pvol3);
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrAVG<floatT>>(_wlineGPU,_wlineGPU,_wlcaoffGPU,_wlcaonGPU),
                        passReadIndex,this->RSonmax);
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrSIN<floatT>>(_wlineGPU,_wlineGPU,_wlc1offGPU),
                        passReadIndex,this->pvol3);
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrSIN<floatT>>(_wlineGPU,_wlineGPU,_wlc1offGPU,_wlc1onGPU),
                        passReadIndex,this->RSonmax);
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrOCT<floatT>>(_wlineGPU,_wlineGPU,_wlc8offGPU),
                        passReadIndex,this->pvol3);
                iterateFunctorNoReturn<onDevice>(
                        RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrOCT<floatT>>(_wlineGPU,_wlineGPU,_wlc8offGPU,_wlc8onGPU),
                        passReadIndex,this->RSonmax);
            }
    
            MemTypeCPU PtrwlcaoffCPU = MemoryManagement:: getMemAt<false>("wlcaoffCPU");
            PtrwlcaoffCPU->template adjustSize<floatT>(this->pvol3);
            PtrwlcaoffCPU->template copyFrom<true>(PtrwlcaoffGPU, PtrwlcaoffGPU->getSize());
            MemoryAccessor _wlcaoffCPU (PtrwlcaoffCPU->getPointer());
    
            MemTypeCPU Ptrwlc1offCPU = MemoryManagement::getMemAt<false>("wlc1offCPU");
            MemTypeCPU Ptrwlc8offCPU = MemoryManagement::getMemAt<false>("wlc8offCPU");
            MemTypeCPU PtrwlcaonCPU  = MemoryManagement::getMemAt<false>("wlcaonCPU");
            MemTypeCPU Ptrwlc1onCPU  = MemoryManagement::getMemAt<false>("wlc1onCPU");
            MemTypeCPU Ptrwlc8onCPU  = MemoryManagement::getMemAt<false>("wlc8onCPU");
            Ptrwlc1offCPU->template adjustSize<floatT>(this->pvol3);
            Ptrwlc8offCPU->template adjustSize<floatT>(this->pvol3);
            PtrwlcaonCPU->template  adjustSize<floatT>(this->RSonmax);
            Ptrwlc1onCPU->template  adjustSize<floatT>(this->RSonmax);
            Ptrwlc8onCPU->template  adjustSize<floatT>(this->RSonmax);
            Ptrwlc1offCPU->template copyFrom<true>(Ptrwlc1offGPU, Ptrwlc1offGPU->getSize());
            Ptrwlc8offCPU->template copyFrom<true>(Ptrwlc8offGPU, Ptrwlc8offGPU->getSize());
            PtrwlcaonCPU->template  copyFrom<true>(PtrwlcaonGPU , PtrwlcaonGPU->getSize());
            Ptrwlc1onCPU->template  copyFrom<true>(Ptrwlc1onGPU , Ptrwlc1onGPU->getSize());
            Ptrwlc8onCPU->template  copyFrom<true>(Ptrwlc8onGPU , Ptrwlc8onGPU->getSize());
            MemoryAccessor _wlc1offCPU (Ptrwlc1offCPU->getPointer());
            MemoryAccessor _wlc8offCPU (Ptrwlc8offCPU->getPointer());
            MemoryAccessor _wlcaonCPU  (PtrwlcaonCPU->getPointer());
            MemoryAccessor _wlc1onCPU  (Ptrwlc1onCPU->getPointer());
            MemoryAccessor _wlc8onCPU  (Ptrwlc8onCPU->getPointer());
    
            for (int dx=0 ; dx<(this->distmax) ; dx++) {
                vec_wlca[dx]   = 0.;
                vec_wlc1[dx]   = 0.;
                vec_wlc8[dx]   = 0.;
            }
            for (int dx=0 ; dx<(this->RSxmax) ; dx++)
            for (int dy=0 ; dy<(this->RSymax) ; dy++)
            for (int dz=0 ; dz<(this->RSzmax) ; dz++) {
                qnorm = dx*dx+dy*dy+dz*dz;
                if (qnorm>(this->distmax)) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
                psite               = dx + dy*(this->pvol1) + dz*(this->pvol2);
                g                   = vec_weight[psite];
                _wlcaoffCPU.getValue<floatT>(psite,wlcaoff);
                _wlc1offCPU.getValue<floatT>(psite,wlc1off);
                _wlc8offCPU.getValue<floatT>(psite,wlc8off);
                vec_wlca[qnorm]    += g*wlcaoff;
                vec_wlc1[qnorm]    += g*wlc1off;
                vec_wlc8[qnorm]    += g*wlc8off;
            }
            for (int dx=(this->RSxmax);dx<(this->RSonmax);dx++) {
                qnorm = dx*dx;
                if (qnorm>(this->distmax)) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
                g                   = 3;
                _wlcaonCPU.getValue<floatT>(dx,wlcaon);
                _wlc1onCPU.getValue<floatT>(dx,wlc1on);
                _wlc8onCPU.getValue<floatT>(dx,wlc8on);
                vec_wlca[qnorm]    += g*wlcaon;
                vec_wlc1[qnorm]    += g*wlc1on;
                vec_wlc8[qnorm]    += g*wlc8on;
            }
            for (int dx=0 ; dx<(this->distmax) ; dx++) {
                if (vec_factor[dx]>0) {
                    /// Apply the correct weights to the correlators.
                    vec_wlca[dx]=vec_wlca[dx]/((floatT)vec_factor[dx]);
                    vec_wlc1[dx]=vec_wlc1[dx]/((floatT)vec_factor[dx]);
                    vec_wlc8[dx]=vec_wlc8[dx]/((floatT)vec_factor[dx]);
                }
            }
    
            for (int dx=0 ; dx<(this->distmax) ; dx++) {
                vec_wlca_full[(this->distmax)*dtau+dx]+=vec_wlca[dx]/Nt;
                vec_wlc1_full[(this->distmax)*dtau+dx]+=vec_wlc1[dx]/Nt;
                vec_wlc8_full[(this->distmax)*dtau+dx]+=vec_wlc8[dx]/Nt;
            }
        }
    }
}

#define CLASS_INIT2(floatT, HALO) \
template class WilsonLineCorrelator<floatT,true,HALO>; \

INIT_PH(CLASS_INIT2)

