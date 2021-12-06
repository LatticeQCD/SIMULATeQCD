/* 
 * PolyakovLoopCorrelator.cu
 * 
 * D. Clarke, 27 Apr 2020
 *
 * This includes some functions for computing Polyakov loop correlations. There are more general implementations of
 * these correlations in math/correlators.h and math/correlators.cu, but they are slower. The reason they are slower
 * is that they repeat kernel calls for each kind of Polyakov loop correlation there is, which means it repeats the
 * same code three times instead of just calling it once, leading to about a 3x slow-down. Please have a look at that
 * header file and cu file if you would like a better understanding of what these kernels are doing. The code is very
 * similar, and I didn't want to repeat comments.
 *
 */

#include "PolyakovLoopCorrelator.h"

/// Kernel to compute off-axis Polyakov loop correlations. For use with single GPU.
/// INTENT: IN--ploop; OUT--plcaoff, plc1off, plc8off
template<class floatT, size_t HaloDepth>
struct PloopCorrOffAxisKernel : CorrelatorTools<floatT, true, HaloDepth> {
    MemoryAccessor _ploop;  /// untraced Polyakov loop array, indexed by spatial site
    MemoryAccessor _plca;   /// average Polyakov loop array, indexed by displacement
    MemoryAccessor _plc1;   /// singlet
    MemoryAccessor _plc8;   /// octet

    PloopCorrOffAxisKernel(MemoryAccessor ploop, MemoryAccessor plca, MemoryAccessor plc1, MemoryAccessor plc8) :
            _ploop(ploop), _plca(plca), _plc1(plc1), _plc8(plc8) {}

    __device__ __host__ void operator()(size_t dindex) {
        typedef GIndexer<All,HaloDepth> GInd;
        size_t       m,n1,n2,n3,n4;
        int          dx,dy,dz,rem;
        floatT       avg,sin,oct;
        GSU3<floatT> polm,poln;

        divmod((int)dindex,(int)(this->pvol2),dz,rem);
        divmod(rem        ,(int)(this->pvol1),dy,dx );

        avg=0.0;
        sin=0.0;
        oct=0.0;
        for(int tx=0;tx<(this->Nx);tx++)
        for(int ty=0;ty<(this->Ny);ty++)
        for(int tz=0;tz<(this->Nz);tz++) {
            m  = GInd::getSiteSpatial(            tx               ,            ty               , tz               ,0).isite;
            n1 = GInd::getSiteSpatial((           tx+dx)%(this->Nx),(           ty+dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n2 = GInd::getSiteSpatial(((this->Nx)+tx-dx)%(this->Nx),(           ty+dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n3 = GInd::getSiteSpatial((           tx+dx)%(this->Nx),((this->Ny)+ty-dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;
            n4 = GInd::getSiteSpatial(((this->Nx)+tx-dx)%(this->Nx),((this->Ny)+ty-dy)%(this->Ny),(tz+dz)%(this->Nz),0).isite;

            _ploop.getValue<GSU3<floatT>>(m ,polm);
            _ploop.getValue<GSU3<floatT>>(n1,poln);
            plc_contrib(polm,poln,avg,sin,oct);

            _ploop.getValue<GSU3<floatT>>(n2,poln);
            plc_contrib(polm,poln,avg,sin,oct);

            _ploop.getValue<GSU3<floatT>>(n3,poln);
            plc_contrib(polm,poln,avg,sin,oct);

            _ploop.getValue<GSU3<floatT>>(n4,poln);
            plc_contrib(polm,poln,avg,sin,oct);
        }
        avg /= ((this->vol3)*36.);
        sin /= ((this->vol3)*12.);
        oct /= ((this->vol3)*4.);
        _plca.setValue<floatT>(dindex,avg);
        _plc1.setValue<floatT>(dindex,sin);
        _plc8.setValue<floatT>(dindex,oct);
    }
};

/// Kernel to compute on-axis Polyakov loop correlations. For use with single GPU.
/// INTENT: IN--pol, plcaoff, plc1off, plc8off; OUT--plcaon, plc1on, plc8on
template<class floatT, size_t HaloDepth>
struct PloopCorrOnAxisKernel : CorrelatorTools<floatT, true, HaloDepth> {
    MemoryAccessor _ploop;
    MemoryAccessor _plcaoff;  /// average Polyakov loop array, off-axis
    MemoryAccessor _plc1off;
    MemoryAccessor _plc8off;
    MemoryAccessor _plcaon;   /// average Polyakov loop array, on-axis
    MemoryAccessor _plc1on;
    MemoryAccessor _plc8on;
    PloopCorrOnAxisKernel(MemoryAccessor ploop, MemoryAccessor plcaoff, MemoryAccessor plc1off, MemoryAccessor plc8off,
                          MemoryAccessor plcaon, MemoryAccessor plc1on, MemoryAccessor plc8on) :
            _ploop(ploop), _plcaoff(plcaoff), _plc1off(plc1off), _plc8off(plc8off),
            _plcaon(plcaon), _plc1on(plc1on), _plc8on(plc8on) {}

    __device__ __host__ void operator()(size_t dx){
        typedef GIndexer<All,HaloDepth> GInd;
        size_t       m,n1,n2,n3;
        GSU3<floatT> polm,poln;
        floatT       avg,sin,oct,plcx,plcy,plcz;

        if(dx<(this->RSxmax)) {
            /// This is the part where we grab the on-axis calculations that were already done in the above kernel.
            _plcaoff.getValue<floatT>(dx      ,plcx);
            _plcaoff.getValue<floatT>(dx*(this->pvol1),plcy);
            _plcaoff.getValue<floatT>(dx*(this->pvol2),plcz);
            avg = (plcx+plcy+plcz)/3.;
            _plc1off.getValue<floatT>(dx      ,plcx);
            _plc1off.getValue<floatT>(dx*(this->pvol1),plcy);
            _plc1off.getValue<floatT>(dx*(this->pvol2),plcz);
            sin = (plcx+plcy+plcz)/3.;
            _plc8off.getValue<floatT>(dx      ,plcx);
            _plc8off.getValue<floatT>(dx*(this->pvol1),plcy);
            _plc8off.getValue<floatT>(dx*(this->pvol2),plcz);
            oct = (plcx+plcy+plcz)/3.;
            _plcaon.setValue<floatT>(dx,avg);
            _plc1on.setValue<floatT>(dx,sin);
            _plc8on.setValue<floatT>(dx,oct);

        } else {
            /// And these are the on-axis correlators that haven't been computed yet.
            avg = 0.;
            sin = 0.;
            oct = 0.;
            for(int tx=0;tx<(this->Nx);tx++)
            for(int ty=0;ty<(this->Ny);ty++)
            for(int tz=0;tz<(this->Nz);tz++) {
                m =GInd::getSiteSpatial( tx               , ty               , tz               ,0).isite;
                n1=GInd::getSiteSpatial((tx+dx)%(this->Nx), ty               , tz               ,0).isite;
                n2=GInd::getSiteSpatial( tx               ,(ty+dx)%(this->Ny), tz               ,0).isite;
                n3=GInd::getSiteSpatial( tx               , ty               ,(tz+dx)%(this->Nz),0).isite;

                _ploop.getValue<GSU3<floatT>>(m ,polm);
                _ploop.getValue<GSU3<floatT>>(n1,poln);
                plc_contrib(polm,poln,avg,sin,oct);
                _ploop.getValue<GSU3<floatT>>(n2,poln);
                plc_contrib(polm,poln,avg,sin,oct);
                _ploop.getValue<GSU3<floatT>>(n3,poln);
                plc_contrib(polm,poln,avg,sin,oct);
            }
            avg /= ((this->vol3)*27.);
            sin /= ((this->vol3)*9.);
            oct /= ((this->vol3)*3.);
            _plcaon.setValue<floatT>(dx,avg);
            _plc1on.setValue<floatT>(dx,sin);
            _plc8on.setValue<floatT>(dx,oct);
        }
    }
};


/// Calculate on-axis and off-axis Polyakov loop correlations for each channel and store them accordingly. The
/// final results for each channel are stored in the arrays vec_plc%. The array vec_factor counts the number of
/// contributions at each distance. It is used in this function to normalize the correlators, but it is useful
/// outside of this function to check whether any correlators exist at a given distance.
template<class floatT, bool onDevice, size_t HaloDepth>
void PolyakovLoopCorrelator<floatT,onDevice,HaloDepth>::PLCtoArrays(std::vector<floatT> &vec_plca,
                                                                    std::vector<floatT> &vec_plc1,
                                                                    std::vector<floatT> &vec_plc8,
                                                                    std::vector<int> &vec_factor,
                                                                    std::vector<int> &vec_weight,
                                                                    bool fastCorr) {

    typedef GIndexer<All,HaloDepth> GInd;

    int psite,qnorm,g;

    PolyakovLoop<floatT,onDevice,HaloDepth> PLoop(_gauge);

    /// These next three lines create a memory accessor for the untraced Polyakov loop array. The getMemAt template
    /// parameter must be true for GPU and false for CPU.
    gMemoryPtr<true> PtrPloopGPU = MemoryManagement::getMemAt<true>("ploopGPU");
    /// This line tells us PtrPloopGPU refers to an array of GSU3<floatT> with a size of global spacelike volume.
    PtrPloopGPU->template adjustSize<GSU3<floatT>>(this->vol3);
    /// Finally we're ready to name the memory accessor. I'll call it _ploopGPU. In general I will try to append each
    /// accessor with GPU or CPU to remind ourselves where it can be used.
    MemoryAccessor _ploopGPU(PtrPloopGPU->getPointer());

    /// The correlations are calculated in two steps, an on-axis step and off-axis step. There are different numbers of
    /// on-axis and off-axis points, so each type of correlation requires two arrays.
    gMemoryPtr<true> PtrplcaoffGPU = MemoryManagement::getMemAt<true>("plcaoffGPU");
    gMemoryPtr<true> Ptrplc1offGPU = MemoryManagement::getMemAt<true>("plc1offGPU");
    gMemoryPtr<true> Ptrplc8offGPU = MemoryManagement::getMemAt<true>("plc8offGPU");
    gMemoryPtr<true> PtrplcaonGPU  = MemoryManagement::getMemAt<true>("plcaonGPU");
    gMemoryPtr<true> Ptrplc1onGPU  = MemoryManagement::getMemAt<true>("plc1onGPU");
    gMemoryPtr<true> Ptrplc8onGPU  = MemoryManagement::getMemAt<true>("plc8onGPU");
    PtrplcaoffGPU->template adjustSize<floatT>(this->pvol3);
    Ptrplc1offGPU->template adjustSize<floatT>(this->pvol3);
    Ptrplc8offGPU->template adjustSize<floatT>(this->pvol3);
    PtrplcaonGPU->template  adjustSize<floatT>(this->RSonmax);
    Ptrplc1onGPU->template  adjustSize<floatT>(this->RSonmax);
    Ptrplc8onGPU->template  adjustSize<floatT>(this->RSonmax);
    MemoryAccessor _plcaoffGPU (PtrplcaoffGPU->getPointer());
    MemoryAccessor _plc1offGPU (Ptrplc1offGPU->getPointer());
    MemoryAccessor _plc8offGPU (Ptrplc8offGPU->getPointer());
    MemoryAccessor _plcaonGPU  (PtrplcaonGPU->getPointer());
    MemoryAccessor _plc1onGPU  (Ptrplc1onGPU->getPointer());
    MemoryAccessor _plc8onGPU  (Ptrplc8onGPU->getPointer());

    /// First populate the untraced Polyakov loop array so it can be used for later calculations.
    PLoop.PloopInArray(_ploopGPU);

    PassIndex passReadIndex;

    if (fastCorr) {
        /// Calculate off-axis correlations, then...
        iterateFunctorNoReturn<onDevice>(
                PloopCorrOffAxisKernel<floatT,HaloDepth>(_ploopGPU,_plcaoffGPU,_plc1offGPU,_plc8offGPU),passReadIndex,this->pvol3 );
        /// ... calculate on-axis correlations. Must be done in this order!
        iterateFunctorNoReturn<onDevice>(
                PloopCorrOnAxisKernel<floatT,HaloDepth> (_ploopGPU,_plcaoffGPU,_plc1offGPU,_plc8offGPU,
                                                         _plcaonGPU,_plc1onGPU,_plc8onGPU ),            passReadIndex,this->RSonmax);
    } else {
        iterateFunctorNoReturn<onDevice>(
                RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrAVG<floatT>>(_ploopGPU,_ploopGPU,_plcaoffGPU),
                passReadIndex,this->pvol3);
        iterateFunctorNoReturn<onDevice>(
                RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrAVG<floatT>>(_ploopGPU,_ploopGPU,_plcaoffGPU,_plcaonGPU),
                passReadIndex,this->RSonmax);
        iterateFunctorNoReturn<onDevice>(
                RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrSIN<floatT>>(_ploopGPU,_ploopGPU,_plc1offGPU),
                passReadIndex,this->pvol3);
        iterateFunctorNoReturn<onDevice>(
                RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrSIN<floatT>>(_ploopGPU,_ploopGPU,_plc1offGPU,_plc1onGPU),
                passReadIndex,this->RSonmax);
        iterateFunctorNoReturn<onDevice>(
                RestrictedOffAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrOCT<floatT>>(_ploopGPU,_ploopGPU,_plc8offGPU),
                passReadIndex,this->pvol3);
        iterateFunctorNoReturn<onDevice>(
                RestrictedOnAxisKernel<floatT,HaloDepth,GSU3<floatT>,floatT,polCorrOCT<floatT>>(_ploopGPU,_ploopGPU,_plc8offGPU,_plc8onGPU),
                passReadIndex,this->RSonmax);
    }

    /// Now we create a memory accessor for the off-axis average polyakov loop correlation that can be read on the CPU.
    gMemoryPtr<false> PtrplcaoffCPU = MemoryManagement:: getMemAt<false>("plcaoffCPU");
    PtrplcaoffCPU->template adjustSize<floatT>(this->pvol3);
    /// This time, we have to copy data from the GPU array to the CPU array. The template parameter for copyFrom should
    /// match the array from which we copy.
    PtrplcaoffCPU->template copyFrom<true>(PtrplcaoffGPU, PtrplcaoffGPU->getSize());
    MemoryAccessor _plcaoffCPU (PtrplcaoffCPU->getPointer());

    /// Similarly create CPU accessors for the remaining five correlation arrays and copy data from the GPU.
    gMemoryPtr<false> Ptrplc1offCPU = MemoryManagement::getMemAt<false>("plc1offCPU");
    gMemoryPtr<false> Ptrplc8offCPU = MemoryManagement::getMemAt<false>("plc8offCPU");
    gMemoryPtr<false> PtrplcaonCPU  = MemoryManagement::getMemAt<false>("plcaonCPU");
    gMemoryPtr<false> Ptrplc1onCPU  = MemoryManagement::getMemAt<false>("plc1onCPU");
    gMemoryPtr<false> Ptrplc8onCPU  = MemoryManagement::getMemAt<false>("plc8onCPU");
    Ptrplc1offCPU->template adjustSize<floatT>(this->pvol3);
    Ptrplc8offCPU->template adjustSize<floatT>(this->pvol3);
    PtrplcaonCPU->template  adjustSize<floatT>(this->RSonmax);
    Ptrplc1onCPU->template  adjustSize<floatT>(this->RSonmax);
    Ptrplc8onCPU->template  adjustSize<floatT>(this->RSonmax);
    Ptrplc1offCPU->template copyFrom<true>(Ptrplc1offGPU, Ptrplc1offGPU->getSize());
    Ptrplc8offCPU->template copyFrom<true>(Ptrplc8offGPU, Ptrplc8offGPU->getSize());
    PtrplcaonCPU->template  copyFrom<true>(PtrplcaonGPU , PtrplcaonGPU->getSize());
    Ptrplc1onCPU->template  copyFrom<true>(Ptrplc1onGPU , Ptrplc1onGPU->getSize());
    Ptrplc8onCPU->template  copyFrom<true>(Ptrplc8onGPU , Ptrplc8onGPU->getSize());
    MemoryAccessor _plc1offCPU (Ptrplc1offCPU->getPointer());
    MemoryAccessor _plc8offCPU (Ptrplc8offCPU->getPointer());
    MemoryAccessor _plcaonCPU  (PtrplcaonCPU->getPointer());
    MemoryAccessor _plc1onCPU  (Ptrplc1onCPU->getPointer());
    MemoryAccessor _plc8onCPU  (Ptrplc8onCPU->getPointer());

    /// Calculate the correct distances.
    for (int dx=0 ; dx<(this->distmax) ; dx++) {
        vec_plca[dx]   = 0.;
        vec_plc1[dx]   = 0.;
        vec_plc8[dx]   = 0.;
    }
    for (int dx=0 ; dx<(this->RSxmax) ; dx++)
    for (int dy=0 ; dy<(this->RSymax) ; dy++)
    for (int dz=0 ; dz<(this->RSzmax) ; dz++) {
        qnorm = dx*dx+dy*dy+dz*dz;
        if (qnorm>(this->distmax)) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
        psite = dx + dy*(this->pvol1) + dz*(this->pvol2);
        g     = vec_weight[psite];
        _plcaoffCPU.getValue<floatT>(psite,plcaoff);
        _plc1offCPU.getValue<floatT>(psite,plc1off);
        _plc8offCPU.getValue<floatT>(psite,plc8off);
        /// Remember that g counted the number of unique vectors at displacement qnorm. Unique vectors therefore get
        /// more weight than overcounted vectors.
        vec_plca[qnorm] += g*plcaoff;
        vec_plc1[qnorm] += g*plc1off;
        vec_plc8[qnorm] += g*plc8off;
    }
    for (int dx=(this->RSxmax);dx<(this->RSonmax);dx++) {
        qnorm = dx*dx;
        if (qnorm>(this->distmax)) throw std::runtime_error(stdLogger.fatal("qnorm > distmax"));
        g = 3;
        _plcaonCPU.getValue<floatT>(dx,plcaon);
        _plc1onCPU.getValue<floatT>(dx,plc1on);
        _plc8onCPU.getValue<floatT>(dx,plc8on);
        vec_plca[qnorm] += g*plcaon;
        vec_plc1[qnorm] += g*plc1on;
        vec_plc8[qnorm] += g*plc8on;
    }
    for (int dx=0 ; dx<(this->distmax) ; dx++) {
        if (vec_factor[dx]>0) {
            /// Apply the correct weights to the correlators.
            vec_plca[dx]=vec_plca[dx]/((floatT)vec_factor[dx]);
            vec_plc1[dx]=vec_plc1[dx]/((floatT)vec_factor[dx]);
            vec_plc8[dx]=vec_plc8[dx]/((floatT)vec_factor[dx]);
        }
    }
}

/// Initialize various possibilities of template parameter combinations for the correlators class.
#define CLASS_INIT(floatT,HALO) \
template class PolyakovLoopCorrelator<floatT,true,HALO>;
INIT_PH(CLASS_INIT)
