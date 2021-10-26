/*
 * ColorElectricCorr.h
 *
 * v1.0: L. Altenkort, 28 Jan 2019
 *
 * Measure Color-Electric Correlator (ColorElectricCorr) using the multi-GPU framework. Read sketch from right to left
 * (time advances to the left, space advances to the top)
 *          <----   <------  ^ <---^
 *          |  -  |          |  -  |   +  flipped = "going downwards" + "going upwards"
 * <------  v <---v          <----
 *
 */

#include "ColorElectricCorr.h"
#include "FieldStrengthTensor.h"

template<size_t HaloDepth>
struct ReadIndexSpacetime {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<All, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorElecNaiveSecondWilLineKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor ColorElecNaiveSecondWilLineGpu;

    ColorElecNaiveSecondWilLineKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorElecNaiveSecondWilLineGpu) :
            gaugeAccessor(gauge.getAccessor()), ColorElecNaiveSecondWilLineGpu(ColorElecNaiveSecondWilLineGpu) {}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Nt = (int)GInd::getLatData().globLT;
        size_t Id = site.isite;
        //the second wilson line part
        GSU3<floatT> wl = gsu3_one<floatT>();
        gSite rsite = site;
        rsite = GInd::site_up_up(rsite, 3, 3);
        for ( size_t i = 0; i < Nt-2; ++i)
        {
            wl *= gaugeAccessor.getLink(GInd::getSiteMu(rsite, 3));
            rsite = GInd::site_up(rsite, 3);
        }
        ColorElecNaiveSecondWilLineGpu.setValue<GSU3<floatT>>(Id, wl);
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorElecCloverSecondWilLineKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor ColorElecCloverSecondWilLineGpu;
    
    ColorElecCloverSecondWilLineKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorElecCloverSecondWilLineGpu) :
            gaugeAccessor(gauge.getAccessor()), ColorElecCloverSecondWilLineGpu(ColorElecCloverSecondWilLineGpu) {}
    
    __device__ __host__ void operator()(gSite site) {
        
        typedef GIndexer<All,HaloDepth> GInd;
        int Nt = (int)GInd::getLatData().globLT;
        size_t Id = site.isite;
        //the second wilson line part
        GSU3<floatT> wl = gsu3_one<floatT>();
        gSite rsite = site;
        rsite = GInd::site_up(rsite, 3);
        for ( size_t i = 0; i < Nt-1; ++i)
        {   
            wl *= gaugeAccessor.getLink(GInd::getSiteMu(rsite, 3));
            rsite = GInd::site_up(rsite, 3);
        }
        ColorElecCloverSecondWilLineGpu.setValue<GSU3<floatT>>(Id, wl);
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorElecNaiveKernel{
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor ColorElecNaiveSecondWilLineGpu;
    size_t dt;
    ColorElecNaiveKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorElecNaiveSecondWilLineGpu, size_t dt) :
            gaugeAccessor(gauge.getAccessor()), ColorElecNaiveSecondWilLineGpu(ColorElecNaiveSecondWilLineGpu), dt(dt) {}

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        GCOMPLEX(floatT) result(0,0);

        size_t Id = site.isite;

        GSU3<floatT> wl2;//the second wilsone line
        ColorElecNaiveSecondWilLineGpu.getValue<GSU3<floatT>>(Id, wl2);

        gSite rsite_dn;
        gSite rsite_up;
        GSU3<floatT> p_up;
        GSU3<floatT> p_dn;
        for ( size_t mu = 0; mu <= 2; ++mu){

            rsite_dn = site;
            rsite_up = site;
            ///first square
            p_dn =    gaugeAccessor.getLink(GInd::getSiteMu(rsite_dn, mu))
                      * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_dn, mu), 3))
                      - gaugeAccessor.getLink(GInd::getSiteMu(rsite_dn, 3))
                        * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_dn, 3), mu));
            p_up =    gaugeAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_up, mu), mu))
                      * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_up, mu), 3))
                      - gaugeAccessor.getLink(GInd::getSiteMu(rsite_up, 3))
                        * gaugeAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_up, 3, mu), mu));

            ///change rsites to opposite ends of first squares
            rsite_dn = GInd::site_up_up(rsite_dn, 3, mu);
            rsite_up = GInd::site_up_dn(rsite_up, 3, mu);

            ///first wilson line, length: dt-1
            for ( size_t i = 1; i <= dt-1; ++i){
                p_dn *= gaugeAccessor.getLink(GInd::getSiteMu(rsite_dn, 3));
                p_up *= gaugeAccessor.getLink(GInd::getSiteMu(rsite_up, 3));
                ///advance rsites in time
                rsite_dn = GInd::site_up(rsite_dn, 3);
                rsite_up = GInd::site_up(rsite_up, 3);
            }
            ///second square
            p_dn *=    gaugeAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_dn, mu), mu))
                       * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_dn, mu), 3))
                       - gaugeAccessor.getLink(GInd::getSiteMu(rsite_dn, 3))
                         * gaugeAccessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_dn, 3, mu), mu));
            p_up *=    gaugeAccessor.getLink(GInd::getSiteMu(rsite_up, mu))
                       * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_up, mu), 3))
                       - gaugeAccessor.getLink(GInd::getSiteMu(rsite_up, 3))
                         * gaugeAccessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_up, 3), mu));
            rsite_dn = GInd::site_up_dn(rsite_dn, 3, mu);
            p_dn *= wl2;
            p_up *= wl2;
            result += tr_c(p_up + p_dn);
        }
        GSU3<floatT> temp;
        temp = gaugeAccessor.getLinkDagger(GInd::getSiteMu(rsite_dn, 3));
        ColorElecNaiveSecondWilLineGpu.setValue<GSU3<floatT>>(Id, temp*wl2);

        return result / (-6.*3.);
    }
};
///call this to get the color-electric correlator. Don't forget to exchange halos before this!
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
std::vector<GCOMPLEX(floatT)> ColorElectricCorr<floatT, onDevice, HaloDepth, comp>::getColorElectricCorr_naive() {
    ///exit if lattice is split in time
    if (_gauge.getComm().nodes()[3] != 1){
        throw PGCError("Do not split lattice in time direction for color-electric correlator computation!");
    }

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    size_t vol4 = GInd::getLatData().vol4;

    typedef gMemoryPtr<true> MemTypeGPU;
    MemTypeGPU mem73 = MemoryManagement::getMemAt<true>("ColorElecNaive2ndWilLineGpu");
    mem73->template adjustSize<GSU3<floatT>>(vol4);
    MemoryAccessor ColorElecNaiveSecondWilLineGpu (mem73->getPointer());
    mem73->memset(0);
    ReadIndexSpacetime<HaloDepth> passReadIndex;

    iterateFunctorNoReturn<onDevice>(ColorElecNaiveSecondWilLineKernel<floatT, onDevice, HaloDepth>(_gauge, ColorElecNaiveSecondWilLineGpu), passReadIndex, vol4);
    std::vector<GCOMPLEX(floatT)> ColorElectricCorrNaive_result(Ntau/2);
    ///calculate ColorElectricCorr for all time differences dt>0. The correlator is symmetric; it is sufficient
    /// to only calculate up to Ntau/2.
    for (size_t dt = 1; dt <= Ntau/2; ++dt) {
        _redBase.template iterateOverBulk<All, HaloDepth>
                (ColorElecNaiveKernel<floatT, onDevice, HaloDepth>(_gauge, ColorElecNaiveSecondWilLineGpu, dt));
        _redBase.reduce(ColorElectricCorrNaive_result[dt - 1], elems);
        ColorElectricCorrNaive_result[dt-1] /= vol;
    }
    return ColorElectricCorrNaive_result;
}

template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorElecCloverKernel{

    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor ColorElecCloverSecondWilLineGpu;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;
    size_t dt; //current "delta-time slice"

    ColorElecCloverKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorElecCloverSecondWilLineGpu, size_t dt) : gaugeAccessor(gauge.getAccessor()), FT(gauge.getAccessor()), ColorElecCloverSecondWilLineGpu(ColorElecCloverSecondWilLineGpu), dt(dt) {
    }
    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        const size_t Ntau=GInd::getLatData().lt;

        GSU3<floatT> Corr; ///part going downwards

        GCOMPLEX(floatT) result(0,0);

        size_t Id = site.isite;
        GSU3<floatT> wl2;//the second wilsone line
        ColorElecCloverSecondWilLineGpu.getValue<GSU3<floatT>>(Id, wl2);

        gSite new_site;
        for ( size_t mu = 0; mu <= 2; ++mu){
            new_site = site;

            //first electric field
            Corr = FT(site, mu, 3);

            ///first wilson line, length: dt
            for ( size_t i = 1; i <= dt; ++i){
                Corr *= gaugeAccessor.getLink(GInd::getSiteMu(new_site, 3));
                ///move one step forward in time
                new_site = GInd::site_up(new_site, 3);
            }
            ///second electric field
            Corr *= FT(new_site, mu, 3);

            result += tr_c(Corr*wl2);
        }
        GSU3<floatT> temp;
        temp = gaugeAccessor.getLinkDagger(GInd::getSiteMu(new_site, 3));
        ColorElecCloverSecondWilLineGpu.setValue<GSU3<floatT>>(Id, temp*wl2);
        //! average over 3 because tr(unity matrix)=3.
        //! average over 3 because of the 3 realizations of the observable using clover on the lattice
        //! minus is convention
        return result / (-3.*3.);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
std::vector<GCOMPLEX(floatT)> ColorElectricCorr<floatT, onDevice, HaloDepth, comp>::getColorElectricCorr_clover() {
    if (_gauge.getComm().nodes()[3] != 1){
        throw PGCError("Do not split lattice in time direction!");
    }
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    size_t vol4 = GInd::getLatData().vol4;

    typedef gMemoryPtr<true> MemTypeGPU;
    MemTypeGPU mem74 = MemoryManagement::getMemAt<true>("ColorElecClover2ndWilLineGpu");
    mem74->template adjustSize<GSU3<floatT>>(vol4);
    MemoryAccessor ColorElecCloverSecondWilLineGpu (mem74->getPointer());
    mem74->memset(0);
    ReadIndexSpacetime<HaloDepth> passReadIndex;

    iterateFunctorNoReturn<onDevice>(ColorElecCloverSecondWilLineKernel<floatT, onDevice, HaloDepth>(_gauge, ColorElecCloverSecondWilLineGpu), passReadIndex, vol4);

    std::vector<GCOMPLEX(floatT)> ColorElectricCorrClover_result(Ntau/2);
    for (size_t dt = 1; dt <= Ntau/2; ++dt) {
        _redBase.template iterateOverBulk<All, HaloDepth>(ColorElecCloverKernel<floatT, onDevice, HaloDepth>(_gauge, ColorElecCloverSecondWilLineGpu, dt));
        _redBase.reduce(ColorElectricCorrClover_result[dt - 1], elems);
        ColorElectricCorrClover_result[dt-1] /= vol;
    }
    return ColorElectricCorrClover_result;
}
///explicitly instantiate various instances of the class
#define INIT_ONDEVICE_TRUE(floatT, HALO, comp) \
template class ColorElectricCorr<floatT,true,HALO, comp>;
#define INIT_ONDEVICE_FALSE(floatT,HALO, comp) \
template class ColorElectricCorr<floatT,false,HALO, comp>;

INIT_PHC(INIT_ONDEVICE_TRUE)
INIT_PHC(INIT_ONDEVICE_FALSE)
