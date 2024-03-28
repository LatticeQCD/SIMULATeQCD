/*
 * colorMagneticCorr.cpp
 *
 * v1.0: Hai-Tao Shu, 6 Nov 2020
 * Measure Color-Magnetic Correlator using the multi-GPU framework. Read sketch from right to left
 *
 *                x
 *                ^    ^ y
 *                |   /
 *                |  /
 *                | /
 *     t <_______ |/
 *
 *              <-----------
 *             /|          ^|
 *            / |         / |
 *           /  |        /  |
 *          /   V       /   ^
 *          | - /       | - /   read from right to left
 *          |  /        |  /
 *          | /         | /
 * <------  v           |<     tau direction is the horizontal one. we should consider all situations, i.e. squares in xy, yz, and zx plane. and for each plane there are 4 combinations, for instance in xy plane the links of the squres could start from +x+y, +x-y, -x-y, -x+y directions.
 *
 * B_0=F_{12}=U_1(\vec{x})U_2(\vec{x}+\hat{1})-U_2(\vec{x})U_1(\vec{x}+\hat{2})
 * B_1=F_{20}=U_2(\vec{x})U_0(\vec{x}+\hat{2})-U_0(\vec{x})U_2(\vec{x}+\hat{0})
 * B_2=F_{01}=U_0(\vec{x})U_1(\vec{x}+\hat{0})-U_1(\vec{x})U_0(\vec{x}+\hat{1})
 *
 */

#include "colorMagneticCorr.h"
#include "fieldStrengthTensor.h"

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
struct ColorMagnSecondWilLineKernel{
    SU3Accessor<floatT> SU3Accessor;
    MemoryAccessor ColorMagnSecondWilLineGpu;

    ColorMagnSecondWilLineKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorMagnSecondWilLineGpu) :
            SU3Accessor(gauge.getAccessor()), ColorMagnSecondWilLineGpu(ColorMagnSecondWilLineGpu) {}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Nt = (int)GInd::getLatData().globLT;
        size_t Id = site.isite;
        //the second wilson line part
        SU3<floatT> wl = su3_one<floatT>();
        gSite rsite = site;
        rsite = GInd::site_up(rsite, 3);
        for ( size_t i = 0; i < Nt-1; ++i)
        {
            wl *= SU3Accessor.getLink(GInd::getSiteMu(rsite, 3));
            rsite = GInd::site_up(rsite, 3);
        }
        ColorMagnSecondWilLineGpu.setValue<SU3<floatT>>(Id, wl);
    }
};



template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorMagnNaiveKernel{
    SU3Accessor<floatT> SU3Accessor;
    MemoryAccessor ColorMagnSecondWilLineGpu;
    size_t dt; //current "delta-time slice"
    ColorMagnNaiveKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorMagnSecondWilLineGpu, size_t dt) : SU3Accessor(gauge.getAccessor
    ()), ColorMagnSecondWilLineGpu(ColorMagnSecondWilLineGpu), dt(dt) {
    }
    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;
        const size_t Ntau=GInd::getLatData().lt;
        SU3<floatT> p_plusplus;
        SU3<floatT> p_minusminus;
        SU3<floatT> p_plusminus;
        SU3<floatT> p_minusplus;
        COMPLEX(floatT) result(0,0);

        size_t Id = site.isite;
        SU3<floatT> wl2;//the second wilsone line
        ColorMagnSecondWilLineGpu.getValue<SU3<floatT>>(Id, wl2);

        gSite rsite_plusplus;
        gSite rsite_minusminus;
        gSite rsite_plusminus;
        gSite rsite_minusplus;
        ///calculate ColorMagneticCorr for given delta_t (1<=dt<Ntau)
        ///time direction is the last one, 3
        for ( size_t mu = 0; mu <= 2; ++mu){
            ///use "running sites" to go trough ColorMagneticCorr path
            rsite_plusplus = site;
            rsite_minusminus = site;
            rsite_plusminus = site;
            rsite_minusplus = site;
            ///first square
            p_plusplus = SU3Accessor.getLink(GInd::getSiteMu(rsite_plusplus, mu))
                       * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_plusplus, mu), (mu+1)%3))
                       - SU3Accessor.getLink(GInd::getSiteMu(rsite_plusplus, (mu+1)%3))
                       * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_plusplus, (mu+1)%3), mu));

            p_minusminus = SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_minusminus, mu), mu))
                         * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(rsite_minusminus, mu, (mu+1)%3), (mu+1)%3))
                         - SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_minusminus, (mu+1)%3),(mu+1)%3))
                         * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(rsite_minusminus, (mu+1)%3, mu), mu));

            p_plusminus = SU3Accessor.getLink(GInd::getSiteMu(rsite_plusminus, mu))
                        * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_plusminus, mu, (mu+1)%3), (mu+1)%3))
                        - SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_plusminus, (mu+1)%3), (mu+1)%3))
                        * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_plusminus, (mu+1)%3), mu));

            p_minusplus = SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_minusplus, mu), mu))
                        * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_minusplus, mu), (mu+1)%3))
                        - SU3Accessor.getLink(GInd::getSiteMu(rsite_minusplus, (mu+1)%3))
                        * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_minusplus, (mu+1)%3, mu), mu));
            ///change rsites to opposite ends of first squares
            rsite_plusplus = GInd::site_up_up(rsite_plusplus, (mu+1)%3, mu);
            rsite_minusminus = GInd::site_dn_dn(rsite_minusminus, (mu+1)%3, mu);
            rsite_plusminus = GInd::site_up_dn(rsite_plusminus, mu, (mu+1)%3);
            rsite_minusplus = GInd::site_up_dn(rsite_minusplus, (mu+1)%3, mu);
            ///first wilson line, length: dt
            for ( size_t i = 1; i <= dt; ++i){
                p_plusplus *= SU3Accessor.getLink(GInd::getSiteMu(rsite_plusplus, 3));
                p_minusminus *= SU3Accessor.getLink(GInd::getSiteMu(rsite_minusminus, 3));
                p_plusminus *= SU3Accessor.getLink(GInd::getSiteMu(rsite_plusminus, 3));
                p_minusplus *= SU3Accessor.getLink(GInd::getSiteMu(rsite_minusplus, 3));
                ///advance rsites in time
                rsite_plusplus = GInd::site_up(rsite_plusplus, 3);
                rsite_minusminus = GInd::site_up(rsite_minusminus, 3);
                rsite_plusminus = GInd::site_up(rsite_plusminus, 3);
                rsite_minusplus = GInd::site_up(rsite_minusplus, 3);
            }
            ///second square
            p_plusplus *= SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_plusplus, mu), mu))
                        * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(rsite_plusplus, mu, (mu+1)%3), (mu+1)%3))
                        - SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_plusplus, (mu+1)%3), (mu+1)%3))
                        * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(rsite_plusplus, (mu+1)%3, mu), mu));
            p_minusminus *= SU3Accessor.getLink(GInd::getSiteMu(rsite_minusminus, mu))
                          * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_minusminus, mu), (mu+1)%3))
                          - SU3Accessor.getLink(GInd::getSiteMu(rsite_minusminus, (mu+1)%3))
                          * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(rsite_minusminus, (mu+1)%3), mu));

            p_plusminus *= SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_plusminus, mu), mu))
                         * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_plusminus, mu), (mu+1)%3))
                         - SU3Accessor.getLink(GInd::getSiteMu(rsite_plusminus, (mu+1)%3))
                         * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_plusminus, (mu+1)%3, mu), mu));
            p_minusplus *=SU3Accessor.getLink(GInd::getSiteMu(rsite_minusplus, mu))
                        * SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(rsite_minusplus, mu, (mu+1)%3), (mu+1)%3))
                        - SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(rsite_minusplus, (mu+1)%3), (mu+1)%3))
                        * SU3Accessor.getLink(GInd::getSiteMu(GInd::site_dn(rsite_minusplus, (mu+1)%3), mu));

            result += tr_c(p_plusplus*wl2+p_minusminus*wl2+p_minusplus*wl2+p_plusminus*wl2);
        }
        rsite_plusplus = GInd::site_dn_dn(rsite_plusplus, 0, 2);

        SU3<floatT> temp;
        temp = SU3Accessor.getLinkDagger(GInd::getSiteMu(rsite_plusplus, 3));
        ColorMagnSecondWilLineGpu.setValue<SU3<floatT>>(Id, temp*wl2);

        return result / (-12.*3); //tr(unity matrix) = 3. 12 is for normalization. -1 is because such discretization produces extra minus sign than the proper definition.
    }
};
///call this to get the color-magnetic correlator. Don't forget to exchange halos before this!
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
std::vector<COMPLEX(floatT)> ColorMagneticCorr<floatT, onDevice, HaloDepth, comp>::getColorMagneticCorr_naive() {
    ///exit if lattice is split in time
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    size_t vol4 = GInd::getLatData().vol4;

    typedef gMemoryPtr<true> MemTypeGPU;
    MemTypeGPU mem75 = MemoryManagement::getMemAt<true>("ColorMagnNaive2ndWilLineGpu");
    mem75->template adjustSize<SU3<floatT>>(vol4);
    MemoryAccessor ColorMagnNaiveSecondWilLineGpu (mem75->getPointer());
    mem75->memset(0);
    ReadIndexSpacetime<HaloDepth> passReadIndex;

    iterateFunctorNoReturn<onDevice>(ColorMagnSecondWilLineKernel<floatT, onDevice, HaloDepth>(_gauge, ColorMagnNaiveSecondWilLineGpu), passReadIndex, vol4);
    std::vector<COMPLEX(floatT)> ColorMagntricCorrNaive_result(Ntau/2);
    ///calculate ColorMagntricCorr for all time differences dt>0. The correlator is symmetric; it is sufficient
    /// to only calculate up to Ntau/2.
    for (size_t dt = 1; dt <= Ntau/2; ++dt) {
        _redBase.template iterateOverBulk<All, HaloDepth>
                (ColorMagnNaiveKernel<floatT, onDevice, HaloDepth>(_gauge, ColorMagnNaiveSecondWilLineGpu, dt));
        _redBase.reduce(ColorMagntricCorrNaive_result[dt - 1], elems);
        ColorMagntricCorrNaive_result[dt-1] /= vol;
    }
    return ColorMagntricCorrNaive_result;
}



template<class floatT, bool onDevice, size_t HaloDepth>
struct ColorMagnCloverKernel{

    SU3Accessor<floatT> SU3Accessor;
    MemoryAccessor ColorMagnCloverSecondWilLineGpu;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;
    size_t dt; //current "delta-time slice"

    ColorMagnCloverKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor ColorMagnCloverSecondWilLineGpu, size_t dt) : SU3Accessor(gauge.getAccessor()), FT(gauge.getAccessor()), ColorMagnCloverSecondWilLineGpu(ColorMagnCloverSecondWilLineGpu), dt(dt) {
    }
    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        const size_t Ntau=GInd::getLatData().lt;

        SU3<floatT> Corr; ///part going downwards

        COMPLEX(floatT) result(0,0);

        size_t Id = site.isite;
        SU3<floatT> wl2;//the second wilsone line
        ColorMagnCloverSecondWilLineGpu.getValue<SU3<floatT>>(Id, wl2);

        gSite new_site;
        for ( size_t mu = 0; mu <= 2; ++mu){
            new_site = site;

            //first magnetic field
            Corr = FT(site, mu, (mu+1)%3);

            ///first wilson line, length: dt
            for ( size_t i = 1; i <= dt; ++i){
                Corr *= SU3Accessor.getLink(GInd::getSiteMu(new_site, 3));
                ///move one step forward in time
                new_site = GInd::site_up(new_site, 3);
            }
            ///second magnetic field
            Corr *= FT(new_site, mu, (mu+1)%3);

            result += tr_c(Corr*wl2);
        }
        SU3<floatT> temp;
        temp = SU3Accessor.getLinkDagger(GInd::getSiteMu(new_site, 3));
        ColorMagnCloverSecondWilLineGpu.setValue<SU3<floatT>>(Id, temp*wl2);

        return result / (3.*3.); //tr(unity matrix) = 3
    }
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
std::vector<COMPLEX(floatT)> ColorMagneticCorr<floatT, onDevice, HaloDepth, comp>::getColorMagneticCorr_clover() {
    if (_gauge.getComm().nodes()[3] != 1){
        throw std::runtime_error(stdLogger.fatal("Do not split lattice in time direction!"));
    }

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Nt = (size_t)GInd::getLatData().globLT;
    size_t vol4 = GInd::getLatData().vol4;

    typedef gMemoryPtr<true> MemTypeGPU;
    MemTypeGPU mem76 = MemoryManagement::getMemAt<true>("ColorMagnClover2ndWilLineGpu");
    mem76->template adjustSize<SU3<floatT>>(vol4);
    MemoryAccessor ColorMagnCloverSecondWilLineGpu (mem76->getPointer());
    mem76->memset(0);
    ReadIndexSpacetime<HaloDepth> passReadIndex;

    iterateFunctorNoReturn<onDevice>(ColorMagnSecondWilLineKernel<floatT, onDevice, HaloDepth>(_gauge, ColorMagnCloverSecondWilLineGpu), passReadIndex, vol4);

    std::vector<COMPLEX(floatT)> ColorMagntricCorrClover_result(Ntau/2);
    for (size_t dt = 1; dt <= Ntau/2; ++dt) {
        _redBase.template iterateOverBulk<All, HaloDepth>(ColorMagnCloverKernel<floatT, onDevice, HaloDepth>(_gauge, ColorMagnCloverSecondWilLineGpu, dt));
        _redBase.reduce(ColorMagntricCorrClover_result[dt - 1], elems);
        ColorMagntricCorrClover_result[dt-1] /= vol;
    }
    return ColorMagntricCorrClover_result;
}
///explicitly instantiate various instances of the class
#define INIT_ONDEVICE_TRUE(floatT, HALO, comp) \
template class ColorMagneticCorr<floatT,true,HALO, comp>;
#define INIT_ONDEVICE_FALSE(floatT,HALO, comp) \
template class ColorMagneticCorr<floatT,false,HALO, comp>;

INIT_PHC(INIT_ONDEVICE_TRUE)
INIT_PHC(INIT_ONDEVICE_FALSE)
