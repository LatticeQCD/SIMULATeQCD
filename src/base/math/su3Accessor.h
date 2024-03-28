//
// Created by Lukas Mazur on 17.12.18.
//

#pragma once
#include "su3Constructor.h"

#define BACKWARD_CONST 16

__host__ __device__ inline int Back(const int i) {
    return i + BACKWARD_CONST;
}


template<class floatT_memory, CompressionType comp = R18>
class SU3Accessor : public GaugeConstructor<floatT_memory, comp> {

public:

    explicit SU3Accessor(COMPLEX(floatT_memory) *const elements[EntryCount<comp>::count])
            : GaugeConstructor<floatT_memory, comp>(elements) {}

    /// Constructor for one memory chunk, where all entries are separated by object_count
    __host__ __device__ explicit SU3Accessor(COMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GaugeConstructor<floatT_memory, comp>(elementsBase, object_count) {}

    explicit SU3Accessor() : GaugeConstructor<floatT_memory, comp>() {}

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> getElement(const gSiteMu &siteMu) const {
        return static_cast<SU3<floatT_compute>>(this->reconstruct(siteMu));
    }

    template<Layout LatLayout, size_t HaloDepth>
    __host__ __device__ inline size_t getIndexComm(size_t isiteFull, size_t mu) const {
        gSiteMu siteMu = GIndexer<LatLayout, HaloDepth>::getSiteMuFull(isiteFull, mu);
        return siteMu.indexMuFull;
    }

    template<Layout LatLayout, size_t HaloDepth, class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> getElementComm(size_t isiteFull, size_t mu) const {
        gSiteMu siteMu = GIndexer<LatLayout, HaloDepth>::getSiteMuFull(isiteFull, mu);
        return getElement<floatT_compute>(siteMu);
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline void setElementComm(size_t isiteFull, const SU3<floatT_compute>& mat) {
        gSiteMu siteMu;
        siteMu.indexMuFull = isiteFull;
        setElement<floatT_compute>(siteMu, mat);
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline void setElement(const gSiteMu &siteMu, const SU3<floatT_compute> &mat) {
        this->construct(siteMu, static_cast<SU3<floatT_memory>>(mat));
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> getLink(const gSiteMu &siteMu) const {
        return static_cast<SU3<floatT_compute>>(this->reconstruct(siteMu));
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> getLinkDagger(const gSiteMu &siteMu) const {
        return static_cast<SU3<floatT_compute>>(this->reconstructDagger(siteMu));
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline void setLink(const gSiteMu &siteMu, SU3<floatT_compute> mat) {
        this->construct(siteMu, static_cast<SU3<floatT_memory>>(mat));
    }

    template<class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> operator()(const gSiteMu &siteMu) const {
        return static_cast<SU3<floatT_compute>>(this->reconstruct(siteMu));
    }

    template<Layout LatLayout, size_t HaloDepth, class floatT_compute=floatT_memory>
    __host__ __device__ inline SU3<floatT_compute> getLinkPath(gSite &site, int dir) const {

        typedef GIndexer<LatLayout, HaloDepth> GInd;

        SU3<floatT_compute> ret;

        bool back = false;

        if (dir >= BACKWARD_CONST) {
            back = true;
            dir -= BACKWARD_CONST;
        }

        if (!back) {
            ret = getElement<floatT_compute>(GInd::getSiteMu(site, dir));
            site = GInd::site_up(site, dir);
        } else {
            site = GInd::site_dn(site, dir);
            ret = getLinkDagger<floatT_compute>(GInd::getSiteMu(site, dir));
        }
        return ret;
    }


    template<Layout LatLayout, size_t HaloDepth, class floatT_compute=floatT_memory, typename... Args>
    __host__ __device__ inline SU3<floatT_compute> getLinkPath(gSite &site, int dir, Args... args) const {

        typedef GIndexer<LatLayout, HaloDepth> GInd;

        SU3<floatT_compute> ret;

        bool back = false;

        if (dir >= BACKWARD_CONST) {
            back = true;
            dir -= BACKWARD_CONST;
        }

        if (!back) {
            ret = getElement<floatT_compute>(GInd::getSiteMu(site, dir));
            site = GInd::site_up(site, dir);
        } else {
            site = GInd::site_dn(site, dir);
            ret = getLinkDagger<floatT_compute>(GInd::getSiteMu(site, dir));
        }

        return ret * getLinkPath<LatLayout, HaloDepth, floatT_compute>(site, args...);
    }

    template<Layout LatLayout, size_t HaloDepth, class floatT_compute=floatT_memory, typename... Args>
    __host__ __device__ inline SU3<floatT_compute> getLinkPath(gSiteMu &siteMu, int dir, Args... args) const {
        typedef GIndexer<LatLayout, HaloDepth> GInd;

        gSite site = siteMu;
        SU3<floatT_compute> ret = getLinkPath<LatLayout, HaloDepth, floatT_compute>(site, siteMu.mu, dir, args...);
        siteMu = GInd::getSiteMu(site, siteMu.mu);
        return ret;
    }
};

