#ifndef VECTARRAY
#define VECTARRAY


#include "complex.h"
#include "vect.h"
#include "../indexer/bulkIndexer.h"
#include "stackedArray.h"
#include "generalAccessor.h"
#include "../utilities/static_for_loop.h"

template<class floatT, size_t elems>
struct VectArrayAcc : public GeneralAccessor<GPUcomplex<floatT>, elems > {


    explicit VectArrayAcc(GPUcomplex<floatT> *const elements[elems])
            : GeneralAccessor<GPUcomplex<floatT>, elems >(elements) {
    }

    __host__ __device__ explicit VectArrayAcc(GPUcomplex<floatT> *elementsBase, size_t object_count)
            : GeneralAccessor<GPUcomplex<floatT>, elems >(elementsBase, object_count) {
    }
    explicit VectArrayAcc() : GeneralAccessor<GPUcomplex<floatT>, elems >() { }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect<floatT_compute,elems> getElement(const gSite &site) const {
        Vect<floatT_compute,elems> res;

        static_for<0, elems>::apply([&](auto i){
            res.template setElement<i>(GPUcomplex<floatT_compute>(static_cast<floatT_compute>(this->template getElementEntry<i>(site.isiteFull).cREAL),
                                                static_cast<floatT_compute>(this->template getElementEntry<i>(site.isiteFull).cIMAG)));
        });

        return res;
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElement(const gSite &site, const Vect<floatT_compute,elems> &vec) {
        static_for<0, elems>::apply([&](auto i){
            this->template setElementEntry<i>(site.isiteFull, vec.template getElement<i>());
        });
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect<floatT_compute,elems> getElement(const gSiteStack &site) const {

        Vect<floatT_compute,elems> res;

        static_for<0, elems>::apply([&](auto i)
        {
            res.template setElement<i>(GPUcomplex<floatT_compute>(static_cast<floatT_compute>(this->template getElementEntry<i>(site.isiteStackFull).cREAL),
                                                static_cast<floatT_compute>(this->template getElementEntry<i>(site.isiteStackFull).cIMAG)));
        });
        return res;
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElement(const gSiteStack &site, const Vect<floatT_compute,elems> &vec) {

        static_for<0, elems>::apply([&](auto i)
        {
            this->template setElementEntry<i>(site.isiteStackFull, vec.template getElement<i>());
        });
    }

    __host__ __device__ inline void setEntriesComm(VectArrayAcc<floatT,elems> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        static_for<0, elems>::apply([&](auto i)
        {
            this->template setElementEntry<i>(setIndex, src_acc.template getElementEntry<i>(getIndex));
        });
    }

    template<Layout LatLayout, size_t HaloDepth>
    __host__ __device__ inline size_t getIndexComm(size_t isiteFull, size_t stack) const {
        gSiteStack site = GIndexer<LatLayout, HaloDepth>::getSiteStackFull(isiteFull, stack);
        return site.isiteStackFull;
    }


    template<Layout LatLayout, size_t HaloDepth,class floatT_compute=floatT>
    __host__ __device__ inline Vect<floatT_compute,elems> getElementComm(size_t isiteFull, size_t stack) const {
        gSiteStack site = GIndexer<LatLayout, HaloDepth>::getSiteStackFull(isiteFull, stack);
        return getElement<floatT_compute>(site);
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElementComm(size_t isiteFull, __attribute__((unused)) size_t stack, const Vect<floatT_compute,elems> &vec) {
        gSiteStack site;
        site.isiteFull = isiteFull;
        site.isiteStackFull = isiteFull;
        setElement<floatT_compute>(site, vec);
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect<floatT_compute,elems> operator()(const gSite &site) const {
        return this->getElement<floatT_compute>(site);
    };

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect<floatT_compute,elems> operator()(const gSiteStack &site) const {
        return this->getElement<floatT_compute>(site);
    };
};


template<class floatT, size_t elems_, bool onDevice>
class VectArray : public stackedArray<onDevice, GPUcomplex<floatT>, elems_> {

public:
    friend class VectArray<floatT,elems_, !onDevice>;

    explicit VectArray(const size_t elems, std::string VectArrayName="VectArray") :
            stackedArray<onDevice, GPUcomplex<floatT>, elems_>(elems,VectArrayName) {}

    VectArrayAcc<floatT,elems_> getAccessor() const {
        GPUcomplex<floatT> *elements[elems_]; 
        static_for<0, elems_>::apply([&](auto i){
             elements[i] = this->template getEntry<GPUcomplex<floatT> >(i);
        });
        return VectArrayAcc<floatT,elems_>(elements);
    }
};

template<class floatT>
using Vect3ArrayAcc = VectArrayAcc<floatT,3>;

template<class floatT, bool onDevice>
using Vect3Array = VectArray<floatT,3,onDevice>;



#endif
