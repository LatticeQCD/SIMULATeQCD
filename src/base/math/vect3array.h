#ifndef GVECT3ARRAY_H_
#define GVECT3ARRAY_H_


#include "complex.h"
#include "vect3.h"
#include "../indexer/bulkIndexer.h"
#include "stackedArray.h"
#include "generalAccessor.h"


template<class floatT>
struct Vect3arrayAcc : public GeneralAccessor<COMPLEX(floatT), 3 > {


    explicit Vect3arrayAcc(COMPLEX(floatT) *const elements[3])
            : GeneralAccessor<COMPLEX(floatT), 3 >(elements) {
    }

    __host__ __device__ explicit Vect3arrayAcc(COMPLEX(floatT) *elementsBase, size_t object_count)
            : GeneralAccessor<COMPLEX(floatT), 3 >(elementsBase, object_count) {
    }
    explicit Vect3arrayAcc() : GeneralAccessor<COMPLEX(floatT), 3 >() { }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect3<floatT_compute> getElement(const gSite &site) const {
        return static_cast<Vect3<floatT_compute>>(Vect3<floatT>(
                this->template getElementEntry<0>(site.isiteFull),
                this->template getElementEntry<1>(site.isiteFull),
                this->template getElementEntry<2>(site.isiteFull)));
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElement(const gSite &site, const Vect3<floatT_compute> &vec) {
        this->template setElementEntry<0>(site.isiteFull, vec.getElement0());
        this->template setElementEntry<1>(site.isiteFull, vec.getElement1());
        this->template setElementEntry<2>(site.isiteFull, vec.getElement2());
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect3<floatT_compute> getElement(const gSiteStack &site) const {
        Vect3<floatT> ret(
                this->template getElementEntry<0>(site.isiteStackFull),
                this->template getElementEntry<1>(site.isiteStackFull),
                this->template getElementEntry<2>(site.isiteStackFull));
        return static_cast<Vect3<floatT_compute>>(ret);
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElement(const gSiteStack &site, const Vect3<floatT_compute> &vec) {
        this->template setElementEntry<0>(site.isiteStackFull, vec.getElement0());
        this->template setElementEntry<1>(site.isiteStackFull, vec.getElement1());
        this->template setElementEntry<2>(site.isiteStackFull, vec.getElement2());
    }

    __host__ __device__ inline void setEntriesComm(Vect3arrayAcc<floatT> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<0>(setIndex, src_acc.template getElementEntry<0>(getIndex));
        this->template setElementEntry<1>(setIndex, src_acc.template getElementEntry<1>(getIndex));
        this->template setElementEntry<2>(setIndex, src_acc.template getElementEntry<2>(getIndex));
    }

    template<Layout LatLayout, size_t HaloDepth>
    __host__ __device__ inline size_t getIndexComm(size_t isiteFull, size_t stack) const {
        gSiteStack site = GIndexer<LatLayout, HaloDepth>::getSiteStackFull(isiteFull, stack);
        return site.isiteStackFull;
    }


    template<Layout LatLayout, size_t HaloDepth,class floatT_compute=floatT>
    __host__ __device__ inline Vect3<floatT_compute> getElementComm(size_t isiteFull, size_t stack) const {
        gSiteStack site = GIndexer<LatLayout, HaloDepth>::getSiteStackFull(isiteFull, stack);
        return getElement<floatT_compute>(site);
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline void setElementComm(size_t isiteFull, __attribute__((unused)) size_t stack, const Vect3<floatT_compute> &vec) {
        gSiteStack site;
        site.isiteFull = isiteFull;
        site.isiteStackFull = isiteFull;
        setElement<floatT_compute>(site, vec);
    }

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect3<floatT_compute> operator()(const gSite &site) const {
        return this->getElement<floatT_compute>(site);
    };

    template<class floatT_compute=floatT>
    __host__ __device__ inline Vect3<floatT_compute> operator()(const gSiteStack &site) const {
        return this->getElement<floatT_compute>(site);
    };
};


template<class floatT, bool onDevice>
class Vect3array : public stackedArray<onDevice, COMPLEX(floatT), 3> {

public:
    friend class Vect3array<floatT, !onDevice>;

    explicit Vect3array(const size_t elems, std::string Vect3arrayName="Vect3array") :
            stackedArray<onDevice, COMPLEX(floatT), 3>(elems,Vect3arrayName) {}

    Vect3arrayAcc<floatT> getAccessor() const {
        COMPLEX(floatT) *const elements[3] = {
                this->template getEntry<COMPLEX(floatT) >(0),
                this->template getEntry<COMPLEX(floatT) >(1),
                this->template getEntry<COMPLEX(floatT) >(2)};
        return Vect3arrayAcc<floatT>(elements);
    }
};


#endif /* SU3ARRAY.HCU */



