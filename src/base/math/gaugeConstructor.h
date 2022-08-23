//
// Created by Lukas Mazur on 17.12.18.
//

#ifndef GAUGECONSTRUCTOR_H
#define GAUGECONSTRUCTOR_H

#include "generalAccessor.h"
#include "gsu3.h"
#include "../../modules/HISQ/staggeredPhases.h"

template <CompressionType comp>
class EntryCount; // implement this one as well, if you want to have a default...

template <> class EntryCount<R18> { public: static const int count = 9; };
template <> class EntryCount<R14> { public: static const int count = 7; };
template <> class EntryCount<U3R14> { public: static const int count = 7; };
template <> class EntryCount<R12> { public: static const int count = 6; };
template <> class EntryCount<STAGG_R12> { public: static const int count = 6; };




enum elemIndices {
    e00, e01, e02, e10, e11, e12, e20, e21, e22
};

template<class floatT_memory, CompressionType comp>
struct GaugeConstructor : public GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<comp>::count> {
    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<comp>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<comp>::count >(elements) {
    }
    /// Constructor for one memory chunk, where all entries are separated by object_count
    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<comp>::count >(elementsBase, object_count){
    }
    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<comp>::count >(){ }
};

template<class floatT_memory>
struct GaugeConstructor<floatT_memory, R18> : public GeneralAccessor<GCOMPLEX(
        floatT_memory), EntryCount<R18>::count> {

    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<R18>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R18>::count>(elements) {}

    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R18>::count >(elementsBase, object_count) {}

    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R18>::count>() {}

    HOST_DEVICE inline void setEntriesComm(GaugeConstructor<floatT_memory,R18> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<e00>(setIndex, src_acc.template getElementEntry<e00>(getIndex));
        this->template setElementEntry<e01>(setIndex, src_acc.template getElementEntry<e01>(getIndex));
        this->template setElementEntry<e02>(setIndex, src_acc.template getElementEntry<e02>(getIndex));
        this->template setElementEntry<e10>(setIndex, src_acc.template getElementEntry<e10>(getIndex));
        this->template setElementEntry<e11>(setIndex, src_acc.template getElementEntry<e11>(getIndex));
        this->template setElementEntry<e12>(setIndex, src_acc.template getElementEntry<e12>(getIndex));
        this->template setElementEntry<e20>(setIndex, src_acc.template getElementEntry<e20>(getIndex));
        this->template setElementEntry<e21>(setIndex, src_acc.template getElementEntry<e21>(getIndex));
        this->template setElementEntry<e22>(setIndex, src_acc.template getElementEntry<e22>(getIndex));
    }

    HOST_DEVICE inline void construct(const gSiteMu& idx, const GSU3<floatT_memory> &mat) {
        this->template setElementEntry<e00>(idx.indexMuFull, mat.getLink00());
        this->template setElementEntry<e01>(idx.indexMuFull, mat.getLink01());
        this->template setElementEntry<e02>(idx.indexMuFull, mat.getLink02());
        this->template setElementEntry<e10>(idx.indexMuFull, mat.getLink10());
        this->template setElementEntry<e11>(idx.indexMuFull, mat.getLink11());
        this->template setElementEntry<e12>(idx.indexMuFull, mat.getLink12());
        this->template setElementEntry<e20>(idx.indexMuFull, mat.getLink20());
        this->template setElementEntry<e21>(idx.indexMuFull, mat.getLink21());
        this->template setElementEntry<e22>(idx.indexMuFull, mat.getLink22());
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstruct(const gSiteMu& idx) const {
        GSU3<floatT_memory> ret(
                this->template getElementEntry<e00>(idx.indexMuFull),
                this->template getElementEntry<e01>(idx.indexMuFull),
                this->template getElementEntry<e02>(idx.indexMuFull),
                this->template getElementEntry<e10>(idx.indexMuFull),
                this->template getElementEntry<e11>(idx.indexMuFull),
                this->template getElementEntry<e12>(idx.indexMuFull),
                this->template getElementEntry<e20>(idx.indexMuFull),
                this->template getElementEntry<e21>(idx.indexMuFull),
                this->template getElementEntry<e22>(idx.indexMuFull));
        return ret;
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstructDagger(const gSiteMu& idx) const {
        return GSU3<floatT_memory>(conj(this->template getElementEntry<e00>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e10>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e20>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e01>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e11>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e21>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e02>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e12>(idx.indexMuFull)),
                                   conj(this->template getElementEntry<e22>(idx.indexMuFull)));
    }
};

template<class floatT_memory>
struct GaugeConstructor<floatT_memory, U3R14> : public GeneralAccessor<GCOMPLEX(
        floatT_memory), EntryCount<U3R14>::count > {

    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<U3R14>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<U3R14>::count >(elements) {
    }
    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<U3R14>::count >(elementsBase, object_count){
    }
    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<U3R14>::count >(){ }


    HOST_DEVICE inline void setEntriesComm(GaugeConstructor<floatT_memory,U3R14> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<e00>(setIndex, src_acc.template getElementEntry<e00>(getIndex));
        this->template setElementEntry<e01>(setIndex, src_acc.template getElementEntry<e01>(getIndex));
        this->template setElementEntry<e02>(setIndex, src_acc.template getElementEntry<e02>(getIndex));
        this->template setElementEntry<e10>(setIndex, src_acc.template getElementEntry<e10>(getIndex));
        this->template setElementEntry<e11>(setIndex, src_acc.template getElementEntry<e11>(getIndex));
        this->template setElementEntry<e12>(setIndex, src_acc.template getElementEntry<e12>(getIndex));
        this->template setElementEntry<e20>(setIndex, src_acc.template getElementEntry<e20>(getIndex));
    }

    HOST_DEVICE inline void construct(const gSiteMu& idx, const GSU3<floatT_memory> &mat) {
        this->template setElementEntry<e00>(idx.indexMuFull, mat.getLink00());
        this->template setElementEntry<e01>(idx.indexMuFull, mat.getLink01());
        this->template setElementEntry<e02>(idx.indexMuFull, mat.getLink02());
        this->template setElementEntry<e10>(idx.indexMuFull, mat.getLink10());
        this->template setElementEntry<e11>(idx.indexMuFull, mat.getLink11());
        this->template setElementEntry<e12>(idx.indexMuFull, mat.getLink12());
        this->template setElementEntry<e20>(idx.indexMuFull, det(mat));
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstruct(const gSiteMu& idx) const {
        GSU3<floatT_memory> ret(
                this->template getElementEntry<e00>(idx.indexMuFull),
                this->template getElementEntry<e01>(idx.indexMuFull),
                this->template getElementEntry<e02>(idx.indexMuFull),
                this->template getElementEntry<e10>(idx.indexMuFull),
                this->template getElementEntry<e11>(idx.indexMuFull),
                this->template getElementEntry<e12>(idx.indexMuFull),
                (floatT_memory)1.0, (floatT_memory)1.0, (floatT_memory)1.0);
        GCOMPLEX(floatT_memory) phase = this->template getElementEntry<e20>(idx.indexMuFull);
        ret.u3reconstruct(phase);

        return ret;
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstructDagger(const gSiteMu& idx) const {
        GSU3<floatT_memory> tmp = GSU3<floatT_memory>(
                conj(this->template getElementEntry<e00>(idx.indexMuFull)),
                conj(this->template getElementEntry<e10>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e01>(idx.indexMuFull)),
                conj(this->template getElementEntry<e11>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e02>(idx.indexMuFull)),
                conj(this->template getElementEntry<e12>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.));

        GCOMPLEX(floatT_memory) phase = this->template getElementEntry<e20>(idx.indexMuFull);
        tmp.u3reconstructDagger(conj(phase));
        return tmp;
    }
};

template<class floatT_memory>
struct GaugeConstructor<floatT_memory, R14> : public GeneralAccessor<GCOMPLEX(
        floatT_memory), EntryCount<R14>::count > {

    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<R14>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R14>::count >(elements) {
    }
    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R14>::count >(elementsBase, object_count) {
    }
    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R14>::count>() { }

    HOST_DEVICE inline void setEntriesComm(GaugeConstructor<floatT_memory,R14> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<e00>(setIndex, src_acc.template getElementEntry<e00>(getIndex));
        this->template setElementEntry<e01>(setIndex, src_acc.template getElementEntry<e01>(getIndex));
        this->template setElementEntry<e02>(setIndex, src_acc.template getElementEntry<e02>(getIndex));
        this->template setElementEntry<e10>(setIndex, src_acc.template getElementEntry<e10>(getIndex));
        this->template setElementEntry<e11>(setIndex, src_acc.template getElementEntry<e11>(getIndex));
        this->template setElementEntry<e12>(setIndex, src_acc.template getElementEntry<e12>(getIndex));
        this->template setElementEntry<e20>(setIndex, src_acc.template getElementEntry<e20>(getIndex));
    }

    HOST_DEVICE inline void construct(const gSiteMu& idx, const GSU3<floatT_memory> &mat) {
        this->template setElementEntry<e00>(idx.indexMuFull, mat.getLink00());
        this->template setElementEntry<e01>(idx.indexMuFull, mat.getLink01());
        this->template setElementEntry<e02>(idx.indexMuFull, mat.getLink02());
        this->template setElementEntry<e10>(idx.indexMuFull, mat.getLink10());
        this->template setElementEntry<e11>(idx.indexMuFull, mat.getLink11());
        this->template setElementEntry<e12>(idx.indexMuFull, mat.getLink12());
        this->template setElementEntry<e20>(idx.indexMuFull, det(mat));
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstruct(const gSiteMu& idx) const {
        GSU3<floatT_memory> ret(
                this->template getElementEntry<e00>(idx.indexMuFull),
                this->template getElementEntry<e01>(idx.indexMuFull),
                this->template getElementEntry<e02>(idx.indexMuFull),
                this->template getElementEntry<e10>(idx.indexMuFull),
                this->template getElementEntry<e11>(idx.indexMuFull),
                this->template getElementEntry<e12>(idx.indexMuFull),
                (floatT_memory)1, (floatT_memory)1, (floatT_memory)1);
        GCOMPLEX(floatT_memory) phase = this->template getElementEntry<e20>(idx.indexMuFull);
        ret.reconstruct14(phase);

        return ret;
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstructDagger(const gSiteMu& idx) const {
        GSU3<floatT_memory> tmp = GSU3<floatT_memory>(
                conj(this->template getElementEntry<e00>(idx.indexMuFull)),
                conj(this->template getElementEntry<e10>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e01>(idx.indexMuFull)),
                conj(this->template getElementEntry<e11>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e02>(idx.indexMuFull)),
                conj(this->template getElementEntry<e12>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.));

        GCOMPLEX(floatT_memory) phase = this->template getElementEntry<e20>(idx.indexMuFull);
        tmp.reconstruct14Dagger(conj(phase));
        return tmp;
    }
};

template<class floatT_memory>
struct GaugeConstructor<floatT_memory, R12> : public GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R12>::count> {

    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<R12>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R12>::count >(elements) {
    }
    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R12>::count>(elementsBase, object_count) {
    }
    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<R12>::count>() { }

    HOST_DEVICE inline void setEntriesComm(GaugeConstructor<floatT_memory, R12> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<e00>(setIndex, src_acc.template getElementEntry<e00>(getIndex));
        this->template setElementEntry<e01>(setIndex, src_acc.template getElementEntry<e01>(getIndex));
        this->template setElementEntry<e02>(setIndex, src_acc.template getElementEntry<e02>(getIndex));
        this->template setElementEntry<e10>(setIndex, src_acc.template getElementEntry<e10>(getIndex));
        this->template setElementEntry<e11>(setIndex, src_acc.template getElementEntry<e11>(getIndex));
        this->template setElementEntry<e12>(setIndex, src_acc.template getElementEntry<e12>(getIndex));
    }

    HOST_DEVICE inline void construct(const gSiteMu& idx, const GSU3<floatT_memory> &mat) {
        this->template setElementEntry<e00>(idx.indexMuFull, mat.getLink00());
        this->template setElementEntry<e01>(idx.indexMuFull, mat.getLink01());
        this->template setElementEntry<e02>(idx.indexMuFull, mat.getLink02());
        this->template setElementEntry<e10>(idx.indexMuFull, mat.getLink10());
        this->template setElementEntry<e11>(idx.indexMuFull, mat.getLink11());
        this->template setElementEntry<e12>(idx.indexMuFull, mat.getLink12());
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstruct(const gSiteMu& idx) const {
        GSU3<floatT_memory> ret(
                this->template getElementEntry<e00>(idx.indexMuFull),
                this->template getElementEntry<e01>(idx.indexMuFull),
                this->template getElementEntry<e02>(idx.indexMuFull),
                this->template getElementEntry<e10>(idx.indexMuFull),
                this->template getElementEntry<e11>(idx.indexMuFull),
                this->template getElementEntry<e12>(idx.indexMuFull),
                (floatT_memory)1, (floatT_memory)1, (floatT_memory)1);
        ret.su3reconstruct12();

        return ret;
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstructDagger(const gSiteMu& idx) const {
        GSU3<floatT_memory> tmp = GSU3<floatT_memory>(
                conj(this->template getElementEntry<e00>(idx.indexMuFull)),
                conj(this->template getElementEntry<e10>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e01>(idx.indexMuFull)),
                conj(this->template getElementEntry<e11>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e02>(idx.indexMuFull)),
                conj(this->template getElementEntry<e12>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.));

        tmp.su3reconstruct12Dagger();
        return tmp;
    }
};


template<class floatT_memory>
struct GaugeConstructor<floatT_memory, STAGG_R12> : public GeneralAccessor<GCOMPLEX(
        floatT_memory), EntryCount<STAGG_R12>::count> {
private:
    calcStaggeredPhase staggPhase;
    calcStaggeredBoundary staggBound;

public:

    explicit GaugeConstructor(GCOMPLEX(floatT_memory) *const elements[EntryCount<STAGG_R12>::count])
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<STAGG_R12>::count>(elements) {
        throw std::runtime_error(stdLogger.fatal("STAGG_R12 should not be used at the moment"));
    }
    HOST_DEVICE explicit GaugeConstructor(GCOMPLEX(floatT_memory) *elementsBase, size_t object_count)
            : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<STAGG_R12>::count >(elementsBase, object_count) {
    }
    explicit GaugeConstructor() : GeneralAccessor<GCOMPLEX(floatT_memory), EntryCount<STAGG_R12>::count>() { }

    HOST_DEVICE inline void setEntriesComm(GaugeConstructor<floatT_memory,STAGG_R12> &src_acc,
                                                   size_t setIndex, size_t getIndex) {
        this->template setElementEntry<e00>(setIndex, src_acc.template getElementEntry<e00>(getIndex));
        this->template setElementEntry<e01>(setIndex, src_acc.template getElementEntry<e01>(getIndex));
        this->template setElementEntry<e02>(setIndex, src_acc.template getElementEntry<e02>(getIndex));
        this->template setElementEntry<e10>(setIndex, src_acc.template getElementEntry<e10>(getIndex));
        this->template setElementEntry<e11>(setIndex, src_acc.template getElementEntry<e11>(getIndex));
        this->template setElementEntry<e12>(setIndex, src_acc.template getElementEntry<e12>(getIndex));
    }

    HOST_DEVICE inline void construct(const gSiteMu& idx, const GSU3<floatT_memory> &mat) {
        this->template setElementEntry<e00>(idx.indexMuFull, mat.getLink00());
        this->template setElementEntry<e01>(idx.indexMuFull, mat.getLink01());
        this->template setElementEntry<e02>(idx.indexMuFull, mat.getLink02());
        this->template setElementEntry<e10>(idx.indexMuFull, mat.getLink10());
        this->template setElementEntry<e11>(idx.indexMuFull, mat.getLink11());
        this->template setElementEntry<e12>(idx.indexMuFull, mat.getLink12());
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstruct(const gSiteMu& idx) const {
        GSU3<floatT_memory> ret(
                this->template getElementEntry<e00>(idx.indexMuFull),
                this->template getElementEntry<e01>(idx.indexMuFull),
                this->template getElementEntry<e02>(idx.indexMuFull),
                this->template getElementEntry<e10>(idx.indexMuFull),
                this->template getElementEntry<e11>(idx.indexMuFull),
                this->template getElementEntry<e12>(idx.indexMuFull),
                (floatT_memory)1, (floatT_memory)1, (floatT_memory)1);

        int phase = staggBound(idx) * staggPhase(idx);
        ret.u3reconstruct(phase);
        return ret;
    }

    HOST_DEVICE inline GSU3<floatT_memory> reconstructDagger(const gSiteMu& idx) const {
        GSU3<floatT_memory> tmp = GSU3<floatT_memory>(
                conj(this->template getElementEntry<e00>(idx.indexMuFull)),
                conj(this->template getElementEntry<e10>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e01>(idx.indexMuFull)),
                conj(this->template getElementEntry<e11>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.),
                conj(this->template getElementEntry<e02>(idx.indexMuFull)),
                conj(this->template getElementEntry<e12>(idx.indexMuFull)),
                GCOMPLEX(floatT_memory)(0., 0.));

        int phase = staggBound(idx) * staggPhase(idx);
        tmp.u3reconstructDagger(phase);
        return tmp;
    }
};

#endif //GAUGECONSTRUCTOR_H
