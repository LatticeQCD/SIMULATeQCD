//
// Created by Lukas Mazur on 10.12.18.
//

#ifndef STACKEDARRAY_H
#define STACKEDARRAY_H

#include "../memoryManagement.h"

//TODO Use the new class
template<bool onDevice, class entryType, int entryCount>
class stackedArray {

private:
    size_t _arraySize;
    gMemoryPtr<onDevice> _memory;

public:
    //! constructor
    explicit stackedArray(size_t arraySize, std::string arrayName="STACKED") :
            _arraySize(arraySize),
            _memory(MemoryManagement::getMemAt<onDevice>(arrayName))
    {
        _memory->template adjustSize<entryType>(arraySize*entryCount);
    }

    //! copy constructor
    stackedArray(stackedArray<onDevice,entryType,entryCount>&) = default;

    //! copy assignment
    stackedArray<onDevice,entryType,entryCount>& operator=(stackedArray<onDevice,entryType,entryCount>&) = default;

    //! move constructor
    stackedArray(stackedArray<onDevice,entryType,entryCount>&& source) noexcept :
            _arraySize(source._arraySize),
            _memory(std::move(source._memory))
    {
        source._arraySize = 0;
    }

    //! move assignment
    stackedArray<onDevice,entryType,entryCount>& operator=(stackedArray<onDevice,entryType,entryCount>&&) = delete;

    //! destructor
    ~stackedArray() = default;

    friend class stackedArray<!onDevice, entryType, entryCount>;

    template<bool onDeviceSrc>
    void copyFrom(const stackedArray<onDeviceSrc, entryType, entryCount> &src) {
        _memory->template copyFrom<onDeviceSrc>(src._memory, src.getSize());
    }

    template<bool onDeviceSrc>
    void copyFromPartial(const stackedArray<onDeviceSrc, entryType, entryCount> &src, const size_t Nelems,
                         const size_t offsetSelf = 0, const size_t offsetSrc = 0) {

        for (int i = 0; i < entryCount; i++) {
            _memory->template copyFrom<onDeviceSrc>(src._memory, Nelems*sizeof(entryType),
                                                    i*getSizeOneEntry() + offsetSelf*sizeof(entryType),
                                                    i*src.getSizeOneEntry() + offsetSrc*sizeof(entryType));
        }
    }

    template<bool onDeviceDest>
    void copyTo(stackedArray<onDeviceDest, entryType, entryCount> &dest) {
            dest._memory->template copyFrom<onDevice>(_memory, getSize());
    }

    void swap(stackedArray<onDevice, entryType, entryCount> &in) {
        if (getSize() == in.getSize()) {
            _memory->swap(in._memory);
        } else throw std::runtime_error(stdLogger.fatal("stackedArray.h: swap not allowed when Array size is different!"));
    }

    size_t getSize() const {
        return sizeof(entryType) * entryCount * _arraySize;
    }

    size_t getSizeOneEntry() const {
        return sizeof(entryType) * _arraySize;
    }

    template<class returnType>
    returnType* getEntry(int idx) const {
        return _memory->template getPointer<returnType>()+idx*_arraySize;
    }
};

#endif //STACKEDARRAY_H
