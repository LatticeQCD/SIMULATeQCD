#ifndef SU3ARRAY_HCU
#define SU3ARRAY_HCU


#include "complex.h"
#include "su3.h"
#include "vect3.h"
#include "../indexer/bulkIndexer.h"
#include "stackedArray.h"

#include "su3Accessor.h"

/**
 * This class allows storage of an array of gpu_su3 instances
 * in a format that allows coalesced access to the elements if
 * each thread reads on element.
 *
 * The dynamic memory in this class is managed by the memory management.
 *
 * Access to the elements happens via a SU3Accessor object
 * which can easily be sent to the device.
 */
template<class floatT, bool onDevice, CompressionType comp = R18>
class SU3array : public stackedArray<onDevice, COMPLEX(floatT),EntryCount<comp>::count> {

public:
    friend class SU3array<floatT, !onDevice>;

    explicit SU3array(const size_t elems, std::string SU3arrayName="SU3array") :
            stackedArray<onDevice, COMPLEX(floatT), EntryCount<comp>::count>(elems,SU3arrayName) {}

    template<CompressionType compression=comp>
    SU3Accessor<floatT, compression> getAccessor() const {

        COMPLEX(floatT) *elements[EntryCount<comp>::count];

        for (int i = 0; i < EntryCount<comp>::count; i++){
            elements[i] = this->template getEntry<COMPLEX(floatT)>(i);
        }

        return SU3Accessor<floatT, compression>(elements);
    }
};

#endif /* SU3ARRAY.HCU */

