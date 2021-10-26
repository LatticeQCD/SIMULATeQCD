#ifndef GSU3ARRAY_HCU
#define GSU3ARRAY_HCU


#include "gcomplex.h"
#include "gsu3.h"
#include "gvect3.h"
#include "../indexer/BulkIndexer.h"
#include "stackedArray.h"

#include "gaugeAccessor.h"

/**
 * This class allows storage of an array of gpu_su3 instances
 * in a format that allows coalesced access to the elements if
 * each thread reads on element.
 *
 * The dynamic memory in this class is managed by the memory management.
 *
 * Access to the elements happens via a gaugeAccessor object
 * which can easily be sent to the device.
 */
template<class floatT, bool onDevice, CompressionType comp = R18>
class GSU3array : public stackedArray<onDevice, GCOMPLEX(floatT),EntryCount<comp>::count> {

public:
    friend class GSU3array<floatT, !onDevice>;

    explicit GSU3array(const size_t elems, std::string GSU3arrayName="GSU3array") :
            stackedArray<onDevice, GCOMPLEX(floatT), EntryCount<comp>::count>(elems,GSU3arrayName) {}

    template<CompressionType compression=comp>
    gaugeAccessor<floatT, compression> getAccessor() const {

        GCOMPLEX(floatT) *elements[EntryCount<comp>::count];

        for (int i = 0; i < EntryCount<comp>::count; i++){
            elements[i] = this->template getEntry<GCOMPLEX(floatT)>(i);
        }

        return gaugeAccessor<floatT, compression>(elements);
    }
};

#endif /* GSU3ARRAY.HCU */

