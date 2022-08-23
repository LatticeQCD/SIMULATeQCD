/* 
 * generalAccessor.h                                                               
 * 
 * Lukas Mazur, 12 Dec 2018
 * 
 * The GeneralAccessor class makes it easier to access elements of arrays on the GPU by performing pointer
 * arithmetic automatically. This is necessary whenever you want to make an array of values that can themselves be
 * represented as arrays; for example an array of SU(3) matrices can be thought of as an array of floatT arrays
 * of length 9.
 *
 * The fundamental object of the GeneralAccessor class is the _elements array, which is an array of pointer arrays.
 * _elements points to an array of objects indexed by the integer idx. Each object is represented by multiple numbers;
 * these numbers are indexed by elem.
 *
 */

#ifndef GENERALACCESSOR_H
#define GENERALACCESSOR_H

#include "../../define.h"


/// The template parameter object_memory gives the data type of the values pointed to by the pointers.
template<class object_memory, size_t Nentries>
class GeneralAccessor {
protected:

    object_memory *_elements[Nentries];

public:

    template<int elem>
    HOST_DEVICE inline object_memory getElementEntry(const size_t idx) const {
        return (_elements[elem][idx]);
    }

    template<int elem>
    HOST_DEVICE inline void setElementEntry(const size_t idx, object_memory entry) {
        _elements[elem][idx] = static_cast<object_memory>(entry);
    }

    explicit GeneralAccessor(object_memory *const elements[]) {
        for (size_t i = 0; i < Nentries; i++) {
            _elements[i] = elements[i];
        }
    }

    /// Constructor for one memory chunk, where all entries are separated by object_count
   HOST_DEVICE explicit GeneralAccessor(object_memory *elementsBase, size_t object_count) {
        for (size_t i = 0; i < Nentries; i++) {
            _elements[i] = elementsBase + i * object_count;
        }
    }

    explicit GeneralAccessor() {
        for (size_t i = 0; i < Nentries; i++) {
            _elements[i] = nullptr;
        }
    }
};

#endif //GENERALACCESSOR_H
