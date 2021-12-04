#ifndef FILEIO_MISC
#define FILEIO_MISC

#include "../../define.h"
#include <endian.h>

enum Endianness {
    ENDIAN_LITTLE, ENDIAN_BIG, ENDIAN_AUTO
};

//! Swaps bytes for little/big endian conversion
inline void Byte_swap(char *b, int n) {
    int i = 0;
    int j = n - 1;
    while (i < j) {
        std::swap(b[i], b[j]);
        i++, j--;
    }
}

//! convenience class to insert an integer with zero padding into a stream
//! example: stdLogger.debug(Pad0(2,mu));
//! will print mu with width 2 and padded with zeros
class Pad0 {
private:
    const int width;
    const int val;
public:
    Pad0(int _width, int _val)
            : width(_width), val(_val) {};

    friend std::ostream &operator<<(std::ostream &os, const Pad0 &p) {
        const int w = os.width();
        if (w > p.width)
            os << std::setw(w - p.width) << "";
        const char prev = os.fill('0');
        os << std::setw(p.width) << p.val;
        os.fill(prev);
        return os;
    }
};

//! Swaps bytes for little/big endian conversion
template<class T>
inline void Byte_swap(T &f) {
    Byte_swap((char *) &f, sizeof(T));
}

inline Endianness get_endianness(bool swap) {
    Endianness result[2] = {ENDIAN_LITTLE, ENDIAN_BIG};
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return result[swap];
#elif __BYTE_ORDER == __BIG_ENDIAN
    return result[!swap];
#endif
}

//! decide if we need to change endianness in order to obtain endianness target
inline bool switch_endianness(Endianness target) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return target == ENDIAN_BIG;
#elif __BYTE_ORDER == __BIG_ENDIAN
    return target == ENDIAN_LITTLE;
#endif
}


#endif

