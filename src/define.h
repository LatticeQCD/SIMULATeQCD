/*
 * define.h
 *
 * Lukas Mazur, 10 Oct 2017
 *
 */

#pragma once
#include "base/IO/logging.h"
#include "explicit_instantiation_macros.h"

#define COMPILE_WITH_MPI
//! define functions as 'void bla() EMPTY_IF_SCALAR;' in order to give a
//! standard implementation of doing nothing in the case of scalar code
#ifdef COMPILE_WITH_MPI
#define EMPTY_IF_SCALAR
#define RET0_IF_SCALAR
#define RETa_IF_SCALAR
#else
#define EMPTY_IF_SCALAR {}
#define RET0_IF_SCALAR { return 0; }
#define RETa_IF_SCALAR { return a; }
#endif



#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)


enum Layout {
    All, Even, Odd
};
enum HaloSegment {
    X, Y, Z, T,                /// Seqment type 1
    XY, XZ, XT, YZ, YT, ZT,    /// Seqment type 2
    XYZ, XYT, XZT, YZT,        /// Seqment type 3
    XYZT                       /// Seqment type 4
};

enum HaloType {
    Hyperplane = 1,
    Plane = 2,
    Stripe = 4,
    Corner = 8,
    AllTypes = Hyperplane | Plane | Stripe | Corner,
    NoCorner = Hyperplane | Plane | Stripe
};

enum CommSync {
    COMM_START = 16,
    COMM_FINISH = 32,
    COMM_BOTH = COMM_START | COMM_FINISH,
    COMM_OVERLAP = 64
};



const HaloSegment AllHaloSegments[] = {HaloSegment::X, HaloSegment::Y, HaloSegment::Z, HaloSegment::T,
                                       HaloSegment::XY, HaloSegment::XZ, HaloSegment::XT,
                                       HaloSegment::YZ, HaloSegment::YT, HaloSegment::ZT,
                                       HaloSegment::XYZ, HaloSegment::XYT, HaloSegment::XZT, HaloSegment::YZT,
                                       HaloSegment::XYZT};
const HaloSegment HaloHypPlanes[] = {HaloSegment::X, HaloSegment::Y, HaloSegment::Z, HaloSegment::T};
const HaloSegment HaloPlanes[] = {HaloSegment::XY, HaloSegment::XZ, HaloSegment::XT,
                                  HaloSegment::YZ, HaloSegment::YT, HaloSegment::ZT};
const HaloSegment HaloStripes[] = {HaloSegment::XYZ, HaloSegment::XYT, HaloSegment::XZT, HaloSegment::YZT};
const HaloSegment HaloCorners[] = {HaloSegment::XYZT};

inline int HaloSegmentDirections(HaloSegment seg) {
    if (seg == X || seg == Y || seg == Z || seg == T) return 1;
    else if (seg == XY || seg == XZ || seg == XT || seg == YZ || seg == YT || seg == ZT) return 2;
    else if (seg == XYZ || seg == XYT || seg == XZT || seg == YZT) return 4;
    else return 8;
}

inline int haloSegmentCoordToIndex(const HaloSegment halseg, const size_t direction,const int leftRight) {
        int index = 0;
        if (halseg < 4){
            index = halseg * 2 + leftRight;
        }
        else if (halseg < 10) {
            size_t fakt = 2*leftRight + direction;
            fakt = fakt < 2 ? fakt : 5- fakt;
            index = 8+((size_t) halseg - 4) * 4 + fakt;
        }
        else if (halseg < 14){
            size_t fakt = 4*leftRight + direction;
            fakt = fakt < 4 ? fakt : 11 - fakt;
            index = 8+24+((size_t) halseg - 10) * 8 + fakt;
        }
        else {
            size_t fakt = 8*leftRight + direction;
            fakt = fakt < 8 ? fakt : 23 - fakt;
            index =8+24+32+((size_t) halseg - 14) * 16 + fakt;
        }
        return index ;
}


template<int N>
class HSegSelector {
public:

    constexpr HaloType haloType() {
        if (N < 8) return Hyperplane;
        else if (N < 32) return Plane;
        else if (N < 64) return Stripe;
        else return Corner;

    }

    constexpr HaloSegment haloSeg() {
        if (haloType() == Hyperplane) return (HaloSegment) (N / 2);
        else if (haloType() == Plane) return (HaloSegment) (4 + (N - 8) / 4);
        else if (haloType() == Stripe) return (HaloSegment) (10 + (N - 32) / 8);
        else return (HaloSegment) XYZT; // Corner
    }

    constexpr int subIndex() {
        if (haloType() == Hyperplane) return N % 2;
        else if (haloType() == Plane) return (N - 8) % 4;
        else if (haloType() == Stripe) return (N - 32) % 8;
        else return (N - 64); // Corner
    }

    constexpr int dir() {
        if (haloType() == Hyperplane) return 0;
        else if (haloType() == Plane) return ((subIndex() + 1) % 4) / 2;
        else if (haloType() == Stripe) return (subIndex() < 4 ? subIndex() : 3 - (subIndex() % 4));
        else return subIndex() < 8 ? subIndex() : 7 - (subIndex() % 8); // Corner


    }

    constexpr int leftRight() {
        if (haloType() == Hyperplane) return N % 2;
        else if (haloType() == Plane) return ((N - 8) % 4) >= 2;
        else if (haloType() == Stripe) return ((N - 32) % 8) >= 4;
        else return (N - 64) >= 8; // Corner
    }
};

namespace CoutColors {
    const std::string red("\033[0;31m");
    const std::string redBold("\033[1;31m");
    const std::string green("\033[0;32m");
    const std::string greenBold("\033[1;32m");
    const std::string yellow("\033[0;33m");
    const std::string yellowBold("\033[1;33m");
    const std::string cyan("\033[0;36m");
    const std::string cyanBold("\033[1;36m");
    const std::string magenta("\033[0;35m");
    const std::string magentaBold("\033[1;35m");
    const std::string reset("\033[0m");
}


