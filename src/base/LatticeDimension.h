//
// Created by lukas on 22.09.17.
//

#ifndef JUST_INDEXER_LATTICEDIMENSION_H
#define JUST_INDEXER_LATTICEDIMENSION_H

#include <iostream>
#include <stdexcept>
#include "wrapper/gpu_wrapper.h"

class LatticeDimensions {

private :
    int c[4];

public :
    //! Copy constructor
     __host__ LatticeDimensions(const LatticeDimensions &lhs) {
        for (int i = 0; i < 4; i++) c[i] = lhs.c[i];
    }

    LatticeDimensions& operator=(const LatticeDimensions& a) = default;

    /*   __host__ constexpr LatticeDimensions& operator=(const LatticeDimensions& a){
           for (int i = 0; i < 4; i++) c[i] = a.c[i];
           return *this;
   }*/
    //! Default constructor, initializes to (0,0,0,0)
     __host__ LatticeDimensions() {
        c[0] = 0;
        c[1] = 0;
        c[2] = 0;
        c[3] = 0;
    }

    //! Construct from (x,y,z,t)
     __host__ LatticeDimensions(const int x, const int y, const int z, const int t) {
        c[0] = x;
        c[1] = y;
        c[2] = z;
        c[3] = t;
    }

    //! Construct from int* (also works with Parameter<int,4>)
     __host__ LatticeDimensions(const int *dim) {
        for (int i = 0; i < 4; i++)
            c[i] = dim[i];
    }

    //! Cast to int* (for usage in c-style MPI functions)
     __host__ operator int *() { return c; }

    //! same with const
     __host__ operator const int *() const { return c; };

    //! [] operator for member access
     __host__ int &operator[](int mu) { return c[mu]; };

    //! const [] operator for r/o member access
     __host__ const int &operator[](int mu) const { return c[mu]; };

    //! Component-wise multiplication, (x1*x2, y1*y2, z1*z2, t1*t2)
     __host__ LatticeDimensions operator*(const LatticeDimensions lhs) const {
        LatticeDimensions ret;
        for (int i = 0; i < 4; i++) ret.c[i] = c[i] * lhs.c[i];
        return ret;
    }

    //! Component-wise division, (x1/x2, y1/y2, z1/z2, t1/t2)
     __host__ LatticeDimensions operator/(const LatticeDimensions lhs) const {
        LatticeDimensions ret;
        for (int i = 0; i < 4; i++) ret.c[i] = c[i] / lhs.c[i];
        return ret;
    }

    //! modulo operation that returns coordinates within 0<=x<lx etc. (as long as the input is within -lx-1<x<2lx
     __host__ LatticeDimensions operator%(const LatticeDimensions box) const {
        LatticeDimensions result;
        for (int i = 0; i < 4; i++)
            result[i] = (box[i] + c[i]) % box[i];
        return result;
    }

    //! Component-wise addition, (x1+x2, y1+y2, z1+z2, t1+t2)
     __host__ LatticeDimensions operator+(const LatticeDimensions lhs) const {
        LatticeDimensions ret;
        for (int i = 0; i < 4; i++) ret.c[i] = c[i] + lhs.c[i];
        return ret;
    }

    //! scalar multiplication
     __host__ friend LatticeDimensions operator*(const int a, const LatticeDimensions &d) {
        return LatticeDimensions(a * d[0], a * d[1], a * d[2], a * d[3]);
    }

    //! Checks if all four dimension values are equal
     __host__ bool operator==(const LatticeDimensions &lhs) const {
        for (int i = 0; i < 4; i++)
            if (c[i] != lhs.c[i])
                return false;
        return true;
    }

    //! Logical not to "=="
     __host__ bool operator!=(const LatticeDimensions &lhs) const {
        return (!(lhs == *this));
    }

    //! Adds +/- 1 to dimension mu
    __host__ void mv(const int mu, const bool plus) {
        if ((mu < 0) || (mu >= 4))
            throw std::runtime_error(stdLogger.fatal("Wrong mu in LatticeDimensions");
        c[mu] += ((plus) ? (1) : (-1));
    }

    //! Formatted (debug) output
     __host__ friend std::ostream &operator<<(std::ostream &str,
                                    const LatticeDimensions &in) {
        str << "( ";
        for (int i = 0; i < 4; i++) str << in.c[i] << " ";
        str << ")";
        return str;
    }


    //! Return all four entries multiplied
     __host__ long mult() const {
        long res = 1;
        for (int i = 0; i < 4; i++) res *= (long) c[i];
        return res;
    }

     __host__ long summed() const {
        long res = 0;
        for (int i = 0; i < 4; i++) res += (long) c[i];
        return res;
    }


    //! Return if x,y,z,t are 0<=x<lx, 0<=y<ly ...
     __host__ bool inLimit(int x, int y, int z, int t) const {
        return inLimit(LatticeDimensions(x, y, z, t));
    }

     __host__ bool inLimit(const LatticeDimensions &in) const {
        for (int i = 0; i < 4; i++)
            if ((in[i] < 0) || (in[i] >= c[i])) return false;
        return true;
    }

    //! Return an offset matching given coordinates. With input x,y,z,t
    //! this returns x + y*LX + z*LX*LY + t*LX*LY*LZ
     __host__ size_t offset(const LatticeDimensions &in) const {
        size_t ret = in[0] + c[0] * in[1] + c[0] * c[1] * in[2] + c[0] * c[1] * c[2] * in[3];
        return ret;
    }

    //! Return the lowest entry
     __host__ int lowest_value() const {
        int res = c[1];
        for (int i = 0; i < 4; i++)
            if (c[i] < res)res = c[i];
        return res;
    }

    //! Return the lowest entry
    __host__ int lowest_spatial_value() const {
        int res = c[1];
        for (int i = 0; i < 3; i++)
            if (c[i] < res)res = c[i];
        return res;
    }

};

#endif //JUST_INDEXER_LATTICEDIMENSION_H
