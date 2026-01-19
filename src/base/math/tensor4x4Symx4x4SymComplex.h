//
// Created by Jonas Winter on 25.08.2025
//

#pragma once
#include "../../define.h"
#include "indices4x4Sym.h"
#include "matrix4x4SymComplex.h"

enum PrintComplex {
    complex_both,
    complex_real,
    complex_imag
};

// can't define this up here
// template<class floatT>
// typedef void (* twoIndexedVoid)(Tensor4x4Symx4x4SymComplex<floatT> tensor, int first, int second);

// template<class floatT>
// void twoIndexWrapper(Tensor4x4Symx4x4SymComplex<floatT> tensor, twoIndexedVoid func) {
//     for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
//         for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
//             func(tensor, firstIndexPair, secondIndexPair);
//         }
//     }
// }


// 4x4x4x4 tensor with complex-valued floatT components
// symmetric under switching the first two indices and under switching the last two indices
// instead of 4*4*4*4=256 components, "only" 10*10=100 components
template<class floatT>
struct Tensor4x4Symx4x4SymComplex {

    COMPLEX(floatT) elems[10][10];

    // TODO: Difference between ordering of __device__ and __host__?
    __device__ __host__ Tensor4x4Symx4x4SymComplex() {
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] = 0.0;
            }
        }
    }

    __device__ __host__ Tensor4x4Symx4x4SymComplex(floatT value) {
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] = value;
            }
        }
    }

    __device__ __host__ Tensor4x4Symx4x4SymComplex(Matrix4x4SymComplex<floatT> firstEMT, Matrix4x4SymComplex<floatT> secondEMT) {
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] = firstEMT.elems[firstIndexPair] * secondEMT.elems[secondIndexPair];
            }
        }
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT> getSecondMatrix4x4SymComplex(int firstIndexPair) {
        Matrix4x4SymComplex<floatT> secondMatrix = Matrix4x4SymComplex<floatT>();
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            secondMatrix(secondIndexPair, elems[firstIndexPair][secondIndexPair]);
        }
        return secondMatrix;
    }

    __device__ __host__ inline void setSecondMatrix4x4SymComplex(int firstIndexPair, Matrix4x4SymComplex<floatT> secondMatrix) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            elems[firstIndexPair][secondIndexPair] = secondMatrix(secondIndexPair);
        }
    }

    __device__ __host__ inline COMPLEX(floatT) operator()(int firstIndexPair, int secondIndexPair) const {
        return elems[firstIndexPair][secondIndexPair];
    }

    __device__ __host__ inline void operator()(int firstIndexPair, int secondIndexPair, COMPLEX(floatT) value) {
        elems[firstIndexPair][secondIndexPair] = value;
    }

    __device__ __host__ inline COMPLEX(floatT) operator()(int mu, int nu, int rho, int sigma) const {
        int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
        int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
        return elems[firstIndexPair][secondIndexPair];
    }

    __device__ __host__ inline void operator()(int mu, int nu, int rho, int sigma, COMPLEX(floatT) value) {
        int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
        int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
        elems[firstIndexPair][secondIndexPair] = value;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator+=(const Tensor4x4Symx4x4SymComplex<floatT> &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] += y.elems[firstIndexPair][secondIndexPair];
            }
        }
        return *this;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator-=(const Tensor4x4Symx4x4SymComplex<floatT> &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] -= y.elems[firstIndexPair][secondIndexPair];
            }
        }
        return *this;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator*=(const floatT &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] *= y;
            }
        }
        return *this;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator*=(const COMPLEX(floatT) &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] *= y;
            }
        }
        return *this;
    }

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator/=(const floatT &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] /= y;
            }
        }
        return *this;
    }

    __device__ __host__ friend Tensor4x4Symx4x4SymComplex<floatT> operator+(
        const Tensor4x4Symx4x4SymComplex<floatT> &left,
        const Tensor4x4Symx4x4SymComplex<floatT> &right
    ) {
        Tensor4x4Symx4x4SymComplex<floatT> result;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result(firstIndexPair, secondIndexPair, left(firstIndexPair, secondIndexPair) + right(firstIndexPair, secondIndexPair));
            }
        }
        return result;
    }

    __device__ __host__ friend Tensor4x4Symx4x4SymComplex<floatT> operator-(
        const Tensor4x4Symx4x4SymComplex<floatT> &left,
        const Tensor4x4Symx4x4SymComplex<floatT> &right
    ) {
        Tensor4x4Symx4x4SymComplex<floatT> result;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result(firstIndexPair, secondIndexPair, left(firstIndexPair, secondIndexPair) - right(firstIndexPair, secondIndexPair));
            }
        }
        return result;
    }

    __device__ __host__ friend Tensor4x4Symx4x4SymComplex<floatT> operator*(
        const COMPLEX(floatT) &left,
        const Tensor4x4Symx4x4SymComplex<floatT> &right
    ) {
        Tensor4x4Symx4x4SymComplex<floatT> result;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result(firstIndexPair, secondIndexPair, left * right(firstIndexPair, secondIndexPair));
            }
        }
        return result;
    }
    
    __device__ __host__ friend Tensor4x4Symx4x4SymComplex<floatT> operator*(
        const Tensor4x4Symx4x4SymComplex<floatT> &left,
        const COMPLEX(floatT) &right
    ) {
        Tensor4x4Symx4x4SymComplex<floatT> result;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result(firstIndexPair, secondIndexPair, left(firstIndexPair, secondIndexPair) * right);
            }
        }
        return result;
    }

    __device__ __host__ friend Tensor4x4Symx4x4SymComplex<floatT> operator/(
        const Tensor4x4Symx4x4SymComplex<floatT> &left,
        const floatT &right
    ) {
        Tensor4x4Symx4x4SymComplex<floatT> result;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                result(firstIndexPair, secondIndexPair, left(firstIndexPair, secondIndexPair) / right);
            }
        }
        return result;
    }

    __host__ void printIndexPairs() const {
        std::cout << std::scientific << std::showpos << std::setprecision(8);
        std::cout << "Components of Tensor4x4Symx4x4SymComplex:" << std::endl;
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                std::cout <<
                    "tensor[" << firstIndexPair << "][" << secondIndexPair << "]=" << elems[firstIndexPair][secondIndexPair] <<
                std::endl;
            }
        }
    }
    
    __host__ void printFourIndices() const {
        std::cout << std::scientific << std::showpos << std::setprecision(8);
        for (int mu = 0; mu <= 3; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= 3; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
            int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
            std::cout <<
                "tensor_{" <<
                mu << "," << nu << "," << rho << "," << sigma <<
                "}=" << elems[firstIndexPair][secondIndexPair] <<
            std::endl;
        }
    }

    template<PrintComplex printComplex = complex_both>
    __host__ inline void printMatrix10x10() const {
        std::cout << std::scientific << std::showpos << std::setprecision(4);
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                switch (printComplex) {
                    case complex_both:
                        std::cout << elems[firstIndexPair][secondIndexPair] << " ";
                        break;
                    case complex_real:
                        std::cout << real(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                    case complex_imag:
                        std::cout << imag(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                }
            }
            std::cout << std::endl;
        }
    }

    template<PrintComplex printComplex = complex_both>
    __host__ inline void printMatrix4x4FirstSubMatrix(int mu, int nu) const {
        std::cout << std::scientific << std::showpos << std::setprecision(4);
        
        int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
        for (int rho = 0; rho <= 3; rho++) {
            for (int sigma = 0; sigma <= 3; sigma++) {
                int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
                switch (printComplex) {
                    case complex_both:
                        std::cout << elems[firstIndexPair][secondIndexPair] << " ";
                        break;
                    case complex_real:
                        std::cout << real(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                    case complex_imag:
                        std::cout << imag(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                }
            }
            std::cout << std::endl;
        }

    }

    template<PrintComplex printComplex = complex_both>
    __host__ inline void printMatrix4x4SecondSubMatrix(int rho, int sigma) const {
        std::cout << std::scientific << std::showpos << std::setprecision(4);
        
        int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
        for (int mu = 0; mu <= 3; mu++) {
            for (int nu = 0; nu <= 3; nu++) {
                int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
                switch (printComplex) {
                    case complex_both:
                        std::cout << elems[firstIndexPair][secondIndexPair] << " ";
                        break;
                    case complex_real:
                        std::cout << real(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                    case complex_imag:
                        std::cout << imag(elems[firstIndexPair][secondIndexPair]) << " ";
                        break;
                }
            }
            std::cout << std::endl;
        }

    }
    
};

template<class floatT>
__host__ inline std::ostream &operator<<(std::ostream &s, const Tensor4x4Symx4x4SymComplex<floatT> &tensor) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            s << tensor(firstIndexPair, secondIndexPair) << " ";
        }
    }
    return s;
}

template<class floatT>
__device__ __host__ inline bool cmp_all_elements_prec(
    const Tensor4x4Symx4x4SymComplex<floatT> &x,
    const Tensor4x4Symx4x4SymComplex<floatT> &y,
    const floatT prec
) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            if (!compareCOMPLEX(x(firstIndexPair, secondIndexPair), y(firstIndexPair, secondIndexPair), prec)) return false;
        }
    }
    return true;
}
