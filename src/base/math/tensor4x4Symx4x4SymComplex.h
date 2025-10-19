//
// Created by Jonas Winter on 25.08.2025
//

#pragma once
#include "../../define.h"
#include "indices4x4Sym.h"

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

    constexpr Tensor4x4Symx4x4SymComplex(const Tensor4x4Symx4x4SymComplex<floatT>&) = default;

    // type for a function that takes two integers and returns nothing
    // unused, doesn't work right now
    // TODO: Do this another time.
    // typedef __device__ __host__ void (* twoIndexedVoid)(int first, int second);

    // __device__ __host__ void setComponentIndexPairsToValue(int first, int second, COMPLEX(floatT) value) {
    //     elems[first][second] = value;
    // }

    // __device__ __host__ void setComponentIndexPairsToZero(int first, int second) {
    //     setComponentIndexPairsToValue(first, second, 0.0);
    // }

    // __device__ __host__ void doVoidTwoIndexed(twoIndexedVoid func) {
    //     for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
    //         for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
    //             func(firstIndexPair, secondIndexPair);
    //         }
    //     }
    // }

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

    __device__ __host__ inline COMPLEX(floatT) operator()(int firstIndexPair, int secondIndexPair) {
        return elems[firstIndexPair][secondIndexPair];
    }

    __device__ __host__ inline void operator()(int firstIndexPair, int secondIndexPair, COMPLEX(floatT) value) {
        elems[firstIndexPair][secondIndexPair] = value;
    }

    __device__ __host__ inline COMPLEX(floatT) operator()(int mu, int nu, int rho, int sigma) {
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

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT>& operator/=(const floatT &y){
        for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
            for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
                elems[firstIndexPair][secondIndexPair] /= y;
            }
        }
        return *this;
    }

    __host__ inline void printIndexPairs() {
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
    
    __host__ inline void printFourIndices() {
        std::cout << std::scientific << std::showpos << std::setprecision(8);
        for (int mu = 0; mu <= 3; mu++)
        for (int nu = 0; nu <= mu; nu++)
        for (int rho = 0; rho <= 3; rho++)
        for (int sigma = 0; sigma <= rho; sigma++) {
            int firstIndexPair = twoIndicesToIndexPairIndex(mu, nu);
            int secondIndexPair = twoIndicesToIndexPairIndex(rho, sigma);
            std::cout <<
                "Tensor_{" <<
                mu << "," << nu << "," << rho << "," << sigma <<
                "}=" << elems[firstIndexPair][secondIndexPair] <<
            std::endl;
        }
    }

    template<PrintComplex printComplex = complex_both>
    __host__ inline void printMatrix10x10() {
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
    __host__ inline void printMatrix4x4FirstSubMatrix(int mu, int nu) {
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
    __host__ inline void printMatrix4x4SecondSubMatrix(int rho, int sigma) {
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
__device__ __host__ inline bool compareTensor4x4Symx4x4SymComplex(Tensor4x4Symx4x4SymComplex<floatT> a, Tensor4x4Symx4x4SymComplex<floatT> b, floatT tol) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            if (!compareCOMPLEX(a.elems[firstIndexPair][secondIndexPair], b.elems[firstIndexPair][secondIndexPair], tol)) return false;
        }
    }
    return true;
}

template<class floatT>
__host__ inline std::ostream &operator<<(std::ostream &s, Tensor4x4Symx4x4SymComplex<floatT> tensor) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            s << tensor.elems[firstIndexPair][secondIndexPair];
        }
    }
    return s;
}

// template<class floatT>
// void inline printTensor4x4Symx4x4SymComplexComponentIndexPairs(Tensor4x4Symx4x4SymComplex<floatT> tensor, int firstIndexPair, int secondIndexPair) {
//     std::cout <<
//         "firstIndexPair=" << firstIndexPair <<", " <<
//         "secondIndexPair=" << secondIndexPair << ", " <<
//         "component=" << tensor.elems[firstIndexPair][secondIndexPair] <<
//     std::endl;
// }

// // debugging functions
// // __device__ left out because of std::cout
// template<class floatT>
// __attribute__((unused)) void  __host__ inline printTensor4x4Symx4x4SymComplexIndexPairs(const Tensor4x4Symx4x4SymComplex<floatT> &tensor) {
//     std::cout << "Components of Tensor4x4Symx4x4SymComplex:\n" << std::endl;
//     for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
//         for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
//             printTensor4x4Symx4x4SymComplexComponentIndexPairs<floatT>(tensor, firstIndexPair, secondIndexPair);
//         }
//     }
// }

template<class floatT>
__device__ __host__ inline bool cmp_all_elements_prec(
    const Tensor4x4Symx4x4SymComplex<floatT> &x,
    const Tensor4x4Symx4x4SymComplex<floatT> &y,
    const floatT prec
) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            if (!compareCOMPLEX(x.elems[firstIndexPair][secondIndexPair], y.elems[firstIndexPair][secondIndexPair], prec)) return false;
        }
    }
    return true;
}

template<class floatT>
__device__ __host__ inline bool cmp_all_elements_prec_to_value(
    const Tensor4x4Symx4x4SymComplex<floatT> &x,
    const floatT value,
    const floatT prec
) {
    for (int firstIndexPair = 0; firstIndexPair < 10; firstIndexPair++) {
        for (int secondIndexPair = 0; secondIndexPair < 10; secondIndexPair++) {
            COMPLEX(floatT) complexValue = value;
            if (!compareCOMPLEX(x.elems[firstIndexPair][secondIndexPair], complexValue, prec)) return false;
        }
    }
    return true;
}
