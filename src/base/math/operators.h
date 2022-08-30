#ifndef OPERAIndexOR_H
#define OPERAIndexOR_H

#include <type_traits>
#include "gcomplex.h"
#include "gsu3.h"
#include "gvect3.h"
#include "../indexer/BulkIndexer.h"


/*! Using the syntax below stuff like this is possible:
 *   Spinor a, b, c, d
 *   Spinor a = b*c + d;
 * The way this works is the following: Instead of actually performing the operations + and *, they only return an
 * object (GeneralOperator) that holds all the information how to compute the operation. The operation is not
 * executed at the level of the operators, though. Instead the operations are executed at the copy assignment operator
 * (=) of the spinor. This is done by calling a kernel, which takes the operator object as right hand side object.
 * Inside this kernel the operation is then executed.
 */
enum Operation {
    add, subtract, mult, divide
};

template<typename T>
class custom_is_scalar{ public: static constexpr bool value = std::is_scalar<T>::value; };
#ifndef USE_CPU_ONLY
template <>
class custom_is_scalar<__half> {public: static constexpr bool value = true; };
#endif
template<typename T>
class custom_is_class{ public: static constexpr bool value = std::is_class<T>::value; };
#ifndef USE_CPU_ONLY
template <>
class custom_is_class<__half>{ public: static constexpr bool value = false; };
#endif

template<typename typeLHS,
        typename typeRHS,
        Operation op,
        typename testLHS = void,
        typename testRHS = void>
struct GeneralOperator {
};


/*!
 * Set of very general functions which may be combined into one operator call, that is sent to the GPU.
 * Each of the operator participants either has to be a class with the method getAccessor() and the operator ()
 * or a scalar.
 */
template<typename typeLHS, typename typeRHS>
auto general_mult(const typeLHS &lhs, const typeRHS &rhs) {
    return GeneralOperator<typeLHS, typeRHS, mult>(lhs, rhs);
}

template<typename typeLHS, typename typeRHS>
auto general_divide(const typeLHS &lhs, const typeRHS &rhs) {
    return GeneralOperator<typeLHS, typeRHS, divide>(lhs, rhs);
}

template<typename typeLHS, typename typeRHS>
auto general_add(const typeLHS &lhs, const typeRHS &rhs) {
    return GeneralOperator<typeLHS, typeRHS, add>(lhs, rhs);
}

template<typename typeLHS, typename typeRHS>
auto general_subtract(const typeLHS &lhs, const typeRHS &rhs) {
    return GeneralOperator<typeLHS, typeRHS, subtract>(lhs, rhs);
}


//Unfortunately there is a lot of code repetition below, but there is no way to avoid that.


//Class + Class
//Using Multiple type specialization with SFINAE:
//https://stackoverflow.com/questions/40513553/how-to-specialize-methods-for-multiple-types-at-once
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<
        typeLHS, //Left hand side object. (Spinor, GeneralOperator or something else that has the getAccessor() method)
        typeRHS, //Right hand side object
        add, //Type of operator (add mult divide subtract)
        std::enable_if_t<custom_is_class<typeLHS>::value>, //SFINAE to check if typeLHS is a class
        std::enable_if_t<custom_is_class<typeRHS>::value> //SFINAE to check if typeRHS is a class
> {
    //
    //We need to get the type of the return value of getAccessor(). This is what is stored here.
    //https://stackoverflow.com/questions/5580253/get-return-type-of-member-function-without-an-object
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    //Constructor copies the accessors
    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()), _rhs(rhs.getAccessor()) {}

    //GeneralOperator is its own accessor
    GeneralOperator<typeLHS,
            typeRHS,
            add,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    //Call the operator: Do the operation element wise.
    //This is what is called in another operator or in a Kernel which runs the operation
    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        //inline __host__ __device__ auto operator()(const Index i) const {
        auto rhs = _rhs(i);
        auto lhs = _lhs(i);
        return lhs + rhs;
    }
};

//Class + scalar
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        add,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_scalar<typeRHS>::value> //Check if typeRHS is a scalar
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;

    //If typeRHS is a scalar, we do not need to get its accessor type. (There is no accessor)
    const typeRHS _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()),_rhs(rhs) {}

    //GeneralOperator is its own accessor
    GeneralOperator<typeLHS,
            typeRHS,
            add,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_scalar<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto lhs = _lhs(i);
        return lhs + _rhs;
    }
};


//Scalar + Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        add,
        std::enable_if_t<custom_is_scalar<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    typeLHS _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            add,
            std::enable_if_t<custom_is_scalar<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        return _lhs + rhs;
    }
};


//Class - Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        subtract,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            subtract,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        auto lhs = _lhs(i);
        return lhs - rhs;
    }
};


//Class - scalar
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        subtract,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_scalar<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const typeRHS _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()),_rhs(rhs) {}

    GeneralOperator<typeLHS,
            typeRHS,
            subtract,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_scalar<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto lhs = _lhs(i);
        return lhs - _rhs;
    }
};


//Scalar - Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        subtract,
        std::enable_if_t<custom_is_scalar<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    typeLHS _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            subtract,
            std::enable_if_t<custom_is_scalar<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        return _lhs - rhs;
    }
};


//Class * Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        mult,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            mult,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        auto lhs = _lhs(i);
        return lhs * rhs;
    }
};


//Class * scalar
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        mult,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_scalar<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const typeRHS _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()),_rhs(rhs) {}

    GeneralOperator<typeLHS,
            typeRHS,
            mult,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_scalar<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto lhs = _lhs(i);
        return lhs * _rhs;
    }
};




//Scalar * Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        mult,
        std::enable_if_t<custom_is_scalar<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    typeLHS _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            mult,
            std::enable_if_t<custom_is_scalar<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value>
    > getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        return _lhs * rhs;
    }
};


//Class / Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        divide,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            divide,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        auto lhs = _lhs(i);
        return lhs / rhs;
    }
};

//Class / scalar
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        divide,
        std::enable_if_t<custom_is_class<typeLHS>::value>,
        std::enable_if_t<custom_is_scalar<typeRHS>::value>
> {
    const decltype(std::declval<typeLHS>().getAccessor()) _lhs;
    const typeRHS _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs.getAccessor()),_rhs(rhs) {}

    GeneralOperator<typeLHS,
            typeRHS,
            divide,
            std::enable_if_t<custom_is_class<typeLHS>::value>,
            std::enable_if_t<custom_is_scalar<typeRHS>::value> >
    getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto lhs = _lhs(i);
        return lhs / _rhs;
    }
};


//Scalar / Class
template<typename typeLHS, typename typeRHS>
struct GeneralOperator<typeLHS,
        typeRHS,
        divide,
        std::enable_if_t<custom_is_scalar<typeLHS>::value>,
        std::enable_if_t<custom_is_class<typeRHS>::value>
> {
    typeLHS _lhs;
    const decltype(std::declval<typeRHS>().getAccessor()) _rhs;

    GeneralOperator(const typeLHS &lhs, const typeRHS &rhs) :
            _lhs(lhs), _rhs(rhs.getAccessor()) {}

    GeneralOperator<typeLHS,
            typeRHS,
            divide,
            std::enable_if_t<custom_is_scalar<typeLHS>::value>,
            std::enable_if_t<custom_is_class<typeRHS>::value>
    > getAccessor() const {
        return *this;
    }

    template<typename Index>
    inline __host__ __device__ auto operator()(const Index i) const
    {
        auto rhs = _rhs(i);
        return _lhs / rhs;
    }
};


//Define operators for nested operators. Something like a*b + c*d

template<typename typeLHS1, typename typeRHS1, Operation op1,
        typename typeLHS2, typename typeRHS2, Operation op2>
auto operator-(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const GeneralOperator<typeLHS2, typeRHS2, op2> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
            GeneralOperator<typeLHS2, typeRHS2, op2>, subtract>(lhs, rhs);
}

template<typename typeLHS1, typename typeRHS1, Operation op1,
        typename typeLHS2, typename typeRHS2, Operation op2>
auto operator+(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const GeneralOperator<typeLHS2, typeRHS2, op2> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
            GeneralOperator<typeLHS2, typeRHS2, op2>, add>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1,
        typename typeLHS2, typename typeRHS2, Operation op2>
auto operator*(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const GeneralOperator<typeLHS2, typeRHS2, op2> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
            GeneralOperator<typeLHS2, typeRHS2, op2>, mult>(lhs, rhs);
}

template<typename typeLHS1, typename typeRHS1, Operation op1,
        typename typeLHS2, typename typeRHS2, Operation op2>
auto operator/(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const GeneralOperator<typeLHS2, typeRHS2, op2> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
            GeneralOperator<typeLHS2, typeRHS2, op2>, divide>(lhs, rhs);
}



//Enable Combinations operator and scalar. Such as (a*b)*2.0
//Te exclude stuff like Spinor or Gaugefield to be part of these operators,
//we have to do template magic as hell
//
//Once you understand how SFINAE works, it does not look so complicated any more
//Idea taken from https://stackoverflow.com/questions/44387251/thrust-reduction-and-overloaded-operator-const-float3-const-float3-wont-co

template<typename inputType>
using isAllowedType = typename std::enable_if<custom_is_scalar<inputType>::value
                                              #ifndef USE_CPU_ONLY
                                              || std::is_same<inputType, GCOMPLEX(__half)>::value
                                              || std::is_same<inputType, GSU3<__half> >::value
                                              || std::is_same<inputType, gVect3<__half> >::value
                                              #endif
                                              || std::is_same<inputType, GCOMPLEX(float)>::value
                                              || std::is_same<inputType, GCOMPLEX(double)>::value
                                              || std::is_same<inputType, GSU3<float> >::value
                                              || std::is_same<inputType, GSU3<double> >::value
                                              || std::is_same<inputType, gVect3<float> >::value
                                              || std::is_same<inputType, gVect3<double> >::value, inputType>::type;


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeRHS>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        isAllowedType<typeRHS>,
        add>
operator+(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const typeRHS rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, typeRHS, add>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeLHS>
GeneralOperator<isAllowedType<typeLHS>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        add>
operator+(const typeLHS lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<typeLHS, GeneralOperator<typeLHS1, typeRHS1, op1>, add>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeRHS>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        isAllowedType<typeRHS>,
        subtract>
operator-(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const typeRHS rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, typeRHS, subtract>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeLHS>
GeneralOperator<isAllowedType<typeLHS>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        subtract>
operator-(const typeLHS lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<typeLHS, GeneralOperator<typeLHS1, typeRHS1, op1>, subtract>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeRHS>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        isAllowedType<typeRHS>,
        mult>
operator*(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const typeRHS rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, typeRHS, mult>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeLHS>
GeneralOperator<isAllowedType<typeLHS>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        mult>
operator*(const typeLHS lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<typeLHS, GeneralOperator<typeLHS1, typeRHS1, op1>, mult>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeRHS>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        isAllowedType<typeRHS>,
        divide>
operator/(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const typeRHS rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, typeRHS, divide>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename typeLHS>
GeneralOperator<isAllowedType<typeLHS>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        divide>
operator/(const typeLHS lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<typeLHS, GeneralOperator<typeLHS1, typeRHS1, op1>, divide>(lhs, rhs);
}


#endif
