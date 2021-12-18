/* 
 * main_SimpleFunctorTest.cu                                                               
 * 
 * Simplest relevant implementations of Functor syntax. 
 * 
 */

#include <iostream>
#include <exception>
#include "../SIMULATeQCD.h"

#define RELEVANT_SIZE 10

// Simple flag to switch to pure cpu code. Used for debugging
//#define CPU

class SpinorAccessor{
    double* _mem;

    public:

    SpinorAccessor(double* mem) : _mem(mem){
    }

    __host__ __device__ double operator()(const int i) const{
        return _mem[i];
    }

    __host__ __device__ double& operator[](const int i) const{
        return _mem[i];
    }

    __host__ __device__ void setElement(const int i, double elem) const{
        _mem[i] = elem;
    }
};


// Simple class to simulate a Spinor
class Spinor{
    int _size;
    double* _h_mem;
    double* _d_mem;

    friend std::ostream& operator<< (std::ostream& stream, const Spinor& a);
    friend bool cmp_rel(Spinor& lhs, Spinor& rhs, double, double);

    public:

    explicit Spinor(const int _size):_size(_size){
        _h_mem = new double[_size];
#ifndef CPU
        gpuMalloc((void **)&_d_mem, _size*sizeof(double));
#endif
    }

    void upload(){
#ifndef CPU
        gpuMemcpy(_d_mem, _h_mem, _size*sizeof(double), gpuMemcpyHostToDevice);
#endif
    }

    void download(){
#ifndef CPU
        gpuMemcpy(_h_mem, _d_mem, _size*sizeof(double), gpuMemcpyDeviceToHost);
#endif
    }

        
    ~Spinor(){
#ifndef CPU
        gpuFree(_d_mem);
#endif
        delete[] _h_mem;
    }

    void fill(){
        for (int i = 0; i < _size; i++){
            _h_mem[i] = i%RELEVANT_SIZE + 1;
        }
        upload();
    }
    
    SpinorAccessor getAccessor() const{
#ifndef CPU
        return SpinorAccessor(_d_mem);
#else
        return SpinorAccessor(_h_mem);
#endif
    }

    template<typename Function>
    Spinor& operator=(Function func){

        performFunctorsLaunch(this->getAccessor(), func, _size);
        

        return *this;
    }
};


bool cmp_rel(Spinor& lhs, Spinor& rhs, double rel, double prec){
#ifndef CPU
    lhs.download();
    rhs.download();
#endif
    bool success = true;
    for (size_t i = 0; i < RELEVANT_SIZE; i++){
        success &= cmp_rel(lhs._h_mem[i], rhs._h_mem[i], rel, prec);
    }
    return success;
}


void compare_relative(Spinor& ref, Spinor& res, double rel, double prec, std::string text){
    if (cmp_rel(ref, res, rel, prec)){
        rootLogger.info(text ,  " PASSED");
    }else{
        rootLogger.error(text ,  " FAILED");
        rootLogger.error(ref ,  " vs");
        rootLogger.error(res);
    }
}


template<typename T> 
auto operator*(const T lhs, const Spinor& rhs){
    return general_mult(lhs, rhs);
}

template<typename T>
auto operator*(const Spinor& lhs, const T rhs){
    return general_mult(lhs, rhs);
}

auto operator*(const Spinor& lhs, const Spinor& rhs){
    return general_mult(lhs, rhs);
}

auto operator/(const Spinor& lhs, const Spinor& rhs){
    return general_divide(lhs, rhs);
}

template<typename T> 
auto operator/(const T lhs, const Spinor& rhs){
    return general_divide(lhs, rhs);
}

template<typename T> 
auto operator/(const Spinor& lhs, const T rhs){
    return general_divide(lhs, rhs);
}

auto operator+(const Spinor& lhs, const Spinor& rhs){
    return general_add(lhs, rhs);
}

template<typename T> 
auto operator+(const T lhs, const Spinor& rhs){
    return general_add(lhs, rhs);
}

template<typename T> 
auto operator+(const Spinor& lhs, const T rhs){
    return general_add(lhs, rhs);
}

auto operator-(const Spinor& lhs, const Spinor& rhs){
    return general_subtract(lhs, rhs);
}

template<typename T> 
auto operator-(const T lhs, const Spinor& rhs){
    return general_subtract(lhs, rhs);
}

template<typename T> 
auto operator-(const Spinor& lhs, const T rhs){
    return general_subtract(lhs, rhs);
}


#ifndef CPU
template <typename Functor>
__global__ void performFunctors(SpinorAccessor res, Functor op, const int size){

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        res[i] = op(i);
    }
}
#endif


template <typename Function>
void performFunctorsLaunch(SpinorAccessor res, Function op, const int size){

#ifndef CPU
        const dim3 blockDim = 256;
        const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (size)
                                               / static_cast<float> (blockDim.x)));

        rootLogger.debug("Size of operator " ,  sizeof(op) ,  std::endl);
        performFunctors <<< gridDim, blockDim >>>(res, op, size);

        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctorLaunch: Failed to launch kernel", gpuErr);
#else
        for (int i = 0; i < size; i++){
            res[i] = op(i);
            res.setElement(i, op(i));
        }
#endif
}


std::ostream& operator<< (std::ostream& stream, const Spinor& rhs){
    for (int i = 0; i < RELEVANT_SIZE; i++){ //print only first 10 elements
        std::cout << rhs._h_mem[i] << std::endl;
    }
    return stream;
}


#ifndef CPU
__global__ void reference(SpinorAccessor res, SpinorAccessor a, SpinorAccessor b,
        SpinorAccessor c, SpinorAccessor d, const int size){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<size) {
        res[i] = a[i] + b[i] * c[i] + 5 * d[i];
    }
}

#endif



void referenceLaunch(SpinorAccessor res, SpinorAccessor a, SpinorAccessor b, SpinorAccessor c,
        SpinorAccessor d, const int size){

#ifndef CPU

    const dim3 blockDim = 256;
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (size)
                / static_cast<float> (blockDim.x)));

    reference<<< gridDim, blockDim >>>(res, a, b, c, d, size);
    gpuError_t gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("performReferenceLaunch: Failed to launch kernel", gpuErr);
#else
    for (size_t i = 0; i < size; i++) {
        res[i] = a[i] + b[i] * c[i] + 5 * d[i];
    }
#endif
}


struct ReferenceFunctor{
    SpinorAccessor a;
    SpinorAccessor b;
    SpinorAccessor c;
    SpinorAccessor d;

    ReferenceFunctor(Spinor& a, Spinor& b, Spinor& c, Spinor& d) :
        a(a.getAccessor()), b(b.getAccessor()), c(c.getAccessor()), d(d.getAccessor()){}

    __host__ __device__ double operator()(const int i){
        return a[i] + b[i] * c[i] + 5 * d[i];
    }
};


struct ComplexReferenceFunctor{
    SpinorAccessor a;
    SpinorAccessor b;
    SpinorAccessor c;
    SpinorAccessor d;

    ComplexReferenceFunctor(Spinor& a, Spinor& b, Spinor& c, Spinor& d) :
        a(a.getAccessor()), b(b.getAccessor()), c(c.getAccessor()), d(d.getAccessor()){}

    __host__ __device__ double operator()(const int i){

        return a[i]*b[i] + a[i]/b[i]
                         - ((a[i] + b[i]) / (a[i] - 2.3*b[i])) * (2*a[i])
                         + (a[i]*2) * (a[i]/2) / (2/a[i])
                         + (c[i] + 4) + (4 + c[i]) + (c[i] - 4) + (4 - c[i]) + c[i] * (a[i] + c[i])
                         + (a[i]*b[i])*2 + 2*(a[i]*c[i])
                         + (a[i]*b[i])/2 + 2/(a[i]*c[i])
                         + (2 + (a[i]*b[i])) + (2+(a[i]*c[i]))
                         + (2 - (a[i]*b[i])) + (2-(a[i]*c[i]));
    }
};


int main(){

    rootLogger.setVerbosity(DEBUG);

    int nelems = 1e5;
    Spinor a(nelems);
    Spinor b(nelems);
    Spinor c(nelems);
    Spinor d(nelems);

    //result
    Spinor res(nelems);

    //reference
    Spinor ref(nelems);

    a.fill();
    b.fill();
    c.fill();
    d.fill();

    MicroTimer timer;
    timer.start();

    res = a + b * c + 5 * d;
    timer.stop();
    timer.print("combined operators");

    timer.start();
    ref = ReferenceFunctor(a, b, c, d);
    timer.stop();
    timer.print("reference operator");

    compare_relative(res, ref, 1e-8, 1e-8, "Combined operators vs one operator test");

    timer.start();

    referenceLaunch(res.getAccessor(), a.getAccessor(), b.getAccessor(), c.getAccessor(),
            d.getAccessor(), nelems);

    timer.stop();
    timer.print("reference kernel");

    compare_relative(res, ref, 1e-8, 1e-8, "Reference kernel vs operators test");
    
    //This should cover all existing kinds of operator combinations
    res = a*b + a/b
              - ((a + b) / (a - 2.3*b)) * (2*a)
              + (a*2) * (a/2) / (2/a)
              + (c + 4) + (4 + c) + (c - 4) + (4 - c) + c * (a + c)
              + (a*b)*2 + 2*(a*c)
              + (a*b)/2 + 2/(a*c)
              + (2 + (a*b)) + (2+(a*c))
              + (2 - (a*b)) + (2-(a*c));
    
    ref = ComplexReferenceFunctor(a, b, c, d);

    compare_relative(res, ref, 1e-8, 1e-8, "Complex operators vs one operator test");

    return 0;
}

