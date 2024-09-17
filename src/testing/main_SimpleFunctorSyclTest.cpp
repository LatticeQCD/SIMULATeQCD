#include <sycl/sycl.hpp>
#include "../base/math/operators.h"
#include "../base/indexer/haloIndexer.h"
#include "../base/runFunctors_sycl.h"
#include "../base/communication/communicationBase.h"
#include "../base/memoryManagement.h"

#define RELEVANT_SIZE 10


class SpinorAccessor{
    double* _mem;

    public:

    SpinorAccessor(double* mem) : _mem(mem){
    }

    double operator()(const int i) const{
        return _mem[i];
    }

    double& operator[](const int i) const{
        return _mem[i];
    }

    void setElement(const int i, double elem) const{
        _mem[i] = elem;
    }

    void setElement(const gSite site, double elem) const {
        int i = site.isite;
        _mem[i] = elem;
    }
    
};



// Simple class to simulate a Spinor
class Spinor : public RunFunctors<true, SpinorAccessor> {
    const int _size;
    double* _h_mem;
    double* _d_mem;
    sycl::queue _q;
    friend std::ostream& operator<< (std::ostream& stream, const Spinor& a);
    friend bool cmp_rel(Spinor& lhs, Spinor& rhs, double, double);

    public:

    explicit Spinor(const int _size):_size(_size){
        _h_mem = new double[_size];

        // gpuErr = gpuMalloc((void **)&_d_mem, _size*sizeof(double));
        _d_mem = sycl::malloc_device<double>(_size,_q);

    }

    void upload(){
#ifndef CPU
        // gpuError_t gpuErr;
        // gpuErr = gpuMemcpy(_d_mem, _h_mem, _size*sizeof(double), gpuMemcpyHostToDevice);
        // if (gpuErr) GpuError("SimpleFunctorTest, Spinor::upload: gpuMemcpy", gpuErr);

        _q.memcpy(_d_mem, _h_mem, _size*sizeof(double)).wait();
#endif
    }

    void download(){
#ifndef CPU
        // gpuError_t gpuErr;
        // gpuErr = gpuMemcpy(_h_mem, _d_mem, _size*sizeof(double), gpuMemcpyDeviceToHost);
        // if (gpuErr) GpuError("SimpleFunctorTest, Spinor::download: gpuMemcpy", gpuErr);

        _q.memcpy(_h_mem, _d_mem, _size*sizeof(double)).wait();
#endif
    }


    ~Spinor(){
#ifndef CPU
//         gpuError_t gpuErr;
//         gpuErr = gpuFree(_d_mem);
//         if (gpuErr) GpuError("SimpleFunctorTest, Spinor::~Spinor: gpuFree", gpuErr);
        free(_d_mem,_q);
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

        performFunctorsSyclLaunch(this->getAccessor(), func, _size, _q);


        return *this;
    }

    template<typename Function>
    void iterateOverBulk(Function op) {
        CalcGSite<All, 0> calcGSite;
        WriteAtRead writeAtRead;

        this->template iterateFunctor<64>(op, calcGSite, writeAtRead, _size, _q);
    }
};


std::ostream& operator<< (std::ostream& stream, const Spinor& rhs){
    for (int i = 0; i < RELEVANT_SIZE; i++){ //print only first 10 elements
        std::cout << rhs._h_mem[i] << std::endl;
    }
    return stream;
}



struct ReferenceFunctor{
    SpinorAccessor a;
    SpinorAccessor b;
    SpinorAccessor c;
    SpinorAccessor d;

    ReferenceFunctor(Spinor& a, Spinor& b, Spinor& c, Spinor& d) :
        a(a.getAccessor()), b(b.getAccessor()), c(c.getAccessor()), d(d.getAccessor()){}

    __host__ __device__ double operator()(const int i) const {
        return a[i] + b[i] + c[i] + 5 * d[i];
    }
};

struct ComplexReferenceFunctor{
    SpinorAccessor a;
    SpinorAccessor b;
    SpinorAccessor c;
    SpinorAccessor d;

    ComplexReferenceFunctor(Spinor& a, Spinor& b, Spinor& c, Spinor& d) :
        a(a.getAccessor()), b(b.getAccessor()), c(c.getAccessor()), d(d.getAccessor()){}

    __host__ __device__ double operator()(const int i) const {

        return a[i]*b[i] + a[i]/b[i]
                         - ((a[i] + b[i]) / (a[i] - 2.3*b[i])) * (2*a[i])
                         + (a[i]*2) * (a[i]/2) / (2/a[i])
                         + (c[i] + 4) + (4 + c[i]) + (c[i] - 4) + (4 - c[i]) + c[i] * (a[i] + c[i])
                         + (a[i]*b[i])*2 + 2*(a[i]*c[i])
                         + (a[i]*b[i])/2 + 2/(a[i]*c[i])
                         + (2 + (a[i]*b[i])) + (2+(a[i]*c[i]))
                         + (2 - (a[i]*b[i])) + (2-(a[i]*c[i]));
    }

    double operator() (const gSite site) const {
        int i = site.isite;
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
        std::cout << text << " PASSED" << std::endl;
    }else{
        std::cout << text << " FAILED" << std::endl;
        std::cout << ref << " vs" << std::endl;
        std::cout << res << std::endl; 
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



// #ifndef CPU
// template <typename Functor>
// void performFunctors(SpinorAccessor res, Functor op, const int size){

//     const int i = blockDim.x * blockIdx.x + threadIdx.x;
//     // sycl::nd_item<1>

   
//         res[i] = op(i);
    
// }
// #endif


template<typename Function>
void performFunctorsSyclLaunch(SpinorAccessor res, Function op, const int size, sycl::queue q) {
    #ifndef CPU
        
        sycl::range<1> ndr(size);

        std::cout << "Size of operator " << sizeof(op) << "\n";
        
        
        q.parallel_for(ndr, [=](sycl::item<1> item) {
            auto i = item.get_id();
            res[i] = op(i);
            
        }).wait();
    
    #else
        for (int i = 0; i < size; i++){
            res[i] = op(i);
            res.setElement(i, op(i));
        }
    #endif


}


void referenceSyclLaunch(SpinorAccessor res, SpinorAccessor a, SpinorAccessor b, SpinorAccessor c,
        SpinorAccessor d, const int size, sycl::queue q){

#ifndef CPU

    const int blockDim = 256;
    const int gridDim = static_cast<int> (ceilf(static_cast<float> (size)) / static_cast<float> (blockDim));

    std::cout << "blockDim " << blockDim << std::endl;
    std::cout << "gridDim " << gridDim << std::endl;
    sycl::range<1> ndr(size);
    q.parallel_for(ndr, [=](sycl::item<1> item) {
        auto i = item;
        res[i] = a[i] + b[i] + c[i] + 5 * d[i];
    }).wait();

#else
    for (size_t i = 0; i < size; i++) {
        res[i] = a[i] + b[i] * c[i] + 5 * d[i];
    }
#endif
}


int main(int argc, char *argv[]){

    // rootLogger.setVerbosity(DEBUG);
    for (auto platform : sycl::platform::get_platforms())
        {
            std::cout << "Platform: "
                    << platform.get_info<sycl::info::platform::name>()
                    << std::endl;

            for (auto device : platform.get_devices())
            {
                std::cout << "\tDevice: "
                        << device.get_info<sycl::info::device::name>()
                        << std::endl;
            }
        }
    
    auto sycl_devices = sycl::device::get_devices();
    for (auto dev : sycl_devices) {
        if (dev.has(sycl::aspect::gpu)) {
            std::cout << "\tDevice with sycl::aspect::gpu: " << dev.get_info<sycl::info::device::name>() << std::endl;
        }
    }
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


    // res = a + b * c + 5 * d;
    
    // std::cout << res << std::endl<<std::endl;
    ref = ReferenceFunctor(a,b,c,d);
    // std::cout << ref <<std::endl <<std::endl;
    CommunicationBase commBase(&argc, &argv);
    // compare_relative(ref, res, 1e-8, 1e-8, "Combined operators vs one operator test");

    // std::cout << res << std::endl<<std::endl;
    sycl::queue q;//(sycl::gpu_selector_v);
    q.wait();
    auto info = q.get_device().get_info<sycl::info::device::name>();
    std::cout << "Chosen Device: " << info << std::endl;
    
    auto sycl_target = SYCL_TARGET;
    std::cout << "Compiled for SYCL_TARGET: " << sycl_target << std::endl;
    auto test_devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    referenceSyclLaunch(res.getAccessor(), a.getAccessor(), b.getAccessor(), c.getAccessor(),
            d.getAccessor(), nelems, q);

    
    // std::cout << res << std::endl<<std::endl;
    compare_relative(ref, res, 1e-8, 1e-8, "Reference kernel vs operators test");
    
    // This should cover all existing kinds of operator combinations
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
    ComplexReferenceFunctor functor(a,b,c,d);
    res.iterateOverBulk(functor);
    compare_relative(res, ref, 1e-8, 1e-8, "runFunctors_sycl.h test");
    
    return 0;
}