#include <iostream>
#include <functional>
#include <sycl/sycl.hpp>
#include "syclExceptionWrapper.h"

syclError_t syclExceptionWrapper(std::function<void()> func) {
    try {
        func();
        return syclSuccess;
    } catch (sycl::invalid_parameter_error &e) {
        std::cerr << "SYCL Invalid Parameter Error: " << e.what() << std::endl;
        return syclErrorInvalidValue;
    } catch (sycl::memory_allocation_error &e) {
        std::cerr << "SYCL Memory Allocation Error: " << e.what() << std::endl;
        return syclErrorOutOfMemory;
    } catch (sycl::exception &e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return syclErrorUnknown;
    } catch (std::exception &e) {
        std::cerr << "STD Exception: " << e.what() << std::endl;
        return syclErrorUnknown;
    }


}


