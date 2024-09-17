#pragma once

enum syclError_t {
    syclSuccess = 0,
    syclErrorInvalidValue = 1,
    syclErrorOutOfMemory = 2,
    syclErrorUnknown = 999
};


syclError_t syclExceptionWrapper(std::function<void()> func);
