#ifndef _INC_TIMER
#define _INC_TIMER

#include "../define.h"
#include "wrapper/gpu_wrapper.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <math.h>


//! A class to time events/function calls

template<bool device = true>
class StopWatch {
    float _elapsed;
    
    long _bytes;
    long _flops;

    using host_clock = std::chrono::high_resolution_clock;
    
    host_clock::time_point _host_start_time, _host_stop_time;

    inline void _host_start() { 
        _host_start_time = host_clock::now(); 
    }

    inline double _host_stop() {
        float time;
        _host_stop_time = host_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(_host_stop_time - 
                                                                    _host_start_time)
            .count();
        _elapsed += time/1000;
        return _elapsed;
    }

#ifdef __CUDACC__
    gpuEvent_t _device_start_time, _device_stop_time;

    public:

    StopWatch() : _elapsed(0.0), _bytes(0), _flops(0) {
        if(device == true){
            gpuEventCreate(&_device_start_time);
            gpuEventCreate(&_device_stop_time);
        }
    }

    ~StopWatch() {
        if(device == true){
            gpuEventDestroy(_device_start_time);
            gpuEventDestroy(_device_stop_time);
        }
    }

    inline void start() { 
        if(device == true){
            gpuEventRecord(_device_start_time, 0); 
        }
        else{
            _host_start();
        }

    }

    inline double stop() {
        if(device == true){
            gpuEventRecord(_device_stop_time, 0);
            gpuEventSynchronize(_device_stop_time);
            float time;
            gpuEventElapsedTime(&time, _device_start_time, _device_stop_time);
            _elapsed += time;
            return _elapsed;
        }
        else{
            return _host_stop();
        }
    }
#else
    public:
    
    StopWatch() : _elapsed(0.0), _bytes(0), _flops(0) {}

    inline void start() { 
        if(device == true){
            throw std::runtime_error( stdLogger.fatal("StopWatch.start() error:", 
                                    "No device timer support with that compiler!"));
        }
        else{
            _host_start(); 
        }
    }
    inline double stop() { 
        if(device == true){
            throw std::runtime_error( stdLogger.fatal("StopWatch.stop() error:", 
                                    "No device timer support with that compiler!"));
        }
        else{
            return _host_stop(); 
        }
    }
#endif

    void reset() { _elapsed = 0; }

    void print(std::string text) {
        rootLogger.info("Time for " + text + " " ,  _elapsed ,  "ms");
    }


    double ms() const { return _elapsed; }
    //! set how many bytes were processed (for an MB/s output)
    void setBytes(const long b ) { _bytes = b ; }
    //! set how many FLOPs were calculated 
    void setFlops(const long f ) { _flops = f ; } 

    //! return MBytes/s (be sure to call setBytes() before)
    double mbs() const {
        return (_bytes / (ms()*1024*1024/1000.));
    }

    //! return MFLOP/s (be sure to call setFlops() before)
    double mflps() const {
        return (_flops / (ms()*1000.));
    }


    //! Add two timings (bytes and flops are not transfered atm)
    StopWatch operator+(const StopWatch & lhs) const {
        StopWatch ret;
        ret._elapsed = _elapsed + lhs._elapsed;
        return ret;
    }
    //! Substract two timings (_bytes and flops are not transfered atm)
    StopWatch operator-(const StopWatch & lhs) const {
        StopWatch ret;
        ret._elapsed = _elapsed - lhs._elapsed;
        return ret;
    }

    //! Divide the timing by an integer (e.g. loop count)
    StopWatch & operator/=(const int & lhs) {
        _elapsed /= lhs;
        return *this ;
    }   

    //! Calculate ratio of two timings
    double operator/(const StopWatch & lhs) {
        return (_elapsed / lhs._elapsed);
    }



    template<bool _device>
    inline friend std::ostream &operator<<(std::ostream &stream,
            const StopWatch<_device> &rhs);




};

template<bool device>
std::ostream &operator<<(std::ostream &stream, const StopWatch<device> &rhs) {
    stream << "Time = " << rhs._elapsed << "ms";
    return stream;
}

#endif



