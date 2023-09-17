#ifndef _INC_TIMER
#define _INC_TIMER

#include "../define.h"
#include "wrapper/gpu_wrapper.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <cmath>

//! A class to time events/function calls

template<bool device>
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

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
    gpuEvent_t _device_start_time, _device_stop_time;

    public:

    StopWatch() : _elapsed(0.0), _bytes(0), _flops(0) {
        if(device){
            gpuError_t gpuErr = gpuEventCreate(&_device_start_time);

            if (gpuErr) {
                GpuError("Error in gpu Event creation (start)" , gpuErr);
            }

            gpuErr = gpuEventCreate(&_device_stop_time);

            if (gpuErr) {
                GpuError("Error in gpu Event creation (stop)" , gpuErr);
            }
        }
    }

    ~StopWatch() {
        if(device){
            gpuError_t gpuErr = gpuEventDestroy(_device_start_time);
            if (gpuErr) {
                GpuError("Error in gpu Event destruction (start)" , gpuErr);
            }
            gpuErr = gpuEventDestroy(_device_stop_time);
            if (gpuErr) {
                GpuError("Error in gpu Event destruction (stop)" , gpuErr);
            }
        }
    }

    inline void start() { 
        if(device){
            gpuError_t gpuErr = gpuEventRecord(_device_start_time, nullptr);
            if (gpuErr) {
                GpuError("Error in gpuEventRecord (start). Make sure that StopWatch is constructed only after CommunicationBase::init has been called!" , gpuErr);
            }

        }
        else{
            _host_start();
        }

    }

    inline double stop() {
        if(device){
            gpuError_t gpuErr = gpuEventRecord(_device_stop_time, nullptr);
            if (gpuErr) {
                GpuError("Error in gpuEventRecord (stop)" , gpuErr);
            }
            gpuErr = gpuEventSynchronize(_device_stop_time);
            if (gpuErr) {
                GpuError("Error in gpuEventSynchronize" , gpuErr);
            }
            float time;
            gpuErr = gpuEventElapsedTime(&time, _device_start_time, _device_stop_time);
            if (gpuErr) {
                GpuError("Error in gpuEventElapsedTime" , gpuErr);
            }
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

    [[nodiscard]] double microseconds() const { return _elapsed*1000; }
    [[nodiscard]] double milliseconds() const { return _elapsed; }
    [[nodiscard]] double seconds() const { return milliseconds()/1000; }
    [[nodiscard]] double minutes() const { return seconds()/60; }
    [[nodiscard]] double hours() const { return minutes()/60; }
    [[nodiscard]] double days() const { return hours()/24; }


    [[nodiscard]] std::string autoFormat() const {
        /*if(days() > 2){
            return sformat("%.0fd %.0fh %.2fmin", days(), hours(), std::fmod(minutes(),60.0));
        }
        else if(hours() > 2){
            return sformat("%.0fh %.2fmin", hours(), std::fmod(minutes(),60.0));
        }
        else if(minutes() > 2){
            return sformat("%.0fmin %.3fs", minutes(), std::fmod(seconds(),60.0));
        }
        else if(seconds() > 2){
            return sformat("%.3fs", seconds());
        }
        else if(milliseconds() > 2){
            return sformat("%.3fms", milliseconds());
        }
        else{
            return sformat("%.0fµs", microseconds());
        }*/
        return sformat("%.3fs", seconds());//ranluo
    }

    void print(const std::string& text) {
        rootLogger.info("Time for " + text + " " , autoFormat());
    }

    //! set how many bytes were processed (for an MB/s output)
    void setBytes(const long b ) { _bytes = b ; }
    //! set how many FLOPs were calculated 
    void setFlops(const long f ) { _flops = f ; } 

    //! return MBytes/s (be sure to call setBytes() before)
    [[nodiscard]] double mbs() const {
        return (_bytes / (seconds()*1024*1024));
    }

    //! return MFLOP/s (be sure to call setFlops() before)
    [[nodiscard]] double mflps() const {
        return (_flops / seconds());
    }


    //! Add two timings (bytes and flops are not transfered atm)
    template<bool _device>
        StopWatch<device> operator+(const StopWatch<_device> & rhs) const {
            StopWatch<device> ret;
            ret._elapsed = _elapsed + rhs._elapsed;
            return ret;
        }

    template<bool _device>
        StopWatch<device> & operator+=(const StopWatch<_device> & rhs) {
            _elapsed += rhs._elapsed;
            return *this ;
        }   

    //! Substract two timings (_bytes and flops are not transfered atm)
    template<bool _device>
        StopWatch<device> operator-(const StopWatch<_device> & rhs) const {
            StopWatch<device> ret;
            ret._elapsed = _elapsed - rhs._elapsed;
            return ret;
        }

    template<bool _device>
        StopWatch<device> & operator-=(const StopWatch<_device> & rhs) {
            _elapsed -= rhs._elapsed;
            return *this ;
        }   


    StopWatch<device> operator*(const int & rhs) {
        StopWatch<device> ret;
        ret._elapsed = _elapsed * rhs;
        return ret;
    }

    StopWatch<device> & operator*=(const int & rhs) {
        _elapsed *= rhs;
        return *this ;
    }   


    //! Calculate ratio of two timings
    StopWatch<device> operator/(const int & rhs) {
        StopWatch<device> ret;
        ret._elapsed = _elapsed / rhs;
        return ret;
    }

    //! Divide the timing by an integer (e.g. loop count)
    StopWatch<device> & operator/=(const int & rhs) {
        _elapsed /= rhs;
        return *this ;
    }   



    template<bool _device>
        inline friend std::ostream &operator<<(std::ostream &stream,
                const StopWatch<_device> &rhs);




};

template<bool device>
inline std::ostream &operator<<(std::ostream &stream, const StopWatch<device> &rhs) {
    stream << rhs.autoFormat();
    return stream;
}

#endif



