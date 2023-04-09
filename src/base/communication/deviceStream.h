/* 
 * deviceStream.h                                                               
 * 
 * L. Mazur 
 * 
 */

#ifndef DEVICESTREAM_H
#define DEVICESTREAM_H

#ifndef USE_CPU_ONLY

#include "../../define.h"
#include "../gutils.h"

template<bool onDevice>
class deviceStream {

public:
    gpuStream_t _stream;

    bool _initialized;
    int _mode;

    deviceStream(deviceStream &obj) = delete;

    deviceStream(deviceStream &&obj) {
        _stream = obj._stream;
        _mode = obj._mode;
        _initialized = obj._initialized;
        obj._initialized = false;
    }

    deviceStream &operator=(deviceStream &&obj) {
        _stream = obj._stream;
        _mode = obj._mode;
        _initialized = obj._initialized;
        obj._initialized = false;
        return *this;
    }


    void init() {
        if (_mode == 1 && onDevice) {
            gpuError_t gpuErr = gpuStreamCreate(&_stream);
            if (gpuErr != gpuSuccess) GpuError("deviceStream.h: deviceStream(0): gpuStreamCreate", gpuErr);
        } else _stream = 0;
        _initialized = true;

    }

    deviceStream(int mode = 1) : _initialized(false), _mode(mode) { // Or put mode >= 3, to initialize nothing!
        init();
    }


    ~deviceStream() {
        if (_initialized && (_mode == 1) && onDevice) {
            gpuError_t gpuErr = gpuStreamDestroy(_stream);
            if (gpuErr != gpuSuccess) GpuError("deviceStream.h: ~deviceStream(): gpuStreamDestroy", gpuErr);
        }
        _initialized = false;
    }

    void synchronize() {
        if (!_initialized) rootLogger.error("deviceStream.h: synchronize: stream not initialized");
        if (_mode == 1 && onDevice) {
            gpuError_t gpuErr = gpuStreamSynchronize(_stream);
            if (gpuErr != gpuSuccess)
                GpuError("deviceStream.h: deviceStream.synchronize: gpuStreamSynchronize(_stream)", gpuErr);
        }
    }

};

#endif
#endif //DEVICEEVENT_H
