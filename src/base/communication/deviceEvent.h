/*
 * deviceEvent.h
 *
 * L. Mazur
 *
 */

#ifndef DEVICEEVENT_H
#define DEVICEEVENT_H

#include "../../define.h"
#include "../gutils.h"
#include "../wrapper/gpu_wrapper.h"

class deviceEvent {
public:
    gpuEvent_t _event;

    bool _initialized;
    int _mode;

    deviceEvent(int mode) : _event(0), _initialized(false), _mode(mode) { // Or put mode >= 3, to initialize nothing!
        init();
    }

    //! copy constructor
    deviceEvent(deviceEvent& obj) = delete;

    //! copy assignment
    deviceEvent& operator=(deviceEvent& obj) = delete;

    //! move constructor
    deviceEvent(deviceEvent &&obj) :
    _event(obj._event),
    _initialized(obj._initialized),
    _mode(obj._mode)
    {
        obj._initialized = false;
        obj._event = gpuEventDefault;
        obj._mode = 999;
    }

    //! move assignment
    deviceEvent &operator=(deviceEvent &&obj) {
        _event = obj._event;
        _initialized = obj._initialized;
        _mode = obj._mode;

        obj._initialized = false;
        obj._event = gpuEventDefault;
        obj._mode = 999;
        return *this;
    }

    ~deviceEvent() {
        if (_initialized && (_mode == 0 || _mode == 1 || _mode == 2)) {
            gpuError_t gpuErr = gpuEventDestroy(_event);
            if (gpuErr != gpuSuccess) GpuError("deviceEvent.h: ~deviceEvent(): gpuEventDestroy", gpuErr);
        }
        _initialized = false;
    }

    void init() {
        if (_mode == 0) {
            gpuError_t gpuErr = gpuEventCreate(&_event);
            if (gpuErr != gpuSuccess) GpuError("deviceEvent.h: deviceEvent(0): gpuEventCreate", gpuErr);
        } else if (_mode == 1) {
            gpuError_t gpuErr = gpuEventCreateWithFlags(&_event, gpuEventDisableTiming);
            if (gpuErr != gpuSuccess) GpuError("deviceEvent.h: deviceEvent(1): gpuEventCreate", gpuErr);
        } else if (_mode == 2) {
            gpuError_t gpuErr = gpuEventCreateWithFlags(&_event, gpuEventDisableTiming | gpuEventInterprocess);
            if (gpuErr != gpuSuccess) GpuError("deviceEvent.h: deviceEvent(2): gpuEventCreate", gpuErr);
        }
        _initialized = true;

    }

    void synchronize() {
        if (!_initialized) rootLogger.error("deviceEvent.h: synchronize: event not initialized");
        gpuError_t gpuErr = gpuEventSynchronize(_event);
        if (gpuErr != gpuSuccess)
            GpuError("deviceEvent.h: deviceEvent.synchronize: gpuEventSynchronize(_event)", gpuErr);
    }

    void record(gpuStream_t stream) {
        if (!_initialized) rootLogger.error("deviceEvent.h: record: event not initialized");
        gpuError_t gpuErr = gpuEventRecord(_event, stream);
        if (gpuErr != gpuSuccess) GpuError("deviceEvent.h: deviceEvent.record(): gpuEventRecord", gpuErr);
    }

    void streamWaitForMe(gpuStream_t stream) {
        if (!_initialized) rootLogger.error("deviceEvent.h: streamWaitForMe: event not initialized");
        gpuError_t gpuErr = gpuStreamWaitEvent(stream, _event, 0);
        if (gpuErr != gpuSuccess)
            GpuError("deviceEvent.h: deviceEvent.streamWaitForMe(): gpuStreamWaitEvent", gpuErr);
    }

    bool query() {
        if (!_initialized) rootLogger.error("deviceEvent.h: query: event not initialized");
        gpuError_t gpuErr = gpuEventQuery(_event);
        if (gpuErr == gpuSuccess) return true;
        else return false;
    }
};

#endif //DEVICEEVENT_H
