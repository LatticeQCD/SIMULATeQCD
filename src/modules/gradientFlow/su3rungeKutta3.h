//
// Created by Lukas Mazur on 17.11.18.
//

#ifndef SU3_RUNGEKUTTA3_H
#define SU3_RUNGEKUTTA3_H

#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../../gauge/GaugeAction.h"
#include "../../base/IO/fileWriter.h"
#include "../../base/IO/parameterManagement.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include "../../base/math/floatComparison.h"
#include "../../gauge/gaugeActionDeriv.h"
#include "../../base/LatticeContainer.h"
#include <algorithm>    // std::sort
#include <cmath>
#include <utility>
#include "../../base/math/su3Exp.h"

//define tolerance
#if SINGLEPREC
#define TOLERANCE 1e-5
#else
#define TOLERANCE 1e-9
#endif

template<class floatT, size_t HaloDepth, typename Zi>
class su3rungeKutta3 {

protected:
    const floatT tolerance = TOLERANCE;

    // Store the actual flow time
    floatT _current_flow_time;
    floatT _stop;

    bool _maxTimeReached;

    floatT _step_size;
    floatT _start_step_size;

    // Pointer to the gaugefields A and B for the computation
    Gaugefield<floatT, true, HaloDepth> &  _gaugeA_device;
    Gaugefield<floatT, true, HaloDepth> _gaugeB_device;

    std::vector <floatT> _necessary_flow_times;
public:
    su3rungeKutta3(Gaugefield<floatT, true, HaloDepth> &inGaugeA, floatT stepSize, floatT start, floatT stop,
                   std::vector <floatT> necessaryFlowTime, floatT dummy_accuracy = 0) :
            _gaugeA_device(inGaugeA),
            _gaugeB_device(inGaugeA.getComm()) {

        // Start with flow time 0
        _current_flow_time = start;
        _stop = stop;
        // Set the start stepSize
        _start_step_size = stepSize;
        _step_size = stepSize;

        _gaugeB_device = _gaugeA_device;

        _necessary_flow_times = necessaryFlowTime;

        std::sort(_necessary_flow_times.begin(), _necessary_flow_times.end());

        _maxTimeReached = false;

    }

    // Check if the max flow time is reached in that case return true
    bool continueFlow() {

        floatT difference = _current_flow_time + _step_size - _stop;

        _maxTimeReached = isApproximatelyZero(_current_flow_time - _stop, tolerance);
        bool _maxTimePassed = isDefinitelyGreaterThan(difference, floatT(0.), tolerance);

        if (_maxTimeReached) {
            return false;
        }
        else if (_maxTimePassed) {
            _step_size = _step_size - difference;

            return true;
        }
        return true;
    }

    // Do a runge Kutta step and return the flow time
    floatT updateFlow();

    bool checkIfnecessaryTime() {
        return isApproximatelyEqual(_current_flow_time,_necessary_flow_times[0],tolerance);
    }

    bool checkIfEndTime() {
        return isApproximatelyEqual(_current_flow_time,_stop,tolerance);
    }

    // If the next flow time is bigger than the  necessary flow time, compute a new flow time so that the next
    // wilson flow step computes the smeared field up to the necessary flow time
    bool checkIfNextTimeIsNessTime() {
        if (_necessary_flow_times.size() > 0) {
            if(isDefinitelyGreaterThan(_current_flow_time,_necessary_flow_times[0], tolerance) || isApproximatelyEqual(_current_flow_time,_necessary_flow_times[0], tolerance))
                _necessary_flow_times.erase(_necessary_flow_times.begin());

            if (_necessary_flow_times.size() > 0 && isDefinitelyGreaterThan(_current_flow_time + _step_size, _necessary_flow_times[0], tolerance)) {
                _step_size = _necessary_flow_times[0] - _current_flow_time;

                return true;
            }

        }
        return false;
    };

    ~su3rungeKutta3() {
    }
};


/*W_1(gaugeAccessor<floatT>, gaugeAccessor<floatT>)
Determine
	W_1 = exp(1/4 * Z_0) * V_mu(x,t)
*/
template<class floatT, size_t HaloDepth, bool onDevice>
struct W_1 {
    // Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gaugeAccessorA;
    gaugeAccessor<floatT> gaugeAccessorB;

    typedef GIndexer<All, HaloDepth> GInd;

    W_1(Gaugefield<floatT, onDevice, HaloDepth> &gaugeA, Gaugefield<floatT, onDevice, HaloDepth> &gaugeB) :
            gaugeAccessorA(gaugeA.getAccessor()),
            gaugeAccessorB(gaugeB.getAccessor()) {}

    // This is the operator that is called inside the Kernel
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {
        // Define a temporary GSU3 matrix for the exp() calculation
        GSU3<floatT> W1;

        // Determine the W1 = exp(1/4 Z_0) * W_0
        SU3Exp(floatT(0.25) * gaugeAccessorB.getLink(siteMu), W1);
        W1 = W1 * gaugeAccessorA.getLink(siteMu);

        // Store the W1 in Gaugefield A
        return W1;
    }
};


/*Z1Z0(gaugeAccessor<floatT>, gaugeAccessor<floatT>, floatT )
Determine
	 8/9 Z_1 - 17/36 Z_0
with Z_1 = h * Z(W_1)
*/
template<class floatT, size_t HaloDepth, bool onDevice, typename Force>
struct Z1Z0 {
    // Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gaugeAccessorA;
    gaugeAccessor<floatT> gaugeAccessorB;
    Force Z_i;
    floatT _stepSize;

    typedef GIndexer<All, HaloDepth> GInd;

    Z1Z0(Gaugefield<floatT, onDevice, HaloDepth> &gaugeA, Gaugefield<floatT, onDevice, HaloDepth> &gaugeB,
         Force Zi, floatT stepSize) :
            gaugeAccessorA(gaugeA.getAccessor()),
            gaugeAccessorB(gaugeB.getAccessor()),
            Z_i(Zi),
            _stepSize(stepSize) {}

    // This is the operator that is called inside the Kernel
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {

        // Define a temporary GSU3 matrix for the Z_1 calculation
        GSU3<floatT> tmpGSU3Z1;

        // Define a temporary GSU3 matrix for the subtraction
        GSU3<floatT> tmpGSU3s;

        // Determine the Z_1 = h * Z( W_1 )
        tmpGSU3Z1 = Z_i(siteMu);

        // Determine 8/9 Z_1 - 17/36 Z_0
        tmpGSU3s = floatT(8./9.) * tmpGSU3Z1 - floatT(17./36.) * gaugeAccessorB.getLink(siteMu);

        return tmpGSU3s;
    }
};


/*W_2(gaugeAccessor<floatT>, gaugeAccessor<floatT>)
Determine
	W_2 = exp(8/9 Z_1 - 17/36 Z_0)
*/
template<class floatT, size_t HaloDepth, bool onDevice>
struct W_2 {
    // Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gaugeAccessorA;
    gaugeAccessor<floatT> gaugeAccessorB;

    typedef GIndexer<All, HaloDepth> GInd;

    W_2(Gaugefield<floatT, onDevice, HaloDepth> &gaugeA, Gaugefield<floatT, onDevice, HaloDepth> &gaugeB) :
            gaugeAccessorA(gaugeA.getAccessor()),
            gaugeAccessorB(gaugeB.getAccessor()) {}

    // This is the operator that is called inside the Kernel
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {
        // Define a temporary GSU3 matrix for the exp() calculation
        GSU3<floatT> W2;

        // Determine the W_2 = exp(8/9 Z_1 - 17/36 Z_0) * W_1
        SU3Exp(gaugeAccessorB.getLink(siteMu), W2);
        W2 = W2 * gaugeAccessorA.getLink(siteMu);

        return W2;
    }
};

/*Vmu( gaugeAccessor<floatT>, gaugeAccessor<floatT>, floatT )
Determine
	V_mu(x,t) = exp(3/4 Z_2 - 8/9 Z_1 + 17/36 Z_0) * W_2
with Z_2 = h Z(W_2)
*/
template<class floatT, size_t HaloDepth, bool onDevice, typename Force>
struct Vmu {
    // Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gaugeAccessorA;
    gaugeAccessor<floatT> gaugeAccessorB;
    Force Z_i;
    floatT _stepSize;

    typedef GIndexer<All, HaloDepth> GInd;

    Vmu(Gaugefield<floatT, onDevice, HaloDepth> &gaugeA, Gaugefield<floatT, onDevice, HaloDepth> &gaugeB,
        Force Zi, floatT stepSize) :
            gaugeAccessorA(gaugeA.getAccessor()),
            gaugeAccessorB(gaugeB.getAccessor()),
            Z_i(Zi),
            _stepSize(stepSize) {}

    // This is the operator that is called inside the Kernel
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {
        // Define a temporary GSU3 matrix for the Z_1 calculation
        GSU3<floatT> tmpGSU3Z2;

        // Define a temporary GSU3 matrix for the sum
        GSU3<floatT> tmpGSU3s;

        // Determine the Z_2
        tmpGSU3Z2 = Z_i(siteMu);

        // Determine 3/2 * Z_2 - (8/9 * Z_0 - 17/36 * Z_1) * W_2
        SU3Exp(floatT(3./4.) * tmpGSU3Z2 - gaugeAccessorB.getLink(siteMu), tmpGSU3s);
        tmpGSU3s = tmpGSU3s * gaugeAccessorA.getLink(siteMu);

        // Store the substrection in Gaugefield B
        return tmpGSU3s;
    }
};


template<class floatT, size_t HaloDepth, typename Zi>
floatT su3rungeKutta3<floatT, HaloDepth, Zi>::updateFlow() {

    // This can (temporarily) reduce the fixed stepsize in order to meet necessary flow times!
    checkIfNextTimeIsNessTime();

    // Get the Flowtime for this Runge Kutta step
    floatT thisStepSize = _step_size;

    // Exchange halo before start
    _gaugeA_device.updateAll();

    // Determine Z_0 = h * Z(W_0) = h * Z(V_mu(x,t))
    _gaugeB_device.iterateOverBulkAllMu(Zi(_gaugeA_device, thisStepSize));

    // Determine W_1
    _gaugeA_device.iterateOverBulkAllMu(W_1<floatT, HaloDepth, true>(_gaugeA_device, _gaugeB_device));

    // Exchange halos
    _gaugeA_device.updateAll();
    _gaugeB_device.updateAll();

    // Determine 8/9 * Z_1 - 17/36 * Z_0
    _gaugeB_device.iterateOverBulkAllMu(Z1Z0<floatT, HaloDepth, true, Zi>
                                                (_gaugeA_device, _gaugeB_device, Zi(_gaugeA_device, thisStepSize),
                                                 thisStepSize));

    // Determine W_2
    _gaugeA_device.iterateOverBulkAllMu(W_2<floatT, HaloDepth, true>(_gaugeA_device, _gaugeB_device));

    // Exchange halos
    _gaugeA_device.updateAll();
    _gaugeB_device.updateAll();

    _gaugeB_device.iterateOverBulkAllMu(Vmu<floatT, HaloDepth, true, Zi>
                                                (_gaugeA_device, _gaugeB_device, Zi(_gaugeA_device, thisStepSize),
                                                 thisStepSize));

    _gaugeB_device.updateAll();

    // For further computation we normalize gauge A as the output field
    // so swap the memory the gauge pointed to. W_2 is not in need anymore.
    _gaugeA_device.swap_memory(_gaugeB_device);

    // Add the actual flow time for the next computation
    _current_flow_time += thisStepSize;
    // Reset step size in case it had to be reduced for necessary flow time
    _step_size = _start_step_size;

    return thisStepSize;
}

#endif //SU3_RUNGEKUTTA3_H
