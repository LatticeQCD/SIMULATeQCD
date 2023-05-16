//
// Created by Lukas Mazur on 24.11.18.
//

#pragma once
#include "su3rungeKutta3.h"
#include "su3rungeKutta3AdStepSize.h"

template<class floatT, size_t HaloDepth, typename Zi>
class su3rungeKutta3AdStepSizeAllGPU : public su3rungeKutta3<floatT, HaloDepth, Zi> {

protected:

    // device gaugefields for the backup method for the adaptive step all on gpu
    Gaugefield<floatT, true, HaloDepth> _gaugeC_device;
    Gaugefield<floatT, true, HaloDepth> _gaugeD_device;

    LatticeContainer<true,floatT> _redBase;

    floatT _accuracy;

    typedef GIndexer<All, HaloDepth> GInd;
public:
    su3rungeKutta3AdStepSizeAllGPU(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
                              floatT stepSize, floatT start, floatT stop,
                             std::vector<floatT> necessaryFlowTime, floatT accuracy) :
            su3rungeKutta3<floatT, HaloDepth, Zi>(inGaugeA, stepSize, start, stop, necessaryFlowTime),
            _gaugeC_device(this->_gaugeA_device.getComm()),
            _gaugeD_device(this->_gaugeA_device.getComm()),
            _redBase(inGaugeA.getComm()),
            _accuracy(accuracy){
        // get the number of elements in vol 4
        const size_t elems = GInd::getLatData().vol4;

        _redBase.adjustSize(elems);
    }

    // do a runge Kutta step and return the flow time
    floatT updateFlow();

    // release the object (destructor)
    ~su3rungeKutta3AdStepSizeAllGPU() {
    }
};

template<class floatT, size_t HaloDepth, typename Zi>
floatT su3rungeKutta3AdStepSizeAllGPU<floatT, HaloDepth, Zi>::updateFlow() {

    floatT thisStepSize;

    // exchange halo before start
    this->_gaugeA_device.updateAll();
    floatT maxDistsq = 0.;

    //do until a good flow occurs:
    while (true) {
        // Check if all neccessary flowtimes are reached
        this->checkIfNextTimeIsNessTime();

        // get the Flowtime for this Runge Kutta step
        thisStepSize = this->_step_size;

            // compute Z_0
            this->_gaugeB_device.iterateOverBulkAllMu(Zi(this->_gaugeA_device, thisStepSize));
            // compute W_1S
            this->_gaugeC_device.iterateOverBulkAllMu(
                    W_1<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeB_device));

            this->_gaugeB_device.updateAll();
            this->_gaugeC_device.updateAll();

            // compute Z_1
            this->_gaugeD_device.iterateOverBulkAllMu(Zi(this->_gaugeC_device, thisStepSize));
            // compute c_1 * Z_1 - c_0 *Z_0
            this->_gaugeD_device.iterateOverBulkAllMu(
                    Z1Z0_adStepSize<floatT, HaloDepth, true>(this->_gaugeB_device, this->_gaugeD_device));
            // compute VBar_mu(c,t+h)
            this->_gaugeD_device.iterateOverBulkAllMu(
                    W_2<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeD_device));
            // compute W_2
            this->_gaugeC_device.iterateOverBulkAllMu(
                    W_2<floatT, HaloDepth, true>(this->_gaugeC_device, this->_gaugeB_device));

            this->_gaugeC_device.updateAll();
            this->_gaugeD_device.updateAll();
            // compute V_mu(x,t+h)
            this->_gaugeB_device.iterateOverBulkAllMu(Vmu<floatT, HaloDepth, true, Zi>
                                                              (this->_gaugeC_device, this->_gaugeB_device,
                                                               Zi(this->_gaugeC_device, thisStepSize),
                                                               thisStepSize));
            this->_gaugeB_device.updateAll();

            _redBase.template iterateOverBulk<All, HaloDepth>(
                    distance<floatT, HaloDepth, true>(this->_gaugeB_device, this->_gaugeD_device));
            _redBase.reduceMax(maxDistsq, GIndexer<All, HaloDepth>::getLatData().vol4, true);

            //place gaugefields where they need to be
            this->_gaugeA_device.swap_memory(this->_gaugeB_device);
            this->_gaugeB_device.swap_memory(this->_gaugeD_device);

        // compute the step size for the next step
        this->_step_size *= floatT(0.95) * pow(_accuracy / (sqrt(maxDistsq)), floatT(1./3.));

        // check if maxDist ist smaller then the tolerance and accept or refuse
        if (sqrt(maxDistsq) > _accuracy) {
            // refuse
            rootLogger.info(CoutColors::magenta ,  "Reject step size " ,  thisStepSize ,  " at flow time " ,  this->_current_flow_time
                              ,  ", try again with step size " ,  this->_step_size ,  CoutColors::reset);
            //restart with original gaugefield
            this->_gaugeA_device.swap_memory(this->_gaugeD_device);

        }
        else {
            //accept
            // add the step to the flow time
            this->_current_flow_time += thisStepSize;
            return thisStepSize;
        }
    }
}


