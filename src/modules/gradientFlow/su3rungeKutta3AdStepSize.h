//
// Created by Lukas Mazur on 24.11.18.
//

#ifndef SU3RUNGEKUTTA3ADSTEPSIZE_H
#define SU3RUNGEKUTTA3ADSTEPSIZE_H

#include "su3rungeKutta3.h"

template<class floatT, size_t HaloDepth, typename Zi>
class su3rungeKutta3AdStepSize : public su3rungeKutta3<floatT, HaloDepth, Zi> {

protected:

    Gaugefield<floatT, false, HaloDepth> _gaugeA_host;
    Gaugefield<floatT, false, HaloDepth> _gaugeB_host;
    // host gaugefields for the backup method for the adaptive step size
    Gaugefield<floatT, false, HaloDepth> _gaugeC_host;
    Gaugefield<floatT, false, HaloDepth> _gaugeD_host;

    LatticeContainer<true,floatT> _redBase;

    floatT _accuracy;

    typedef GIndexer<All, HaloDepth> GInd;
public:
    su3rungeKutta3AdStepSize(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
                              floatT stepSize, floatT start, floatT stop,
                             std::vector<floatT> necessaryFlowTime, floatT accuracy) :
            su3rungeKutta3<floatT, HaloDepth, Zi>(inGaugeA, stepSize, start, stop, necessaryFlowTime),
            _gaugeA_host(this->_gaugeA_device.getComm()),
            _gaugeB_host(this->_gaugeA_device.getComm()),
            _gaugeC_host(this->_gaugeA_device.getComm()),
            _gaugeD_host(this->_gaugeA_device.getComm()),
            _redBase(inGaugeA.getComm()),
            _accuracy(accuracy) {
        // get the number of elements in vol 4
        const size_t elems = GInd::getLatData().vol4;

        _redBase.adjustSize(elems);
    }

    // do a runge Kutta step and return the flow time
    floatT updateFlow();

    // release the object (destructor)
    ~su3rungeKutta3AdStepSize() {
    }
};

/*Z1Z0_adStepSize(gaugeAccessor<floatT> , gaugeAccessor<floatT> )
	compute 8/9 * Z_1 - 17/36 * Z_0 = 8/9 * this->_gaugeD_device - 17/36 * this->_gaugeA_device
	store it in this->_gaugeA_device

	compute 2 Z_1 - Z_0 = 2 * this->_gaugeD_device - this->_gaugeA_device
	store it in this->_gaugeD_device

	compute first then store
	due to the use auf the gaugefields in the computation.
*/
template<class floatT, size_t HaloDepth, bool onDevice>
struct Z1Z0_adStepSize {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> gaugeAccessorA;
    gaugeAccessor<floatT> gaugeAccessorB;

    typedef GIndexer<All, HaloDepth> GInd;

    Z1Z0_adStepSize(Gaugefield<floatT, onDevice, HaloDepth> &gaugeA,
                    Gaugefield<floatT, onDevice, HaloDepth> &gaugeB) :
            gaugeAccessorA(gaugeA.getAccessor()),
            gaugeAccessorB(gaugeB.getAccessor()) {}

    //This is the operator that is called inside the Kernel
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {

        // define two temporary GSU3 for the summation
        GSU3<floatT> Z1Z0_adStepSize_1;
        GSU3<floatT> Z1Z0_adStepSize_2;

        Z1Z0_adStepSize_1 =
                floatT(8./9.) * gaugeAccessorB.getLink(siteMu) - floatT(17./36.) * gaugeAccessorA.getLink(
                        siteMu);
        Z1Z0_adStepSize_2 =
                floatT(2.) * gaugeAccessorB.getLink(siteMu) - gaugeAccessorA.getLink(siteMu);

        // store 8/9 * Z_1 - 17/36 * Z_0 in this->_gaugeA_device
        gaugeAccessorA.setLink(siteMu, Z1Z0_adStepSize_1);

        // store 2 * Z_1 - Z_0 in this->_gaugeD_device
        //gaugeAccessorB.setLink(siteMu, Z1Z0_adStepSize_2);
        return Z1Z0_adStepSize_2;
    }
};


/*distance(gaugeAccessor<floatT>, gaugeAccessor<floatT>, floatT *, const size_t)
	compute the distance square between two GSU3 matrices:
	 	1/3^2 sum_(i,j=1)^3 |(A - B)_ij|^2
		= sum_(i,j=1)^3 real((A - B)_ij)^2 +  imag((A - B)_ij)^2
	pick the max regarding the directions
*/
template<class floatT, size_t HaloDepth, bool onDevice>
struct distance {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> Vmu;
    gaugeAccessor<floatT> W_2;

    typedef GIndexer<All, HaloDepth> GInd;

    distance(Gaugefield<floatT, onDevice, HaloDepth> &Vmu, Gaugefield<floatT, onDevice, HaloDepth> &W_2)
            : Vmu(Vmu.getAccessor()),
              W_2(W_2.getAccessor()) {}

    //This is the operator that is called inside the Kernel
    __device__ __host__ inline floatT operator()(gSite site) {
        // define two discance variables to optain the biggest of all directions
        floatT distOld = 0;
        floatT distNew;

        GSU3<floatT> subGSU3;

        // loopover all directions
        for (int mu = 0; mu < 4; mu++) {
            // set distNew to 0 for the += notation
            distNew = 0;

            // compute Vmu - W_2
            subGSU3 = Vmu.getLink(GInd::getSiteMu(site, mu)) - W_2.getLink(GInd::getSiteMu(site, mu));

            // sum over all matrix elements
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    // compute Real^2 + Imag^2 and sum it
                    distNew += real(subGSU3(i, j)) * real(subGSU3(i, j)) + imag(subGSU3(i, j)) * imag(subGSU3(i, j));
                }
            }

            // choose the biggest distance
            if (distNew > distOld) { distOld = distNew; }
        }

        // norm (*1/9) the chosen distance and write it into the array
        return floatT(1./81.) * distOld;
    }
};


template<class floatT, size_t HaloDepth, typename Zi>
floatT su3rungeKutta3AdStepSize<floatT, HaloDepth, Zi>::updateFlow() {

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

            // copy original configuration to backup
            _gaugeA_host = this->_gaugeA_device;
            _gaugeC_host = _gaugeA_host;
            // compute Z_0
            this->_gaugeB_device.iterateOverBulkAllMu(Zi(this->_gaugeA_device, thisStepSize));
            // store Z_0 * (this->_gaugeB_device) in BackupTwo
            _gaugeB_host = this->_gaugeB_device;
            _gaugeD_host = _gaugeB_host;
            // compute W_1
            this->_gaugeA_device.iterateOverBulkAllMu(
                    W_1<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeB_device));

            this->_gaugeA_device.updateAll();
            this->_gaugeB_device.updateAll();
            // compute Z_1
            this->_gaugeB_device.iterateOverBulkAllMu(Zi(this->_gaugeA_device, thisStepSize));
            // swap W1 (gauge A) with Z_0 BackupTwo
            _gaugeA_host = this->_gaugeA_device;
            this->_gaugeA_device = _gaugeD_host;
            // compute c_1 * Z_1 - c_0 *Z_0
            this->_gaugeB_device.iterateOverBulkAllMu(
                    Z1Z0_adStepSize<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeB_device));
            // swap 8/9 * Z_1 - 17/36 * Z_0 with BackupOne (V_mu(x,t))
            _gaugeD_host = this->_gaugeA_device;
            this->_gaugeA_device = _gaugeC_host;
            // compute VBar_mu(c,t+h)
            this->_gaugeB_device.iterateOverBulkAllMu(
                    W_2<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeB_device));
            // swap VBar_mu(c,t+h) with BackupTwo (W_1)
            _gaugeB_host = this->_gaugeB_device;
            this->_gaugeB_device = _gaugeA_host;
            // swap V_mu(x,t) with 8/9 * Z_1 - 17/36 * Z_0
            this->_gaugeA_device = _gaugeD_host;
            // compute W_2
            this->_gaugeB_device.iterateOverBulkAllMu(
                    W_2<floatT, HaloDepth, true>(this->_gaugeB_device, this->_gaugeA_device));

            this->_gaugeA_device.updateAll();
            this->_gaugeB_device.updateAll();
            // compute V_mu(x,t+h)
            this->_gaugeA_device.iterateOverBulkAllMu(Vmu<floatT, HaloDepth, true, Zi>
                                                              (this->_gaugeB_device, this->_gaugeA_device,
                                                               Zi(this->_gaugeB_device, thisStepSize),
                                                               thisStepSize));
            this->_gaugeA_device.updateAll();
            // swap this->_gaugeB_device (W1) with BackupTwo (VBar_mu)
            this->_gaugeB_device = _gaugeB_host;
            _redBase.template iterateOverBulk<All, HaloDepth>(
                    distance<floatT, HaloDepth, true>(this->_gaugeA_device, this->_gaugeB_device));
            _redBase.reduceMax(maxDistsq, GIndexer<All, HaloDepth>::getLatData().vol4, true);

        // compute the step size for the next step
        this->_step_size *= floatT(0.95) * pow(_accuracy / (sqrt(maxDistsq)), floatT(1./3.));

        // check if maxDist ist smaller then the tolerance and accept or refuse
        if (sqrt(maxDistsq) > _accuracy) {
            // refuse
            rootLogger.info(CoutColors::magenta ,  "Reject step size " ,  thisStepSize ,  " at flow time " ,  this->_current_flow_time
                              ,  ", try again with step size " ,  this->_step_size ,  CoutColors::reset);
            //restart with original gaugefield
            this->_gaugeA_device = _gaugeC_host;
        }
        else {
            //accept
            // add the step to the flow time
            this->_current_flow_time += thisStepSize;
            return thisStepSize;
        }
    }
}



#endif //SU3RUNGEKUTTA3ADSTEPSIZE_H

