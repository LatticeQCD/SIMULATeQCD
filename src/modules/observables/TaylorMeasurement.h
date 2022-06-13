#ifndef TAYLORMEASUREMENT_H
#define TAYLORMEASUREMENT_H

#include <string>
#include "../base/communication/communicationBase.h"
#include "IO/parameterManagement.h"
#include "LatticeDimension.h"
#include "latticeParameters.h"

template<class floatT>
struct gradientFlowParam : LatticeParameters {
{
    // TODO not sure yet what belongs here
    DynamicParameter<int> operatorIds;
    DynamicParameter<int> masses;
    Parameter<int> max_num_iter;
    Parameter<std::string> results_out;

    TaylorMeasurementParameters() {
        add(operatorIds, "OperatorIds");
        add(masses, "Masses");
        add(max_num_iter, "Iterations");
        add(result_out, "OutputFile");
    }
};

class TaylorMeasurement
{
public:
    TaylorMeasurement();
};

#endif // TAYLORMEASUREMENT_H
