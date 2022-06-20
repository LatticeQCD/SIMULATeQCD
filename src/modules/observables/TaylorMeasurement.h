#ifndef TAYLORMEASUREMENT_H
#define TAYLORMEASUREMENT_H

#include <string>
#include "../base/communication/communicationBase.h"
#include "IO/parameterManagement.h"
#include "LatticeDimension.h"
#include "latticeParameters.h"

template<class floatT>
struct TaylorMeasurementParameters : LatticeParameters {
{
    // TODO not sure yet what belongs here
    DynamicParameter<int> operator_ids;
    DynamicParameter<floatT> valence_masses;
    Parameter<int> num_random_vectors;
    // I will also need residue_meas, CGmax, residue from LatticeParameters

    TaylorMeasurementParameters() {
        add(operator_ids, "OperatorIds");
        add(valence_masses, "ValenceMasses");
        add(num_random_vectors, "NumberRandomVectors");
    }
};

class TaylorMeasurement
{
public:
    TaylorMeasurement();
};

#endif // TAYLORMEASUREMENT_H
