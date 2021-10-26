
#ifndef _INC_LATTICE_PARAMETERS
#define _INC_LATTICE_PARAMETERS

#include <string>

#include "../base/communication/communicationBase.h"
#include "IO/parameterManagement.h"
#include "LatticeDimension.h"



//! Class for basic lattice parameters
class LatticeParameters : virtual public ParameterList {
    int nuller[4] = {0,0,0,0};
public:
    Parameter<int, 4> latDim;
    Parameter<int, 4> nodeDim;
    Parameter<int, 4> gpuTopo;
    Parameter<int, 1> confnumber;
    Parameter<std::string> streamName;
    Parameter<double> beta;
    Parameter<std::string> GaugefileName;
    Parameter<std::string> format;
    Parameter<std::string> endianness;  //!< one of "little", "big", "auto"
    LatticeParameters() {
        add(latDim, "Lattice");
        add(nodeDim, "Nodes");
        addDefault<int, 4>(gpuTopo, "Topology", nuller);
        addOptional(beta, "beta");
        addOptional(GaugefileName, "Gaugefile");
        addOptional(format, "format");
        addDefault(endianness, "endianness", std::string("auto"));
        addOptional(confnumber, "conf_nr");
        addOptional(streamName, "stream");
    }

    //! Set by providing values, mainly used in test routines
    void set(LatticeDimensions local, LatticeDimensions nodes, double betaIn = 0.0) {
        beta.set(betaIn);
        GaugefileName.clear();
        latDim.set(local * nodes);
        nodeDim.set(nodes);
    }

    //! Return a file extension with beta and lattice size
    virtual std::string fileExt() const {
        std::stringstream fext;
        fext.fill('0');
        fext << "_s" << std::setw(3) << latDim[0];
        fext << "t" << std::setw(2) << latDim[3];
        fext << "_b" << std::setw(7) << ((int) (beta() * 100000));
        if (streamName.isSet())
            fext << "_" << streamName();
        if (confnumber.isSet())
            fext << "_U" << std::setw(9) << (confnumber());
        return fext.str();
    }
};


//! Class with parameters for diffusion code, additionally to base parameters
class DiffusionParameters : public LatticeParameters {
public:
    //! The number of updates to compute
    Parameter<int> sl_updates;
    Parameter<int> linkIntHBupdates;

    DiffusionParameters() {
        add(sl_updates, "sl_updates");
        add(linkIntHBupdates, "linkInt_updates");
    };
};


//! part of a parameter list that just adds an optional random number seed parameter
class SeedParam : virtual public ParameterList {
    Parameter<int> seed;
public:
    SeedParam() {
        addOptional(seed, "seed");
    }

    //! returns a random number seed given as an option or, if not provided, from the current time
    int get_seed() const {
        if (seed.isSet())
            return seed();
        else
            return time(NULL);
    }
};


#endif
