/*
 * latticeParameters.h
 *
 * Contains basic parameters common to all types of lattice calculation.
 *
 */

#pragma once
#include <string>
#include "../base/communication/communicationBase.h"
#include "IO/parameterManagement.h"
#include "latticeDimension.h"


class LatticeParameters : virtual public ParameterList {

    int nuller[4] = {0,0,0,0};

public:

    Parameter<int, 4> latDim;
    Parameter<int, 4> nodeDim;
    Parameter<int, 4> gpuTopo;
    Parameter<int, 1> confnumber;
    Parameter<std::string> streamName;
    Parameter<double> beta;
    Parameter<std::string> measurements_dir;
    Parameter<std::string> endianness;  //!< one of "little", "big", "auto"
    Parameter<std::string> GaugefileName;
    Parameter<std::string> GaugefileName_out;
    Parameter<std::string> EigenvectorfileName;
    Parameter<std::string> format;
    Parameter<int> prec_out;
    Parameter<bool> use_unit_conf;

    /// ILDG-specific metadata.
    Parameter<std::string> ILDGconfAuthor;
    Parameter<std::string> ILDGauthorInstitute;
    Parameter<std::string> ILDGmachineType;
    Parameter<std::string> ILDGmachineName;
    Parameter<std::string> ILDGmachineInstitute;
    Parameter<std::string> ILDGcollaboration;
    Parameter<std::string> ILDGprojectName;

    LatticeParameters() {
        add(latDim, "Lattice");
        add(nodeDim, "Nodes");
        addDefault<int, 4>(gpuTopo, "Topology", nuller);
        addOptional(beta, "beta");
        addOptional(GaugefileName, "Gaugefile");
	addOptional(EigenvectorfileName, "Eigenvectorfile");
        addOptional(GaugefileName_out, "Gaugefile_out");
        addOptional(format, "format");
        addDefault(endianness, "endianness", std::string("auto"));
        addOptional(confnumber, "conf_nr");
        addOptional(streamName, "stream");
        addDefault(prec_out, "prec_out",0);
        addDefault(measurements_dir, "measurements_dir", std::string("./"));
        addDefault(use_unit_conf, "use_unit_conf", false);

        addOptional(ILDGconfAuthor,      "ILDGconfAuthor");
        addOptional(ILDGauthorInstitute, "ILDGauthorInstitute");
        addOptional(ILDGmachineType,     "ILDGmachineType");
        addOptional(ILDGmachineName,     "ILDGmachineName");
        addOptional(ILDGmachineInstitute,"ILDGmachineInstitute");
        addOptional(ILDGcollaboration,   "ILDGcollaboration");
        addOptional(ILDGprojectName,     "ILDGprojectName");
    }

    //! Set by providing values, mainly used in test routines
    void set(LatticeDimensions local, LatticeDimensions nodes, double betaIn = 0.0) {
        beta.set(betaIn);
        GaugefileName.clear();
        latDim.set(local * nodes);
        nodeDim.set(nodes);
    }

    //! Return a ensemble extension with beta and lattice size
    virtual std::string ensembleExt() const {
        std::stringstream fext;
        fext.fill('0');
        fext << "_s" << std::setw(3) << latDim[0];
        fext << "t" << std::setw(2) << latDim[3];
        fext << "_b" << std::setw(7) << ((int) (beta() * 100000));
        return fext.str();
    }

    //! Return a file extension with beta, lattice size, and configuration number
    virtual std::string fileExt() const {
        std::stringstream fext;
        fext.fill('0');
        fext << ensembleExt();
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
