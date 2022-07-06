#include <fstream>
#include "parameterManagement.h"
#include "../communication/communicationBase.h"

bool ParameterList::readfile(const CommunicationBase& comm, const std::string& filename, int argc, char** argv) {
    std::string filecontent;
    if (comm.IamRoot()) {
        std::string fname = (argc<2)?filename:argv[1];
        std::ifstream in(fname.c_str(), std::ios::in);
        if (fname == "EMPTY_FILE"){
            rootLogger.info("Reading parameters only from command line.");
        } else {
            rootLogger.info("Reading parameters from file :: ", fname);
            if (in.fail()) {
                throw std::runtime_error(stdLogger.fatal("Unable to open parameter file ", fname));
            }
        }
        if (comm.single())
            return readstream(in,argc-2,argv+2);
        else {
            std::ostringstream content;
            content << in.rdbuf();
            filecontent = content.str();
        }
    }
    comm.root2all(filecontent);
    std::istringstream str(filecontent);
    return readstream(str,argc-2,argv+2);
}

bool ParameterList::readstream(std::istream& in, int argc, char** argv, const std::string& prefix,
                               const bool ignore_unknown) {
    std::string error_msg_suffix(" is either not a known parameter or its value could not be cast into the correct "
                                 "data type.");
    while (in.good()) {
        std::string line;
        getline(in, line);
        strpair pair(line);
        if (pair.key.empty())
            continue;
        bool found_match = false;
        for (auto & i : *this)
            if (i->match(pair)){
                found_match = true;
            }
        if (not found_match and not ignore_unknown){
            throw std::runtime_error(stdLogger.fatal(pair.key, error_msg_suffix));
        }
    }

    for (int i=0; i<argc; i++) {
        strpair pair(argv[i]);
        bool found_match = false;
        for (auto & it : *this)
            if (it->match(pair)){
                found_match = true;
            }
        if (not found_match and not ignore_unknown) {
            throw std::runtime_error(stdLogger.fatal(pair.key, error_msg_suffix));
        }
    }

    bool abort = false;
    for (auto & i : *this) {
        ParameterBase& p = *i;

        //fix that!!
        if (p.isSet())
            rootLogger.info("# " ,  prefix ,  " :: " ,  p);
        else if (p.hasdefault)
            rootLogger.info("# " ,  prefix ,  " :: " ,  p ,  " (default)");
        else if (p.isRequired())
            throw std::runtime_error(stdLogger.fatal("# ", prefix, " :: ", p, " required but NOT set"));

        if (p.isRequired() && !p.isSet()) abort = true;
    }

    if (abort) {
        throw std::runtime_error(stdLogger.fatal("Required parameters unset!"));
    }
    return true;
}

