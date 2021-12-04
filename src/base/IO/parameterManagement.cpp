#include <fstream>
#include "parameterManagement.h"
#include "../communication/communicationBase.h"

bool ParameterList::readfile(const CommunicationBase& comm, const std::string& filename, int argc, char** argv) {
    std::string filecontent;
    if (comm.IamRoot()) {
        std::string fname = (argc<2)?filename:argv[1];
        rootLogger.info("Reading parameters from file :: ", fname);
        std::ifstream in(fname.c_str(), std::ios::in);
        if (in.fail()) {
            throw std::runtime_error(stdLogger.fatal("Unable to open parameter file!"));
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

bool ParameterList::readstream(std::istream& in, int argc, char** argv, const std::string& prefix) {
    while (in.good()) {
        std::string line;
        getline(in, line);
        strpair pair(line);
        if (pair.key.empty())
            continue;
        for (auto & i : *this)
            i->match(pair);
    }

    for (int i=0; i<argc; i++) {
        strpair pair(argv[i]);
        for (auto & it : *this)
            it->match(pair);
    }

    bool abort = false;
    for (auto & i : *this) {
        ParameterBase& p = *i;

        //fix that!!
        if (p.isSet())
            rootLogger.info("# " ,  prefix ,  " :: " ,  p.name);
        else if (p.hasdefault)
            rootLogger.info("# " ,  prefix ,  " :: " ,  p.name ,  " (default)");
        else if (p.isRequired())
            throw std::runtime_error(stdLogger.fatal("# ", prefix, " :: ", p.name, " required but NOT set"));

        if (p.isRequired() && !p.isSet()) abort = true;
    }

    if (abort) {
        throw std::runtime_error(stdLogger.fatal("Required parameters unset!"));
    }
    return true;
}


