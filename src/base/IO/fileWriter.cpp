/*
 * fileWriter.cpp
 *
 * L. Mazur
 *
 * Implementation of some methods in fileWriter.h
 *
 */

#include "fileWriter.h"

bool FileWriter::IamRoot(){
    return comm.IamRoot();
}

//! Initialize class, automatically called by constructors
void FileWriter::init(std::ios_base::openmode mode) {
    _mode =mode;
    // open output only on root node
    if (IamRoot()) {
        _ostr.open(_fname.c_str(), mode);
        // output the parameters
        if((mode & std::ios::binary) != std::ios::binary) {
            latParam.toStream(_ostr);
            // set high precision
            _ostr.precision(15);
            _ostr.setf(std::ios::scientific);
        }
    }
}

void FileWriter::createFile(std::string fname, std::ios_base::openmode mode){
    if(!_ostr.is_open()) {
        _fname = fname;
        init(mode);
    }
}

//! Output a header line
LineFormatter FileWriter::header(int prec) {
    _ostr << "#";
    return LineFormatter(_ostr, "", prec);
}

//! Output to a tag
LineFormatter FileWriter::tag(std::string tag, int prec) {
    return LineFormatter(_ostr, tag, prec);
}
