/*
 * fileWriter.h
 *
 * L. Mazur
 *
 * This header file defines objects used to output results to file. The to-be-outputted data are stored in
 * LineFormatter objects, which are essentially ostream objects. They automatically end the line whenever they are
 * destructed; for example they will end lines automatically inside a loop. At the end of each line there is appended
 * a 'tag', which is '#' followed by a string. Tags allow the user to append a label to a line to make for easy
 * searching later with grep or shell scripts.
 *
 * The other object is the FileWriter object, which opens the output file stream, and closes it automatically when it
 * is destroyed. The FileWriter is the object that gives the ostream; this ostream can be passed to the LineFormatter
 * for data output using, for example, FileWriter's header() method.
 *
 */

#ifndef _fileWriter_h_
#define _fileWriter_h_

#include <fstream>
#include "../communication/communicationBase.h"
#include "../latticeParameters.h"
#include "misc.h"

class CommunicationBase;
class LatticeParameters;

//! Get an object that writes a line postfixed with a TAG to a FileWriter
class LineFormatter {

private:
    const std::string _tag; //!< The tag to use
    std::ostream &_ostr;    //!< The output stream to use
    const int fieldwidth;
    int _save_prec;         //!< Output stream precision (saved here)
    bool endl;

public:

    //! Constructor, set the tag and ostream
    LineFormatter(std::ostream &ostr, std::string tag, int prec = 7, bool space = true) :
            _tag(tag), _ostr(ostr), fieldwidth(prec + 8), endl(false) {
        // Insert a space to match formatting with tagged lines
        if (space) _ostr << " ";
        _save_prec = _ostr.precision();
        _ostr.precision(prec);
    };

    //! Output something formatted
    template<typename T>
    LineFormatter & operator<<(const T &obj) {
        _ostr << std::setw(fieldwidth) << obj;
        return *this;
    }

    //! Output something formatted
    template<typename T>
    LineFormatter & operator<<(const GCOMPLEX(T) &obj) {
        _ostr << " ( ";
        _ostr << std::setw(fieldwidth) << obj.cREAL;
        _ostr << " " << std::setw(fieldwidth) << obj.cIMAG;
        _ostr << " ) ";
        return *this;
    }

    void endLine(){
        _ostr << " #" << _tag << std::endl;
        _ostr.precision(_save_prec);
        endl = true;
    }

    //! Destructor, insert tag and a newline
    ~LineFormatter() { if(!endl) endLine(); }
};

//! Forward declaration for friend in FileWriter
//class TaggedLine;

//! Writes lattice computation results to file on root node only
class FileWriter {

private:
    const CommunicationBase &comm;
    const LatticeParameters &latParam;
    std::string _fname;

    std::ofstream _ostr;
    std::ios_base::openmode _mode;

    //! initializes class, automatically called by constructors
    void init(std::ios_base::openmode mode);

//    friend class TaggedLine;

public:

    //! Initialize with comm base and parameter set
    FileWriter(const CommunicationBase &comm, const LatticeParameters &lp) : comm(comm), latParam(lp) {};

    //! Initialize with comm base and parameter set
    FileWriter(const CommunicationBase &comm, const LatticeParameters &lp, std::string fname,
               std::ios_base::openmode mode = std::ios_base::out) :
            comm(comm), latParam(lp), _fname(fname) {
        init(mode);
    };

    //! Close the ostream
    ~FileWriter() { if(_ostr.is_open())_ostr.close(); }

    //! Provide FileStream a << "Text" / [something] operator
    template<class T>
    FileWriter &operator<<(const T &obj) {
        if (IamRoot())
            _ostr << obj;
        return *this;
    }

    template<typename T>
    void write_binary(const T& buffer, size_t count){
        if (IamRoot()) {
            if (switch_endianness(ENDIAN_LITTLE))
                for (size_t i = 0; i < count; i++) {
                    Byte_swap(buffer[i]);
                }
            _ostr.write(reinterpret_cast<const char *>(buffer), sizeof(T) * count);
        }
    }

    void createFile(std::string fname, std::ios_base::openmode mode = std::ios_base::out);

    //! Output a header line
    LineFormatter header(int prec = 7);

    //! Output to a tag
    LineFormatter tag(std::string tag, int prec = 7);

    bool IamRoot();
};

/*
class TaggedLine {

private:
    FileWriter &_fw;
    const std::string _tag;
    const int _prec;

public:

    TaggedLine(FileWriter &fw, std::string tag, int prec = 7) : _fw(fw), _tag(tag), _prec(prec) {};

    //! Output to a tag
    template<typename T>
    LineFormatter operator<<(const T &a) {
        LineFormatter ret(_fw._ostr, _tag, _prec);
        ret << a;
        return ret;
    }

    //! Output infoline
    LineFormatter infoline() {
        _fw << "# Format of lines postfixed with tag :: #" << _tag << "\n";
        _fw << "#";
        return LineFormatter(_fw._ostr, _tag, _prec, false);
    }
};
*/
#endif
