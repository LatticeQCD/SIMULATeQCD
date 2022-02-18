/* 
 * parameterManagement.h                                                               
 * 
 */
#ifndef INC_PARAMETERS
#define INC_PARAMETERS

#include <string>
#include <vector>
#include <ostream>
#include <sstream>
#include <list>
#include "../../define.h"
#include <regex>

//! class that holds the left and right side of a "key=value" line
struct strpair {
    std::string key;
    std::string value;

    explicit strpair(const std::string &line) {
        size_t pos = line.find_first_of('=');
        if (pos == std::string::npos)
            return; //key will be empty, so it will not match anything
        if (line[line.find_first_not_of(' ')] == '#') // if first non-space character is '#' then it's a comment
            return;
        key = line.substr(0, pos);
        key.erase(key.find_last_not_of(" \t\r\n\v\f") + 1); //remove whitespaces at end
        

        value = line.substr(pos + 1);

        pos = key.find_first_of('[');
        if (pos == std::string::npos)
            return; //key will be unchanged
        key=line.substr(0,pos);
        key.erase(key.find_last_not_of(" \t\r\n\v\f") + 1); //remove whitespaces at end

        
    }
};

/*! Base class for some parameter.
 *
 * It has a name and some state flags. Different kinds of parameters
 * (single value, fixed number of values, dynamic number of values) are then derived from this (as templates,
 * with the type and possibly the number of values as template parameters).
 * The idea is that the ParameterList only deals with ParameterBase pointers, which point to actual (derived) Parameters.
 * Different types of parameters have different readstream() and print() implementations, allowing the ParameterList
 * to be generic when setting the parameters.
 */
class ParameterBase {
protected:
    std::string name;
    bool isset;
    bool isrequired;
    bool hasdefault;

    //!This expects the part after "name = " and assigns the value(s)
    virtual bool readstream(std::stringstream &s) = 0;

    //! print value(s) to stream. This is needed because a friend function (like operator<<) can not be virtual.
    virtual void print(std::ostream &o) const = 0;

public:
    ParameterBase() {
        isset = false;
        isrequired = false;
        hasdefault = false;
    };

    bool match(strpair &line) {
        if (line.key != name)
            return false;
        std::stringstream s;
        s.str(line.value);
        return readstream(s);
    }

    bool isSet() const { return isset; };

    void clear() { isset = false; };

    bool isRequired() const { return isrequired; };

    void setRequired(bool r) { isrequired = r; };

    friend std::ostream &operator<<(std::ostream &o, const ParameterBase &a) {
        o << a.name << " = ";
        a.print(o);
        return o;
    };

    friend class ParameterList;
};

//! A parameter with a fixed number of values
template<class T, uint size = 1>
class Parameter : public ParameterBase {
private:
    std::vector<T> values;

    void print(std::ostream &o) const override {
        for (uint i = 0; i < values.size(); i++)
            o << values[i] << ' ';
    };

    bool readstream(std::stringstream &s) override {
        for (uint i = 0; i < size; i++)
            s >> values[i];
        isset = !s.fail();
        return isset;
    };

    friend class ParameterList;

public:
    Parameter() : values(size) {};

    //! read only access operator
    const T &operator[](int i) const { return values.at(i); };

    //! cast to T* (for C-like array access)
    operator const T *() const { return &values[0]; }

    const T *operator()() const { return &values[0]; }

    void set(std::vector<T> toWhat) {
        values = toWhat;
        isset = true;
    };

    void set(const T *v) {
        for (size_t i = 0; i < size; i++)
            values[i] = v[i];
        isset = true;
    };
};

//! A parameter with a single value. This is a specification of the above for size=1
template<class T>
class Parameter<T, 1> : public ParameterBase {
private:
    T value;

    void print(std::ostream &o) const override { o << value; };

    bool readstream(std::stringstream &s) override {
        s >> value;
        isset = !s.fail() && !s.bad();
        return isset;
    };

    friend class ParameterList;

public:
    const T &operator()() const { return value; };

    //read only
    //const T& operator=(const T& v) { value = v; return value ; };
    //writable reference
    T &ref() { return value; };

    //set to value
    void set(const T &toWhat) {
        value = toWhat;
        isset = true;
    };
};


//! A parameter with as many values as provided
template<class T>
class DynamicParameter : public ParameterBase {
    std::vector<T> values;

    void print(std::ostream &o) const override {
        for (unsigned int i = 0; i < values.size(); i++)
            o << values[i] << ' ';
    };

    bool readstream(std::stringstream &s) override {
        // values.clear();
        isset = false;
        while (s.good()) {
            T tmp;
            s >> tmp;
//                     stdLogger.trace(s.fail());
            if (s.fail())
                break;
            values.push_back(tmp);
        }
        isset = true; //(values.size()>=1);
        return isset;
    };

public:
    //!Get how many values were read and are stored in this parameter
    size_t numberValues() const {
        return values.size();
    }

    //set to value
    void set(std::vector<T> toWhat) {
        values = toWhat;
        isset = true;
    };
    //get vector
    std::vector<T> get() {
        return values;
    };

    //!read only access operator
    const T &operator[](int i) const { return values[i]; };

    //!backwards compatibility with old implementation #TODO: set a deprecated flag and see
    //!where the compiled complains
    const T &operator()(int i = 0) const { return values.at(i); }

};


//! explicit specialization of member function to read in multiple strings at once that are separated by a space
template <>
inline bool DynamicParameter<std::string>::readstream(std::stringstream &s)
{
    isset = false;
    while (s.good()) {
        std::string tmp;
        getline( s, tmp, ' ' );
        if (s.fail())
            break;
        if (!tmp.empty()){
            values.push_back(tmp);
        }
    }
    isset = true;
    return isset;
}

class CommunicationBase;

//! see example LatticeParameters on how to use this
class ParameterList : protected std::list<ParameterBase *> {
private:
    bool readstream(std::istream &, int argc, char **argv, const std::string& prefix = "PARAM",
                    bool ignore_unknown = false);

public:
    //! Add a parameter to the internal list
    void add(ParameterBase &p, const std::string &name, bool required = true) {
        p.name = name;
        p.isrequired = required;
        p.isset = false;
        p.hasdefault = false;
        push_back(&p);
    }

    //! Add an optional parameter
    void addOptional(ParameterBase &p, const std::string &name) { add(p, name, false); };

    //! Add a parameter with a default value
    template<class T>
    void addDefault(Parameter<T> &p, const std::string &name, const T &def) {
        addOptional(p, name);
        p.hasdefault = true;
        p.value = def;
    }

    //! Add a parameter with default values
    template<class T, unsigned int count>
    void addDefault(Parameter<T, count> &p, const std::string &name, const T *def) {
        addOptional(p, name);
        p.hasdefault = true;
        for (size_t i = 0; i < count; i++)
            p.values[i] = def[i];
    }

    /** Read parameters from stream, output them and check if every
     * parameter that is required is set.
    */
    bool readstream(std::istream &in, const std::string& prefix = "PARAM", const bool ignore_unknown = false) {
        return readstream(in, 0, nullptr, prefix, ignore_unknown);
    }

    /** Read parameters, output them and check if every parameter that
     * is required is set. If argc and argv are given, the filename is
     * treated as default if no parameters are given on the
     * commandline. Otherwise, the first command line parameter is
     * considered to be the filename while the rest might overwrite
     * values from the file
    */
    bool readfile(const CommunicationBase &comm, const std::string &filename, int argc = 0, char **argv = nullptr);

    //! Semi-formatted (multiple values divided by spaces) output to stream
    friend std::ostream &operator<<(std::ostream &o, const ParameterList &a) {
        for (auto i : a)
            o << (*i) << std::endl;
        return o;
    }

    //! Fully formatted '#ParameterName = Value1 Value2 ...' output
    void toStream(std::ostream &str) const {
        for (auto i : *this) {
            const ParameterBase &p = *i;
            if (p.isSet() || p.hasdefault) {
                str << "# " << p << std::endl;
            }
        }
    }

    //! in order to suppress the generation of a default move
    //! assignment in C++11, we declare a copy assignment operator
    ParameterList &operator=(const ParameterList &other) {
        static_cast<std::list<ParameterBase *> &>(*this) = other;
        return *this;
    }

};


#endif
