# File Writer


The `FileWriter` contains objects to help streamline output to files. The to-be-outputted data
are stored in `LineFormatter` objects, which are essentially `ostream` objects. They automatically
end the line whenever they are destructed; for example they will end lines automatically
inside a loop. At the end of each line there is appended a `tag`, which is a `#` followed by a
string. The `tag` allows the user to append a label to a line to make for easy searching later
with `grep` or shell scripts.

The other object is the `FileWriter` object, which opens the output file stream, and closes
it automatically when it is destroyed. The `FileWriter` is the object that gives the `ostream`;
this `ostream` can be passed to the `LineFormatter` for data output using, for example,
the `FileWriter` `header()` method.

The `FileWriter` is used extensively in the `gradientFlow` application. Very generally, the
`FileWriter` can be used following an example like this one:
```C++
FileWriter filePolyCorrSinglet(commBase, latticeParameters);
filePolyCorrSinglet.createFile("PolyCorrFileName");
LineFormatter newLineplc1 = filePolyCorrSinglet.tag(""); /// Passes ostream to newLineplc1
newLineplc1 << flow_time;
/// Write std::vector vec_plc1 to output file
for (int dx=0 ; dx<distmax ; dx++) {
    newLineplc1 << vec_plc1[dx];
}
```
Remember that when `newLineplc1` leaves scope, it will end the line in the output file. By
default there will be some white space between output from the `LineFormatter`. This can be
seen from the `fieldwidth` argument in its constructor
```C++
LineFormatter(std::ostream &ostr, std::string tag, int prec = 7, bool space = true) :
            _tag(tag), _ostr(ostr), fieldwidth(prec + 8), endl(false)
```
which is then used in the `<<` stream operator
```C++
template<typename T>
    LineFormatter & operator<<(const T &obj) {
        _ostr << std::setw(fieldwidth) << obj;
        return *this;
    }
```
