# Terminal output & terminating the program

Instead of using `printf` or `std::cout`, we use a custom `Logger` class to print statements in a nicely formatted way.
There are two global instances of the class `Logger` that are automatically created when you include files of the base that include `logging.h`. (This is automatically included in `SIMULATeQCD.h`.) 
These two classes are called `stdLogger` and `rootLogger`.

## Printing a statement 

This is accomplished through
```C++
rootLogger.info("Some information about the currently running program.");
```

This will output the statement **on the root process only** in a nicely formatted way with time stamp and also append a newline (no more `std::endl` or \n's that clutter the code).
If you want to output something from every process you can use `stdLogger`.

These are some of the different Logger levels you can choose from: 
```C++
alloc(), debug(), info(), warn(), error(), fatal()
```
The output level can be set by calling
```C++
rootLogger.setVerbosity(LOGLEVEL)
```
where `LOGLEVEL` must be one of these levels:
```C++
ALL, ALLOC, TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF
```
All levels to the left of the selected level are not printed.

For the message, you can pass **as many arguments of any type** to it as you want as long as their stream operator `<<` is overloaded.
For example:
```C++
int x = 5;
float y = 10.2;
std::string msg = "test";
rootLogger.warn("This will work", x, msg, y, "and print just fine"));
```

The `stringFunctions.h` header offers additional convenience function, e.g.:
```C++
int a = 2;
double b = 0.00251;
std::string c = "test";

std::string str1 = sformat("%.3f, %g %s %.2e", 1.23, 0.001, c, b);
std::string str2 = sjoin(a, " ", b, COLORS::red, " 123 ", COLORS::reset, c);
// Result of str1 and str2 is: 
// str1 = "1.230, 0.001 test 2.51e-03"
// str2 = "2 0.00251 123 test"
```


## When to use Logger.error()

Use this when **something goes wrong that can work in principle** but the **program can still continue** and no runtime exception has to be thrown. 
For example, if one subroutine of a test gives the wrong results, but others after it may still be correct,
it makes sense to use `error`.


## Terminating the program in case of error

The way to close your program at runtime because of an error is

```C++
throw std::runtime_error(stdLogger.fatal("Put", "your", "error message here"));
```

This will call rootLogger.fatal() and write out the error message, then `throw std::runtime_error`. 
In your main, you should use something like this to catch any runtime errors and close the program in a clean way:
```C++
int main(int argc, char *argv[]) {
    try {
        <CONTENT OF YOUR MAIN HERE>
    }
    catch (const std::runtime_error &error) {
        return -1;
    }
    return 0;
}
```
