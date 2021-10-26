# Terminal output & terminating the program

Instead of using `printf` or `std::cout`, we use a custom `Logger` class to print statements in a nicely formatted way.
There are two global instances of the class "Logger" that are automatically created when you include files of the base that include "logging.h" (which basically all of them do).
These two classes are called `stdLogger` and `rootLogger`.

## Printing a statement 

is as simple as
```C++
rootLogger.info() << "Some information about the currently running program.";
```

This will output the statement **on the root process only** in a nicely formatted way with timestamp and also append a newline (no more `std::endl` or \n's that clutter the code).
If you want to output something from every process you can use **stdLogger**.

These are some of the different Logger levels you can choose from: 
```C++
alloc, debug, info, warn, error, fatal 
```

## When to use Logger.error()

Use this when **something goes wrong that can work in principle** but the **program can still continue** and no runtime exception has to be thrown. 
Example:
- one subroutine of a test gives the wrong results, but others after it may still be correct


## Terminating the program in case of error

The way to close your program at runtime because of an error is

```C++
throw ParallelGPUCode_error("Put", "your", "error message here");
```

This will call rootLogger.fatal() and write out the error message, then `throw std::runtime_error`. For the message, you can pass **as many arguments of any type** to it as you want as long as their stream operator `<<` is overloaded.
For example:
```C++
int x = 5;
float y = 10.2;
std::string msg = "test";
throw ParallelGPUCode_error("This will work", x, msg, y, "and print just fine");
```

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
