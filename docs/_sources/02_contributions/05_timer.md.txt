# Timing your code

SIMULATeQCD provides a build-in timer called `StopWatch` (`src/base/stopWatch.h`).
It is capable of measuring the time duration CPU-only routines as well as the time duration GPU-kernels.

To initialize the timer you simply call
```C++
StopWatch<device> timer;
```
where `device` is boolean template parameter. 
If `device = false` then only the time duration of CPU routines of the current rank will be considered.
If `device = true` then the time duration of GPU-kernel executions will be considered
as well. **Please do not construct the StopWatch before the CommBase!**

## Measuring time duration

Measuring the time duration is quite easy. Here is an example:

```C++
// Initialize timer which also measure GPU-Kernel execution times:
StopWatch<true> timer;

timer.start();

// ... do some computation ...

timer.stop();

rootLogger.info("First time: ", timer);

timer.reset();
timer.start();

// ... do some computation ...

timer.stop();
rootLogger.info("Second time: ", timer);
```

## Combining different timers

It's also possible to add/multiply/etc. different timers, e.g.:
```C++

StopWatch<true> timer1, timer2, timer3;

timer1.start();

// ... do some computation ...

timer1.stop();

// The streaming operator is overloaded, so you can directly 
// print the time with the correct unit.
// The call below could print for example "First time: 5min 34s":
rootLogger.info("First time: ", timer1); 


timer2.start();

// ... do some computation ...

timer2.stop();
rootLogger.info("Second time: ", timer2);


// Combine different timer
timer3 = timer1 + timer2;
timer3 = timer1 - timer2;

timer3 += timer2;
timer3 -= timer2;

timer3 /= 5;
timer3 *= 3;

```

## Getting time in different units
There are some additional convenience functions to obtain the times in different units:
```C++
// Get time duration in certain units:
double microseconds = timer.microseconds();
double milliseconds = timer.milliseconds();
double seconds = timer.seconds();
double minutes = timer.minutes(); 
double hours = timer.hours();
double days = timer.days();

// Get time duration in a human friendly unit, e.g "123ms"
std::string time_str = timer.autoFormat(); 
```

## Compute FLOP/s and MB/s
To compute FLOP/s and MB/s one may use these functions:
```C++
// set how many bytes were processed (for an MB/s output)
timer.setBytes(500);
// return MBytes/s (be sure to call setBytes() before)
double mbytess = timer.mbs();

// set how many FLOPs were calculated 
timer.setFlops(500);
// return MFLOP/s (be sure to call setFlops() before)
double mflopss = timer.mflps();
```
