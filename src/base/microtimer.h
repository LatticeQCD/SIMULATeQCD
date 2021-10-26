#ifndef _INC_TIMER
#define _INC_TIMER

#include <iostream>
#include <sstream>
#include <ctime>

#include <stdio.h>
#include <sys/time.h>

using std::ostream;


//! A class to time events/function calls
class MicroTimer {


private :

    double starttime;
    double total;

    long bytes;
    long flops;

    //! Implement something system specific to return
    //! the time in ms as double.
    static double tick() {
	struct timeval cur;
	gettimeofday(&cur, NULL);
	long  seconds  = cur.tv_sec ;
	long useconds  = cur.tv_usec ;
	return ((seconds) * 1000 + useconds/1000.);
    }


public :

    //! Constructor, also starts the timer
    MicroTimer() { 
	starttime = tick(); 
    	bytes = -1;
	flops = -1;
	total = 0;
    };
    
    //! reset and start the timer
    void reset() { starttime = tick(); total = 0; };
    //! resume the timer
    void start() { starttime = tick();}
    //! stop the timer
    void stop()  { total += (tick()-starttime); }
    //! return the elapsed time in ms
    double ms() const { return total; }
    //! set how many bytes were processed (for an MB/s output)
    void setBytes(const long b ) { bytes = b ; }
    //! set how many FLOPs were calculated 
    void setFlops(const long f ) { flops = f ; } 

    //! return MBytes/s (be sure to call setBytes() before)
    double mbs() const {
	return (bytes / (ms()*1024*1024/1000.));
    }

    //! return MFLOP/s (be sure to call setFlops() before)
    double mflps() const {
	return (flops / (ms()*1000.));
    }


    //! Add two timings (bytes and flops are not transfered atm)
    MicroTimer operator+(const MicroTimer & lhs) const {
	MicroTimer ret;
	ret.total = total + lhs.total;
	return ret;
    }
    //! Substract two timings (bytes and flops are not transfered atm)
    MicroTimer operator-(const MicroTimer & lhs) const {
	MicroTimer ret;
	ret.total = total - lhs.total;
	return ret;
    }

    //! Divide the timing by an integer (e.g. loop count)
    MicroTimer & operator/=(const int & lhs) {
	total /= lhs;
	return *this ;
    }   

    //! Calculate ratio of two timings
    double operator/(const MicroTimer & lhs) {
	return (total / lhs.total);
    }


 
    //! Write formatted timing output to ostream
    friend ostream& operator<<(ostream &str, const MicroTimer & mt ) {
	char tmp[255];
	if ((mt.bytes>0)&&(mt.flops>0))
	    sprintf(tmp, "[T: %7.2lf ms   %7.2f MB  %7.2f MFlops]", mt.ms(), mt.mbs(), mt.mflps());
	else if ((mt.bytes>0))
	sprintf(tmp, "[T: %7.2lf ms   %7.2f MB/s]", mt.ms(), mt.mbs());
	else
	    sprintf(tmp, "[T: %7.2lf ms]", mt.ms());
	str << tmp;
	return str ;
    }


} ;


#endif



