#ifndef TESTING_H
#define TESTING_H

#include "../base/gutils.h"
#include "../base/math/floatComparison.h"
#include "../define.h"
#include "../gauge/gauge_kernels.cu"

void check(bool condition, const std::string text) {
    if (condition) {
        rootLogger.info(text ,  CoutColors::green ,  " PASSED" ,  CoutColors::reset);
    } else {
        rootLogger.error(text  ,  CoutColors::red  ,  " FAILED" ,  CoutColors::reset);
    }
}

template<typename T>
void compare_exact(const T &ref, const T &res, const std::string text) {
    if (ref == res) {
        rootLogger.info(text ,  CoutColors::green ,  " PASSED" ,  CoutColors::reset);
    } else {
        rootLogger.error(text  ,  CoutColors::red ,  " FAILED" ,  CoutColors::reset);
        rootLogger.error(ref ,  " vs");
        rootLogger.error(res);
    }
}

template<typename T>
void compare_relative(const T &ref, const T &res, const double rel, const double prec, const std::string text) {
    if (cmp_rel(ref, res, rel, prec)) {
        rootLogger.info(text ,  CoutColors::green ,  " PASSED" ,  CoutColors::reset);
    } else {
        rootLogger.error(text  ,  CoutColors::red ,   " FAILED" ,  CoutColors::reset);
        rootLogger.error(ref ,  " vs");
        rootLogger.error(res);
    }
}

template <class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
bool compare_fields(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeR, floatT tol=1e-6) {

    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;
    LatticeContainer<true, int> dummy(gaugeL.getComm());
    dummy.adjustSize(elems);
    dummy.template iterateOverBulk<All,HaloDepth>(count_faulty_links<floatT,onDevice,HaloDepth,comp>(gaugeL,gaugeR,tol));
    int faults = 0;
    dummy.reduce(faults, elems);
    rootLogger.info(faults ,  " faults detected!");
    if (faults > 0) {
        return false;
    } else {
        return true;
    }
}

#endif

