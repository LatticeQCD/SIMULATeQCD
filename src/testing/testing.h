/* 
 * testing.h                                                               
 * 
 */

#pragma once

#include "../base/gutils.h"
#include "../base/math/floatComparison.h"
#include "../define.h"
#include "../gauge/gauge_kernels.cpp"


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

template<typename T>
void compare_elementwise_prec(const T &ref, const T &res, const double rel, const double prec, const std::string text) {
    if (cmp_all_elements_prec(ref, res, prec)) {
        rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text);
    } else {
        rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text);
        rootLogger.error(ref, " vs");
        rootLogger.error(res);
    }
}

template<bool onDevice, typename elemType>
void compare_lattice_containers_elementwise_prec(
    LatticeContainer<onDevice, elemType> &lattice_left,
    LatticeContainer<onDevice, elemType> &lattice_right,
    double prec,
    const std::string text
) {
    // create dummy array to store site-by-site comparisons in
    const size_t elems = GIndexer<All>::getLatData().vol4;
    LatticeContainer<true, int> comparison_lattice(lattice_left.get_CommBase());
    comparison_lattice.adjustSize(elems);

    // site-by-site comparison
    comparison_lattice.template iterateOverBulk<All, 0>(compareLatticeContainers<elemType>(lattice_left.getAccessor(), lattice_right.getAccessor(), prec));

    // reduce number of faults
    int counts = 0;
    comparison_lattice.reduce(counts, elems);

    // print results
    if (counts > 0) {
        rootLogger.error(counts, " faults detected!");
        rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text);
    } else {
        // rootLogger.info(counts, " faults detected!");
        rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text);
    }

}

template<bool onDevice, typename elemType>
void compare_lattice_container_elementwise_prec_to_value(
    LatticeContainer<onDevice, elemType> &lattice,
    elemType value,
    double prec,
    const std::string text
) {
    // create dummy array to store site-by-site comparisons in
    const size_t elems = GIndexer<All>::getLatData().vol4;
    LatticeContainer<true, int> comparison_lattice(lattice.get_CommBase());
    comparison_lattice.adjustSize(elems);

    // site-by-site comparison
    comparison_lattice.template iterateOverBulk<All, 0>(compareLatticeContainerToValue<elemType>(lattice.getAccessor(), value, prec));

    // reduce number of faults
    int counts = 0;
    comparison_lattice.reduce(counts, elems);

    // print results
    if (counts > 0) {
        rootLogger.error(counts, " faults detected!");
        rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text);
    } else {
        // rootLogger.info(counts, " faults detected!");
        rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text);
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

