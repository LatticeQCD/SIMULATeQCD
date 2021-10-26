#ifndef TESTING_H
#define TESTING_H

#include "../base/gutils.h"
#include "../base/math/floatComparison.h"
#include "../define.h"

void check(bool condition, const std::string text) {
    if (condition) {
        rootLogger.info() << text << CoutColors::green << " PASSED" << CoutColors::reset;
    } else {
        rootLogger.error() << text  << CoutColors::red  << " FAILED" << CoutColors::reset;
    }
}

template<typename T>
void compare_exact(const T &ref, const T &res, const std::string text) {
    if (ref == res) {
        rootLogger.info() << text << CoutColors::green << " PASSED" << CoutColors::reset;
    } else {
        rootLogger.error() << text  << CoutColors::red << " FAILED" << CoutColors::reset;
        rootLogger.error() << ref << " vs";
        rootLogger.error() << res;
    }
}

template<typename T>
void compare_relative(const T &ref, const T &res, const double rel, const double prec, const std::string text) {
    if (cmp_rel(ref, res, rel, prec)) {
        rootLogger.info() << text << CoutColors::green << " PASSED" << CoutColors::reset;
    } else {
        rootLogger.error() << text  << CoutColors::red <<  " FAILED" << CoutColors::reset;
        rootLogger.error() << ref << " vs";
        rootLogger.error() << res;
    }
}

#endif
