//
// Created by Lukas Mazur on 17.11.18.
//

#ifndef FLOATCOMPARISON_H
#define FLOATCOMPARISON_H

#include <cmath>
#include <limits>
#include "../wrapper/gpu_wrapper.h"



/// copied from https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison

/// This can be used on the GPU.
template<class T>
HOST_DEVICE bool cmp_rel(const T a, const T b, const double rel, const double prec) {
    if (abs(a-b) / abs(a+b) < rel && abs(a-b) < prec) {
        return true;
    }
    return false;
}

/// Implements relative method - do not use for comparing with zero. Use this most of the time, tolerance needs to
/// be meaningful in your context.
template<typename TReal>
HOST_DEVICE static bool isApproximatelyEqual(const TReal a, const TReal b, const TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    TReal diff = std::fabs(a - b);
    if (diff <= tolerance)
        return true;

    if (diff < fmax(fabs(a), fabs(b)) * tolerance)
        return true;

    return false;
}

/// Supply tolerance that is meaningful in your context. For example, default tolerance may not work if you are
/// comparing double with float.
template<typename TReal>
static bool isApproximatelyZero(const TReal a, const TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if (std::fabs(a) <= tolerance)
        return true;
    return false;
}

/// Use this when you want to be on safe side. For example, don't start over unless signal is above 1.
template<typename TReal>
static bool isDefinitelyLessThan(const TReal a, const TReal b, const TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    TReal diff = a - b;
    if (diff < tolerance)
        return true;

    if (diff < std::fmax(std::fabs(a), std::fabs(b)) * tolerance)
        return true;

    return false;
}
template<typename TReal>
static bool isDefinitelyGreaterThan(const TReal a, const TReal b, const TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    TReal diff = a - b;
    if (diff > tolerance)
        return true;

    if (diff > std::fmax(std::fabs(a), std::fabs(b)) * tolerance)
        return true;

    return false;
}

/// Implements ULP method. Use this when you are only concerned about floating point precision issues. For example, if
/// you want to see if a is 1.0 by checking if it's within 10 closest representable floating point numbers around 1.0.
template<typename TReal>
static bool isWithinPrecisionInterval(const TReal a, const TReal b, unsigned int interval_size = 1)
{
    TReal min_a = a - (a - std::nextafter(a, std::numeric_limits<TReal>::lowest())) * interval_size;
    TReal max_a = a + (std::nextafter(a, std::numeric_limits<TReal>::max()) - a) * interval_size;

    return min_a <= b && max_a >= b;
}

#endif //FLOATCOMPARISON_H
