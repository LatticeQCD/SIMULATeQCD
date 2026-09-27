/*
 * MDWF interface to the AlgRemez minimax rational approximation
 * (Mike Clark, src/tools/rational_approx/alg_remez.C, unchanged).
 *
 * mdwfRemezPower(pnum, pden, lambda_low, lambda_high, order, digits) computes
 * the (order, order) rational function minimizing the maximum relative error
 * of x^(pnum/pden) on [lambda_low, lambda_high] in `digits`-digit arithmetic,
 * and returns its partial-fraction expansion and that of its reciprocal:
 *
 *   x^(+pnum/pden) ~ constant + sum_i residue_i / (x + pole_i)   (power)
 *   x^(-pnum/pden) ~ constant + sum_i residue_i / (x + pole_i)   (inverse)
 *
 * This is the same call poly4.C (ratApprox) makes, with no masses and a single
 * power, so the coefficients match its r_inv_* (power) and r_* (inverse)
 * output for the same inputs.
 *
 * This header deliberately has no GMP/MPFR types: it can be included from CUDA
 * translation units. The implementation (MDWFRemez.cpp) is plain host C++ and
 * must be compiled and linked with GMP and MPFR.
 */

#pragma once

#include <string>
#include <vector>

struct MDWFRemezPartialFractions {
    double constant;
    std::vector<double> residues;
    std::vector<double> poles;
};

struct MDWFRemezApproximation {
    int pnum;
    int pden;
    double lambda_low;
    double lambda_high;
    int order;
    int digits;
    double max_relative_error;   // as returned by AlgRemez::generateApprox
    MDWFRemezPartialFractions power;
    MDWFRemezPartialFractions inverse;
};

MDWFRemezApproximation mdwfRemezPower(int pnum, int pden, double lambda_low, double lambda_high,
                                      int order, int digits);

double mdwfRemezEvaluate(const MDWFRemezPartialFractions &pf, double x);

std::string mdwfRemezDescribe(const MDWFRemezApproximation &approx);
