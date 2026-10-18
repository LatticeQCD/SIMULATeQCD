/*
 * Deterministic "random" fields from a counter-based hash, identical in SIMULATeQCD and GPT
 * (tests/regression/gpt/hash_noise.py), for number-by-number tests of a whole MD trajectory (T7):
 *
 *   uniform   u(seed, vec, j)  = (splitmix64(seed ^ splitmix64((vec << 40) ^ j)) >> 11) * 2^-53      in [0, 1)
 *   gaussian  g(seed, vec, j)  = sqrt(-2 log(1 - u(2j))) cos(2 pi u(2j + 1))                          N(0, 1)
 *
 * Momenta: link (global lexicographic site x + Lx (y + Ly (z + Lz t)), mu), algebra index a = 0..7 (Gell-Mann
 * lambda_{a+1}): g_a = g(seed, vec, (4 site + mu) 8 + a); SIMULATeQCD P = sum_a g_a lambda_a / sqrt(2) (kinetic
 * energy (1/2) tr P^2 = (1/2) sum g_a^2, U' = exp(i t P) U).
 * Spinors: component k = 3 spin + colour of a 4D field: re = g(seed, vec, 2 (12 site + k)), im = g(..., + 1).
 */

#pragma once

#include "MDWFMeasurements.h"

#include <cmath>
#include <cstdint>

__host__ __device__ inline double mdwfHashUniform(uint64_t seed, uint64_t vec, uint64_t j) {
    const uint64_t h = mdwfSplitMix64(seed ^ mdwfSplitMix64((vec << 40) ^ j));
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
}

__host__ __device__ inline double mdwfHashGaussian(uint64_t seed, uint64_t vec, uint64_t j) {
    const double u1 = mdwfHashUniform(seed, vec, 2 * j), u2 = mdwfHashUniform(seed, vec, 2 * j + 1);
    return sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * M_PI * u2);
}

template<size_t HaloDepth>
__host__ __device__ inline uint64_t mdwfGlobalLex(const sitexyzt &coord) {
    typedef GIndexer<All, HaloDepth> GInd;
    LatticeData lat = GInd::getLatData();
    const sitexyzt g = lat.globalPos(coord);
    return static_cast<uint64_t>(g.x)
           + lat.globLX * (static_cast<uint64_t>(g.y)
                           + lat.globLY * (static_cast<uint64_t>(g.z) + lat.globLZ * static_cast<uint64_t>(g.t)));
}

// P(link) = sum_a g_a lambda_a / sqrt(2), hermitian traceless.
template<size_t HaloDepth>
struct MDWFHashMomentum {
    uint64_t _seed, _vec;

    MDWFHashMomentum(uint64_t seed, uint64_t vec) : _seed(seed), _vec(vec) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        const uint64_t base = (4 * mdwfGlobalLex<HaloDepth>(site.coord) + site.mu) * 8;
        double g[8];
        for (int a = 0; a < 8; a++) {
            g[a] = mdwfHashGaussian(_seed, _vec, base + a) / sqrt(2.0);
        }
        const double s3 = 1.0 / sqrt(3.0);
        typedef COMPLEX(double) C;
        // Gell-Mann: l1 (01 real), l2 (01 imag), l3 diag(1,-1,0), l4 (02 real), l5 (02 imag), l6 (12 real),
        // l7 (12 imag), l8 diag(1,1,-2)/sqrt(3); l2: [[0,-i],[i,0]].
        return SU3<double>(C(g[2] + s3 * g[7], 0.0), C(g[0], -g[1]), C(g[3], -g[4]),
                           C(g[0], g[1]), C(-g[2] + s3 * g[7], 0.0), C(g[5], -g[6]),
                           C(g[3], g[4]), C(g[5], g[6]), C(-2.0 * s3 * g[7], 0.0));
    }
};

// Spinor of complex Gaussian components on the sites of the field's layout. A 5D field (stack = s = 0 ... Ls - 1)
// takes noise vector vec + s on slice s (GPT: the 5D coordinate s of F_grid); a 4D field (stack 0) is unchanged.
template<Layout LatLayout, size_t HaloDepth>
struct MDWFHashSpinor {
    uint64_t _seed, _vec;
    double _scale;

    MDWFHashSpinor(uint64_t seed, uint64_t vec, double scale = 1.0) : _seed(seed), _vec(vec), _scale(scale) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        const uint64_t lex = mdwfGlobalLex<HaloDepth>(site.coord);
        const uint64_t vec = _vec + static_cast<uint64_t>(site.stack);
        Vect12<double> out(0.0);
        for (int k = 0; k < 12; k++) {
            const uint64_t j = 2 * (12 * lex + k);
            out.data[k] = COMPLEX(double)(_scale * mdwfHashGaussian(_seed, vec, j), _scale * mdwfHashGaussian(_seed, vec, j + 1));
        }
        return out;
    }
};
