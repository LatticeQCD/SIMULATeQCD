/*
 * main_fourierTest.cpp
 *
 * Jonas Winte
 *
 * Test the Fourier transforms of FourierClass against a naive DFT on the host.
 * Convention under test (unitary):
 *     F_sign[f](k) = N^{-1/2} sum_x exp(+i sign 2 pi k x / N) f(x),
 * applied once per transformed direction.
 *
 * Covered functions of FourierClass:
 *   1D primitives, for dir = 0, 1, 2, 3 and sign = +1, -1, against the host DFT:
 *     - performFourierTransformDirection<dir>                                     (scalar container, line layout)
 *     - performFourierTransformDirectionPolymorph<dir, Matrix4x4SymComplex>       (single rank only)
 *   composite transforms, for SpatialTemporal = Spatial, Temporal, Both; each checked against the host DFT for
 *     both signs, for forward then backward, for Parseval, and for the transform of a delta at the origin:
 *     - performFourierTransformMatrix4x4SymComponentwise<mode>
 *     - performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<mode>
 *     - performFourierTransformPolymorph<COMPLEX, mode>                           (single rank only)
 *     - performFourierTransformPolymorph<Matrix4x4SymComplex, mode>               (single rank only)
 *     - performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<mode>             (single rank only), also checked
 *       for agreement with the componentwise tensor transform
 *
 * Not covered: the Tensor4x4Symx4x4SymComplex instantiations of performFourierTransformDirectionPolymorph and
 * performFourierTransformPolymorph, which are disabled (per-thread buffers too large, see fourierPolymorph).
 * 
 */

#include "../simulateqcd.h"
#include "../experimental/fourierNon2.h"
#include "../base/math/matrix4x4SymComplex.h"
#include "../base/math/tensor4x4Symx4x4SymComplex.h"
#include <array>
#include <complex>
#include <random>

#define PREC double
#define HaloDepth 0

typedef std::complex<double> cplx;
typedef GIndexer<All, HaloDepth> GInd;

constexpr double tolerance = 1e-11;


// ------------------------------------------------------------------------------- element types <-> host components

template<typename elemType>
struct ElemTraits;

template<>
struct ElemTraits<COMPLEX(PREC)> {
    static constexpr int ncomp = 1;
    static COMPLEX(PREC) fromHost(const cplx *c) {
        return COMPLEX(PREC)(c[0].real(), c[0].imag());
    }
    static void toHost(const COMPLEX(PREC) &e, cplx *c) {
        c[0] = cplx(real(e), imag(e));
    }
};

template<>
struct ElemTraits<Matrix4x4SymComplex<PREC>> {
    static constexpr int ncomp = 10;
    static Matrix4x4SymComplex<PREC> fromHost(const cplx *c) {
        Matrix4x4SymComplex<PREC> m;
        for (int i = 0; i < 10; i++) m.elems[i] = COMPLEX(PREC)(c[i].real(), c[i].imag());
        return m;
    }
    static void toHost(const Matrix4x4SymComplex<PREC> &m, cplx *c) {
        for (int i = 0; i < 10; i++) c[i] = cplx(real(m.elems[i]), imag(m.elems[i]));
    }
};

template<>
struct ElemTraits<Tensor4x4Symx4x4SymComplex<PREC>> {
    static constexpr int ncomp = 100;
    static Tensor4x4Symx4x4SymComplex<PREC> fromHost(const cplx *c) {
        Tensor4x4Symx4x4SymComplex<PREC> tensor;
        for (int a = 0; a < 10; a++)
        for (int b = 0; b < 10; b++) {
            tensor.elems[a][b] = COMPLEX(PREC)(c[a*10+b].real(), c[a*10+b].imag());
        }
        return tensor;
    }
    static void toHost(const Tensor4x4Symx4x4SymComplex<PREC> &tensor, cplx *c) {
        for (int a = 0; a < 10; a++)
        for (int b = 0; b < 10; b++) {
            c[a*10+b] = cplx(real(tensor.elems[a][b]), imag(tensor.elems[a][b]));
        }
    }
};


// ------------------------------------------------------------------------------- global field on the host

std::array<int, 4> globalExtents() {
    sitexyzt globL = GInd::getLatData().globalLatticeXYZT();
    return {globL[0], globL[1], globL[2], globL[3]};
}

std::array<int, 4> localExtents() {
    return {(int) GInd::getLatData().lx, (int) GInd::getLatData().ly, (int) GInd::getLatData().lz, (int) GInd::getLatData().lt};
}

// global coordinates of the local origin
std::array<int, 4> localOffset() {
    sitexyzt origin = GInd::getLatData().globalPos(sitexyzt(0, 0, 0, 0));
    return {origin[0], origin[1], origin[2], origin[3]};
}

// the full global field, identical on every rank; ncomp complex components per site
struct HostField {
    std::array<int, 4> GL;
    int ncomp;
    std::vector<cplx> data;

    explicit HostField(int _ncomp) : GL(globalExtents()), ncomp(_ncomp) {
        data.assign((size_t) GL[0] * GL[1] * GL[2] * GL[3] * ncomp, cplx(0.0, 0.0));
    }

    size_t index(const std::array<int, 4> &g) const {
        return (size_t) g[0] + GL[0] * ((size_t) g[1] + GL[1] * ((size_t) g[2] + GL[2] * (size_t) g[3]));
    }

    cplx *at(const std::array<int, 4> &g) { return &data[index(g) * ncomp]; }
    const cplx *at(const std::array<int, 4> &g) const { return &data[index(g) * ncomp]; }

    double maxAbs() const {
        double result = 0.0;
        for (const cplx &value : data) result = std::max(result, std::abs(value));
        return result;
    }

    double normSquared() const {
        double result = 0.0;
        for (const cplx &value : data) result += std::norm(value);
        return result;
    }
};

template<typename Body>
void forAllGlobalSites(const std::array<int, 4> &GL, Body body) {
    std::array<int, 4> g;
    for (g[3] = 0; g[3] < GL[3]; g[3]++)
    for (g[2] = 0; g[2] < GL[2]; g[2]++)
    for (g[1] = 0; g[1] < GL[1]; g[1]++)
    for (g[0] = 0; g[0] < GL[0]; g[0]++) {
        body(g);
    }
}

HostField randomField(int ncomp, unsigned int seed) {
    HostField field(ncomp);
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (cplx &value : field.data) {
        double re = uniform(engine);
        double im = uniform(engine);
        value = cplx(re, im);
    }
    return field;
}

// component c has the value (c+1) at the global origin and zero elsewhere
HostField deltaField(int ncomp) {
    HostField field(ncomp);
    cplx *origin = field.at({0, 0, 0, 0});
    for (int c = 0; c < ncomp; c++) origin[c] = cplx(c + 1.0, 0.0);
    return field;
}

// naive 1D DFT along dir, unitary normalization
HostField hostDFT1D(const HostField &in, int dir, int sign) {
    HostField out(in.ncomp);
    const int N = in.GL[dir];
    const double norm = 1.0 / std::sqrt((double) N);

    std::vector<cplx> phase(N);
    for (int n = 0; n < N; n++) {
        double angle = sign * 2.0 * M_PI * n / N;
        phase[n] = cplx(std::cos(angle), std::sin(angle));
    }

    forAllGlobalSites(in.GL, [&](const std::array<int, 4> &g) {
        cplx *result = out.at(g);
        const int k = g[dir];
        std::array<int, 4> h = g;
        for (int x = 0; x < N; x++) {
            h[dir] = x;
            const cplx *source = in.at(h);
            const cplx p = phase[((size_t) k * x) % N];
            for (int c = 0; c < in.ncomp; c++) result[c] += p * source[c];
        }
        for (int c = 0; c < in.ncomp; c++) result[c] *= norm;
    });

    return out;
}

std::array<bool, 4> transformedDirections(SpatialTemporal spatialTemporal) {
    switch (spatialTemporal) {
        case SpatialTemporal::Spatial:
            return {true, true, true, false};
        case SpatialTemporal::Temporal:
            return {false, false, false, true};
        default:
            return {true, true, true, true};
    }
}

std::string modeName(SpatialTemporal spatialTemporal) {
    switch (spatialTemporal) {
        case SpatialTemporal::Spatial:
            return "Spatial";
        case SpatialTemporal::Temporal:
            return "Temporal";
        default:
            return "Both";
    }
}

HostField hostDFT(const HostField &in, const std::array<bool, 4> &dirs, int sign) {
    HostField result = in;
    for (int dir = 0; dir < 4; dir++) {
        if (dirs[dir]) result = hostDFT1D(result, dir, sign);
    }
    return result;
}

// analytic transform of deltaField: (c+1) * prod_{transformed} N^{-1/2}, on sites whose untransformed coordinates are 0
HostField expectedDeltaTransform(int ncomp, const std::array<bool, 4> &dirs) {
    HostField field(ncomp);
    double value = 1.0;
    for (int dir = 0; dir < 4; dir++) {
        if (dirs[dir]) value /= std::sqrt((double) field.GL[dir]);
    }
    forAllGlobalSites(field.GL, [&](const std::array<int, 4> &g) {
        for (int dir = 0; dir < 4; dir++) {
            if (!dirs[dir] && g[dir] != 0) return;
        }
        cplx *site = field.at(g);
        for (int c = 0; c < ncomp; c++) site[c] = cplx((c + 1.0) * value, 0.0);
    });
    return field;
}


// ------------------------------------------------------------------------------- host <-> lattice containers

template<typename Body>
void forAllLocalSites(Body body) {
    const std::array<int, 4> L = localExtents();
    for (int t = 0; t < L[3]; t++)
    for (int z = 0; z < L[2]; z++)
    for (int y = 0; y < L[1]; y++)
    for (int x = 0; x < L[0]; x++) {
        sitexyzt global = GInd::getLatData().globalPos(sitexyzt(x, y, z, t));
        body(GInd::getSite(x, y, z, t), std::array<int, 4>{global[0], global[1], global[2], global[3]});
    }
}

template<typename elemType>
void upload(const HostField &field, LatticeContainer<false, elemType> &host, LatticeContainer<true, elemType> &device) {
    LatticeContainerAccessor accessor = host.getAccessor();
    forAllLocalSites([&](gSite site, const std::array<int, 4> &g) {
        accessor.setElement(site, ElemTraits<elemType>::fromHost(field.at(g)));
    });
    device.copyFromLatticeContainer(host);
}

// max_x,c |device - reference| / max |reference|, over all ranks
template<typename elemType>
double relativeDeviation(LatticeContainer<true, elemType> &device, LatticeContainer<false, elemType> &host,
                         const HostField &reference, CommunicationBase &commBase) {
    host.copyFromLatticeContainer(device);
    LatticeContainerAccessor accessor = host.getAccessor();
    std::vector<cplx> buffer(reference.ncomp);
    double maxDiff = 0.0;
    forAllLocalSites([&](gSite site, const std::array<int, 4> &g) {
        ElemTraits<elemType>::toHost(accessor.getElement<elemType>(site), buffer.data());
        const cplx *ref = reference.at(g);
        for (int c = 0; c < reference.ncomp; c++) maxDiff = std::max(maxDiff, std::abs(buffer[c] - ref[c]));
    });
    maxDiff = commBase.globalMaximum(maxDiff);
    return maxDiff / std::max(reference.maxAbs(), 1e-300);
}

template<typename elemType>
double normSquared(LatticeContainer<true, elemType> &device, LatticeContainer<false, elemType> &host, CommunicationBase &commBase) {
    host.copyFromLatticeContainer(device);
    LatticeContainerAccessor accessor = host.getAccessor();
    std::vector<cplx> buffer(ElemTraits<elemType>::ncomp);
    double result = 0.0;
    forAllLocalSites([&](gSite site, const std::array<int, 4> &) {
        ElemTraits<elemType>::toHost(accessor.getElement<elemType>(site), buffer.data());
        for (const cplx &value : buffer) result += std::norm(value);
    });
    return commBase.reduce(result);
}


// ------------------------------------------------------------------------------- test bookkeeping

struct TestContext {
    CommunicationBase &commBase;
    FourierClass<PREC> &fourierClass;
    bool lerror = false;
    int containerCount = 0;

    TestContext(CommunicationBase &_commBase, FourierClass<PREC> &_fourierClass) : commBase(_commBase), fourierClass(_fourierClass) {}

    bool singleRank() { return commBase.getNumberProcesses() == 1; }

    // memory names of containers that are not "SHARED_" have to be unique
    std::string uniqueName(const std::string &prefix) { return prefix + "_" + std::to_string(containerCount++); }

    void checkDeviation(double deviation, const std::string &text) {
        if (deviation <= tolerance) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text, " (rel. dev. ", deviation, ")");
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text, " (rel. dev. ", deviation, ")");
            lerror = true;
        }
    }

    // GPU errors throw; record them as a failed check and go on with the remaining ones
    template<typename Check>
    void runGuarded(const std::string &text, Check check) {
        try {
            check();
        } catch (const std::runtime_error &error) {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text, " threw: ", error.what());
            lerror = true;
        }
    }
};

template<typename elemType>
struct ContainerPair {
    LatticeContainer<false, elemType> host;
    LatticeContainer<true, elemType> device;

    ContainerPair(TestContext &ctx, const std::string &prefix, size_t size) :
        host(ctx.commBase, ctx.uniqueName(prefix + "Host"), ctx.uniqueName(prefix + "HostB"), ctx.uniqueName(prefix + "HostC"), ctx.uniqueName(prefix + "HostD")),
        device(ctx.commBase, ctx.uniqueName(prefix + "Device"), ctx.uniqueName(prefix + "DeviceB"), ctx.uniqueName(prefix + "DeviceC"), ctx.uniqueName(prefix + "DeviceD")) {
        host.adjustSize(size);
        device.adjustSize(size);
    }
};


// ------------------------------------------------------------------------------- 1D primitives

// performFourierTransformDirection works on a scalar container in the "line layout": the local layout
// x + lx*(y + ly*(z + lz*t)), where after the call the extent along dir is the global one (gathered over ranks)
template<int dir>
void testDirection(TestContext &ctx, int sign) {
    const std::array<int, 4> GL = globalExtents();
    const std::array<int, 4> L = localExtents();
    const std::array<int, 4> offset = localOffset();
    const size_t globalVolume = (size_t) GL[0] * GL[1] * GL[2] * GL[3];

    HostField input = randomField(1, 1000 + dir);
    HostField reference = hostDFT1D(input, dir, sign);

    ContainerPair<COMPLEX(PREC)> redBase(ctx, "fourierTestDirection", globalVolume);

    LatticeContainerAccessor accessor = redBase.host.getAccessor();
    forAllLocalSites([&](gSite, const std::array<int, 4> &g) {
        std::array<int, 4> c = {g[0] - offset[0], g[1] - offset[1], g[2] - offset[2], g[3] - offset[3]};
        size_t index = c[0] + (size_t) L[0] * (c[1] + (size_t) L[1] * (c[2] + (size_t) L[2] * c[3]));
        accessor.setElement(index, ElemTraits<COMPLEX(PREC)>::fromHost(input.at(g)));
    });
    redBase.device.copyFromLatticeContainer(redBase.host);

    ctx.fourierClass.performFourierTransformDirection<dir>(redBase.device, redBase.host, sign);

    redBase.host.copyFromLatticeContainer(redBase.device);
    accessor = redBase.host.getAccessor();

    std::array<int, 4> D = L;
    D[dir] = GL[dir];
    double maxDiff = 0.0;
    std::array<int, 4> c;
    for (c[3] = 0; c[3] < D[3]; c[3]++)
    for (c[2] = 0; c[2] < D[2]; c[2]++)
    for (c[1] = 0; c[1] < D[1]; c[1]++)
    for (c[0] = 0; c[0] < D[0]; c[0]++) {
        size_t index = c[0] + (size_t) D[0] * (c[1] + (size_t) D[1] * (c[2] + (size_t) D[2] * c[3]));
        std::array<int, 4> g;
        for (int i = 0; i < 4; i++) g[i] = (i == dir) ? c[i] : c[i] + offset[i];
        cplx value;
        ElemTraits<COMPLEX(PREC)>::toHost(accessor.getElement<COMPLEX(PREC)>(index), &value);
        maxDiff = std::max(maxDiff, std::abs(value - reference.at(g)[0]));
    }
    maxDiff = ctx.commBase.globalMaximum(maxDiff);

    ctx.checkDeviation(maxDiff / reference.maxAbs(),
        "performFourierTransformDirection<" + std::to_string(dir) + ">, sign " + std::to_string(sign) + " vs host DFT");
}

template<int dir, typename elemType>
void testDirectionPolymorph(TestContext &ctx, const std::string &typeName, int sign) {
    const std::string text = "performFourierTransformDirectionPolymorph<" + std::to_string(dir) + ", " + typeName + ">, sign " + std::to_string(sign);
    if (!ctx.singleRank()) {
        rootLogger.info("SKIPPED (single rank only): ", text);
        return;
    }

    ctx.runGuarded(text, [&]() {
        HostField input = randomField(ElemTraits<elemType>::ncomp, 2000 + dir);
        HostField reference = hostDFT1D(input, dir, sign);

        ContainerPair<elemType> field(ctx, "fourierTestDirectionPolymorph", GInd::getLatData().vol4);
        upload(input, field.host, field.device);

        ctx.fourierClass.performFourierTransformDirectionPolymorph<dir, elemType>(field.device, sign);

        ctx.checkDeviation(relativeDeviation(field.device, field.host, reference, ctx.commBase), text + " vs host DFT");
    });
}

template<int dir>
void testPrimitivesInDirection(TestContext &ctx) {
    for (int sign : {1, -1}) {
        testDirection<dir>(ctx, sign);
        testDirectionPolymorph<dir, Matrix4x4SymComplex<PREC>>(ctx, "Matrix4x4SymComplex", sign);
        // no Tensor4x4Symx4x4SymComplex: that instantiation is disabled (per-thread buffers too large, see fourierPolymorph)
    }
}


// ------------------------------------------------------------------------------- composite transforms

// transform(in, out, sign) is one of the four composite functions for a fixed SpatialTemporal mode
template<typename elemType, typename Transform>
void testCompositeChecks(TestContext &ctx, const std::string &text, SpatialTemporal spatialTemporal, Transform transform) {
    constexpr int ncomp = ElemTraits<elemType>::ncomp;
    const std::array<bool, 4> dirs = transformedDirections(spatialTemporal);
    const size_t vol4 = GInd::getLatData().vol4;

    ContainerPair<elemType> in(ctx, "fourierTestIn", vol4);
    ContainerPair<elemType> out(ctx, "fourierTestOut", vol4);
    ContainerPair<elemType> back(ctx, "fourierTestBack", vol4);

    HostField input = randomField(ncomp, 3000 + ncomp + 10 * (int) spatialTemporal);
    upload(input, in.host, in.device);

    // both signs against the host DFT
    for (int sign : {1, -1}) {
        transform(in.device, out.device, sign);
        ctx.checkDeviation(relativeDeviation(out.device, out.host, hostDFT(input, dirs, sign), ctx.commBase),
            text + ", sign " + std::to_string(sign) + " vs host DFT");
    }

    // forward then backward gives the input back
    transform(in.device, out.device, 1);
    transform(out.device, back.device, -1);
    ctx.checkDeviation(relativeDeviation(back.device, back.host, input, ctx.commBase), text + ", forward then backward");

    // Parseval (unitary transform)
    const double normIn = input.normSquared();
    const double normOut = normSquared(out.device, out.host, ctx.commBase);
    ctx.checkDeviation(std::abs(normOut - normIn) / normIn, text + ", Parseval");

    // delta at the origin becomes a constant along the transformed directions
    upload(deltaField(ncomp), in.host, in.device);
    transform(in.device, out.device, 1);
    ctx.checkDeviation(relativeDeviation(out.device, out.host, expectedDeltaTransform(ncomp, dirs), ctx.commBase), text + ", delta");
}

template<typename elemType, typename Transform>
void testComposite(TestContext &ctx, const std::string &text, SpatialTemporal spatialTemporal, Transform transform) {
    ctx.runGuarded(text, [&]() { testCompositeChecks<elemType>(ctx, text, spatialTemporal, transform); });
}

template<SpatialTemporal spatialTemporal>
void testCompositesInMode(TestContext &ctx) {
    const std::string mode = modeName(spatialTemporal);
    FourierClass<PREC> &fourier = ctx.fourierClass;

    testComposite<Matrix4x4SymComplex<PREC>>(ctx, "performFourierTransformMatrix4x4SymComponentwise<" + mode + ">", spatialTemporal,
        [&](LatticeContainer<true, Matrix4x4SymComplex<PREC>> &in, LatticeContainer<true, Matrix4x4SymComplex<PREC>> &out, int sign) {
            fourier.performFourierTransformMatrix4x4SymComponentwise<spatialTemporal>(in, out, sign);
        });

    testComposite<Tensor4x4Symx4x4SymComplex<PREC>>(ctx, "performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<" + mode + ">", spatialTemporal,
        [&](LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> &in, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> &out, int sign) {
            fourier.performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<spatialTemporal>(in, out, sign);
        });

    if (!ctx.singleRank()) {
        rootLogger.info("SKIPPED (single rank only): performFourierTransformPolymorph<..., ", mode, "> and performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<", mode, ">");
        return;
    }

    testComposite<COMPLEX(PREC)>(ctx, "performFourierTransformPolymorph<COMPLEX, " + mode + ">", spatialTemporal,
        [&](LatticeContainer<true, COMPLEX(PREC)> &in, LatticeContainer<true, COMPLEX(PREC)> &out, int sign) {
            fourier.performFourierTransformPolymorph<COMPLEX(PREC), spatialTemporal>(in, out, sign);
        });

    testComposite<Matrix4x4SymComplex<PREC>>(ctx, "performFourierTransformPolymorph<Matrix4x4SymComplex, " + mode + ">", spatialTemporal,
        [&](LatticeContainer<true, Matrix4x4SymComplex<PREC>> &in, LatticeContainer<true, Matrix4x4SymComplex<PREC>> &out, int sign) {
            fourier.performFourierTransformPolymorph<Matrix4x4SymComplex<PREC>, spatialTemporal>(in, out, sign);
        });

    // performFourierTransformPolymorph<Tensor4x4Symx4x4SymComplex> is disabled (see fourierPolymorph); HalfPolymorph replaces it

    testComposite<Tensor4x4Symx4x4SymComplex<PREC>>(ctx, "performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<" + mode + ">", spatialTemporal,
        [&](LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> &in, LatticeContainer<true, Tensor4x4Symx4x4SymComplex<PREC>> &out, int sign) {
            fourier.performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<spatialTemporal>(in, out, sign);
        });

    // the "half" trick must agree with the plain componentwise transform
    const size_t vol4 = GInd::getLatData().vol4;
    ContainerPair<Tensor4x4Symx4x4SymComplex<PREC>> in(ctx, "fourierTestHalfIn", vol4);
    ContainerPair<Tensor4x4Symx4x4SymComplex<PREC>> half(ctx, "fourierTestHalf", vol4);
    ContainerPair<Tensor4x4Symx4x4SymComplex<PREC>> componentwise(ctx, "fourierTestComponentwise", vol4);

    HostField input = randomField(100, 4000 + (int) spatialTemporal);
    upload(input, in.host, in.device);
    fourier.performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<spatialTemporal>(in.device, half.device, 1);
    fourier.performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<spatialTemporal>(in.device, componentwise.device, 1);

    HostField componentwiseResult(100);
    componentwise.host.copyFromLatticeContainer(componentwise.device);
    LatticeContainerAccessor accessor = componentwise.host.getAccessor();
    forAllLocalSites([&](gSite site, const std::array<int, 4> &g) {
        ElemTraits<Tensor4x4Symx4x4SymComplex<PREC>>::toHost(accessor.getElement<Tensor4x4Symx4x4SymComplex<PREC>>(site), componentwiseResult.at(g));
    });
    ctx.checkDeviation(relativeDeviation(half.device, half.host, componentwiseResult, ctx.commBase),
        "HalfPolymorph<" + mode + "> equals Componentwise<" + mode + ">");
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/fourierTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth, param, commBase);

    FourierClass<PREC> fourierClass(commBase);
    fourierClass.checkLatticeExtents();

    TestContext ctx(commBase, fourierClass);

    rootLogger.info("Testing the 1D primitives...");
    testPrimitivesInDirection<0>(ctx);
    testPrimitivesInDirection<1>(ctx);
    testPrimitivesInDirection<2>(ctx);
    testPrimitivesInDirection<3>(ctx);

    rootLogger.info("Testing the composite transforms...");
    testCompositesInMode<SpatialTemporal::Spatial>(ctx);
    testCompositesInMode<SpatialTemporal::Temporal>(ctx);
    testCompositesInMode<SpatialTemporal::Both>(ctx);

    if (ctx.lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green, "passed!", CoutColors::reset);
    }

    return 0;
}
