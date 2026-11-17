// Focused checks for the CPU execution bridge before running physics tests.
#include "../simulateqcd.h"
#include <array>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
struct Accessor {
    double* data;
    void setElement(size_t i, double x) { data[i]=x; }
};
struct Array : RunFunctors<true, Accessor> {
    std::vector<double> data;
    explicit Array(size_t n): data(n,-1.0) {}
    Accessor getAccessor() const override { return {const_cast<double*>(data.data())}; }
};
struct Read {
    size_t operator()(dim3 b, uint3 bi, uint3 ti) {
        return (b.x*bi.x+ti.x)*6+ti.y*3+ti.z;
    }
};
struct Write {
    size_t operator()(size_t i) { return i; }
    size_t operator()(size_t i, size_t loop) { return 3*i+loop; }
};
struct Value {
    double operator()(size_t i) { return double(i+1); }
};
struct Loop {
    double initial=0;
    void initialize(size_t i) { initial=double(i); }
    double operator()(size_t, size_t j) { return initial+10*j; }
};
struct Store {
    double* data;
    void operator()(size_t i) { data[i]=double(i+2); }
};
struct CommIndex {
    size_t operator()(size_t site, size_t mu) { return mu*17+site; }
};
}

int main(int argc, char** argv) {
    try {
        CommunicationBase comm(&argc,&argv);
        comm.init(LatticeDimensions(1,1,1,1));
        // Rectangle paths in the gauge force need s + mu - 2*nu.
        // Check against coordinate arithmetic, including periodic boundaries
        // and mu == nu, rather than another compound indexer helper.
        LatticeParameters indexParam;
        indexParam.set(LatticeDimensions(8,8,8,4), LatticeDimensions(1,1,1,1));
        initIndexer(2, indexParam, comm);
        using GInd = GIndexer<All,2>;
        const int extent[4] = {8,8,8,4};
        size_t indexChecks = 0;
        for (size_t i=0; i<GInd::getLatData().vol4; ++i) {
            const auto site = GInd::getSite(i);
            for (int mu=0; mu<4; ++mu) {
                for (int nu=0; nu<4; ++nu) {
                    auto expected = site.coord;
                    expected[mu] += 1;
                    expected[nu] -= 2;
                    for (int d=0; d<4; ++d)
                        expected[d] = (expected[d]+extent[d]) % extent[d];
                    auto actual = GInd::site_up_2dn(site,mu,nu);
                    const auto expectedSite = GInd::getSite(expected.x,expected.y,expected.z,expected.t);
                    require(actual.isite==expectedSite.isite &&
                            actual.isiteFull==expectedSite.isiteFull,
                            "CPU rectangle-path site indices");
                    for (int d=0; d<4; ++d)
                        require(actual.coord[d]==expected[d], "CPU rectangle-path coordinates");
                    ++indexChecks;
                }
            }
        }
        rootLogger.info("CPU rectangle-path indexing: PASS (", indexChecks, " checks)");
        require(cmp_rel(0.0,0.0,1e-6,1e-6),"CPU comparison of zero components");
        require(cmp_rel(0.125,0.125,1e-6,1e-6),"CPU comparison of sub-unit components");
        require(!cmp_rel(0.125,0.25,1e-6,1e-6),"CPU comparison rejects unequal components");
        // Random gauge construction must not depend on argument evaluation
        // order. Also verify exactly 18 scalar draws are consumed per link.
        uint4 actualState = make_uint4(12345,23456,34567,45678);
        uint4 expectedState = actualState;
        std::array<double,18> draws{};
        for (double& x : draws) x=get_rand<double>(&expectedState);
        SU3<double> expectedGauge;
        for (int row=0;row<3;++row) {
            for (int col=0;col<3;++col) {
                const int k=2*(3*row+col);
                expectedGauge(row,col)=1.0-2.0*COMPLEX(double)(draws[k],draws[k+1]);
            }
        }
        expectedGauge.su3unitarize();
        SU3<double> actualGauge;
        actualGauge.random(&actualState);
        // SU3::operator== delegates to the legacy relative comparison, which
        // divides by |a+b| and is unsafe for zero components. These two paths
        // perform identical arithmetic, so compare finite scalar components
        // directly without a relative denominator or tolerance.
        for (int row=0;row<3;++row) {
            for (int col=0;col<3;++col) {
                const auto actual = actualGauge(row,col);
                const auto expected = expectedGauge(row,col);
                require(std::isfinite(real(actual)) && std::isfinite(imag(actual)) &&
                        std::isfinite(real(expected)) && std::isfinite(imag(expected)) &&
                        real(actual)==real(expected) && imag(actual)==imag(expected),
                        "CPU random gauge draw ordering");
            }
        }
        require(actualState.x==expectedState.x && actualState.y==expectedState.y &&
                actualState.z==expectedState.z && actualState.w==expectedState.w,
                "CPU random gauge draw count");
        constexpr size_t count=17*6;
        Array a(count);
        a.iterateFunctor(Value{},Read{},Write{},17,2,3);
        for (size_t i=0;i<count;++i) require(a.data[i]==double(i+1),"CPU multidimensional launch");
        a.iterateWithConstObject(9.0,Read{},Write{},17,2,3);
        for (double x:a.data) require(x==9.0,"CPU constant launch");
        iterateFunctorNoReturn<true>(Store{a.data.data()},Read{},17,2,3);
        for (size_t i=0;i<count;++i) require(a.data[i]==double(i+2),"CPU no-return launch");
        iterateFunctorComm<true>(Value{},a.getAccessor(),CommIndex{},17,count);
        for (size_t i=0;i<count;++i) require(a.data[i]==double(i+1),"CPU halo launch");
        Array loops(3*count);
        loops.iterateFunctorLoop<3>(Loop{},Read{},Write{},17,2,3,nullptr,2);
        for (size_t i=0;i<count;++i) {
            require(loops.data[3*i]==double(i),"CPU loop initialization");
            require(loops.data[3*i+1]==double(i+10),"CPU loop accumulation");
            require(loops.data[3*i+2]==-1.0,"CPU loop Nmax bound");
        }

        double values[]={-8,2,5,-1};
        double sum=0, maximum=0, segments[3]={};
        size_t bytes=0, offsets[]={0,2,2,4};
        unsigned char scratch=0;
        require(CubReduce<double>(nullptr,&bytes,values,&sum,4)==gpuSuccess && bytes>0,"CPU reduction query");
        require(CubReduce<double>(&scratch,&bytes,values,&sum,4)==gpuSuccess && sum==-2,"CPU sum");
        require(CubReduceMax<double>(&scratch,&bytes,values,&maximum,4)==gpuSuccess && maximum==5,"CPU maximum");
        require(CubReduceStacked<double>(&scratch,&bytes,values,segments,3,offsets)==gpuSuccess,"CPU segmented reduction");
        require(segments[0]==-6 && segments[1]==0 && segments[2]==4,"CPU segment boundaries");

        void* memory=nullptr;
        double copied[4]={};
        require(gpuMalloc(&memory,sizeof(values))==gpuSuccess,"CPU compute allocation");
        require(gpuMemcpy(memory,values,sizeof(values),gpuMemcpyHostToDevice)==gpuSuccess,"CPU copy in");
        require(gpuMemcpy(copied,memory,sizeof(copied),gpuMemcpyDeviceToHost)==gpuSuccess,"CPU copy out");
        require(gpuFree(memory)==gpuSuccess,"CPU free");
        for (size_t i=0;i<4;++i) require(copied[i]==values[i],"CPU copy contents");
        gpuIpcMemHandle_t handle{};
        require(gpuIpcGetMemHandle(&handle,values)==cpuUnsupported,"CPU must reject GPU IPC");
        StopWatch<true> timer;
        timer.start();
        timer.stop();
        require(std::isfinite(timer.seconds()) && timer.seconds()>=0,"CPU compute timer");
        rootLogger.info("CPU runtime regression: PASS");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "CPU runtime regression: FAIL: " << e.what() << '\n';
        return 1;
    }
}
