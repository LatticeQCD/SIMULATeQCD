/* 
 * main_HaloTest.cu                                                               
 * 
 * Lukas Mazur, 9 Oct 2017
 * 
 */

#include "../SIMULATeQCD.h"
#include "testing.h"
#include "HaloTestParam.h"

#define PREC double

template<Layout LatticeLayout, size_t HaloDepth>
size_t getGlobalIndex(LatticeDimensions coord, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    LatticeData lat = GInd::getLatData();
    LatticeDimensions globCoord = lat.globalPos(coord);

    return globCoord[0] + globCoord[1] * lat.globLX
                        + globCoord[2] * lat.globLX * lat.globLY
                        + globCoord[3] * lat.globLX * lat.globLY * lat.globLZ
                        + mu           * lat.globLX * lat.globLY * lat.globLZ * lat.globLT;
}

template<class floatT, Layout LatticeLayout, size_t HaloDepth, bool onDevice>
void fillIndices(Gaugefield<floatT, onDevice,HaloDepth> &gauge) {
    Gaugefield<floatT, false,HaloDepth> gauge_host( gauge.getComm());
    gaugeAccessor<floatT> gaugeAcc = gauge_host.getAccessor();
    typedef GIndexer<All, HaloDepth> GInd;

    for (size_t x = 0; x < GInd::getLatData().lx; x++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
                for (size_t t = 0; t < GInd::getLatData().lt; t++)
                    for (int mu = 0; mu < 4; mu++) {
                        LatticeDimensions localCoord = LatticeDimensions(x, y, z, t);
                        size_t globIndex = getGlobalIndex<LatticeLayout, HaloDepth>(localCoord, mu);
                        GSU3<floatT> tmp((floatT) globIndex);

                        gSite site = GInd::getSite(x, y, z, t);
                        gaugeAcc.setLink(GInd::getSiteMu(site, mu), tmp);
                    }
    gauge = gauge_host;
}



template<class floatT, Layout LatticeLayout, size_t HaloDepth, bool onDevice>
bool CheckIndices(Gaugefield<floatT, onDevice,HaloDepth> &gauge, LatticeDimensions Halo) {
    Gaugefield<floatT, false,HaloDepth> gauge_host( gauge.getComm());
    gauge_host = gauge;

    gaugeAccessor<floatT> gaugeAcc = gauge_host.getAccessor();
    if(!onDevice) gaugeAcc = gauge.getAccessor();

    typedef GIndexer<All, HaloDepth> GInd;
    typedef HaloIndexer<All, HaloDepth> HInd;
    bool passed = true;

    for (int x = -Halo[0]; x < (int) GInd::getLatData().lx + Halo[0]; x++)
        for (int y = -Halo[1]; y < (int) GInd::getLatData().ly + Halo[1]; y++)
            for (int z = -Halo[2]; z < (int) GInd::getLatData().lz + Halo[2]; z++)
                for (int t = -Halo[3]; t < (int) GInd::getLatData().lt + Halo[3]; t++)
                    for (int mu = 0; mu < 4; mu++) {
                        LatticeDimensions localCoord = LatticeDimensions(x, y, z, t);
                        size_t globIndex = getGlobalIndex<LatticeLayout, HaloDepth>(localCoord, mu);
                        GSU3<floatT> tmpA((floatT) globIndex);

                        gSite site = GInd::getSite(x, y, z, t);
                        GSU3<floatT> tmpB = gaugeAcc.getLink(GInd::getSiteMu(site, mu));

                            if (!compareGSU3(tmpA, tmpB)) {
                                sitexyzt fullcoord = GInd::coordToFullCoord(sitexyzt(localCoord[0],localCoord[1],localCoord[2],localCoord[3]));
                                HaloSegment hseg = HInd::getHSeg(fullcoord);
                                int lr = HInd::getlr(fullcoord);
                                passed = false;
                            }
                    }
    return passed;
}

template<class floatT, bool onDevice, size_t HaloDepth>
struct TestKernel {

    typedef GIndexer<All, HaloDepth> GInd;

    TestKernel() {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu site) {

        size_t gVol1 = GInd::getLatData().globvol1;
        size_t gVol2 = GInd::getLatData().globvol2;
        size_t gVol3 = GInd::getLatData().globvol3;
        size_t gVol4 = GInd::getLatData().globvol4;

        sitexyzt gCoord = GInd::getLatData().globalPos(site.coord);

        size_t globIndex = gCoord.x + gCoord.y * gVol1 + gCoord.z * gVol2 + gCoord.t * gVol3 + site.mu * gVol4;

        GSU3<floatT> tmpA((floatT) globIndex);

        return tmpA;
    }
};

template<size_t HaloDepth, bool onDevice>
void run_func(CommunicationBase& commBase, const int * NodeDim, bool forceHalos){

    rootLogger.info(" ");
    rootLogger.info(" ");
    rootLogger.info("=======================================");
    rootLogger.info(CoutColors::yellowBold , "            Test on " ,  (onDevice ? "GPU" : "CPU") ,  CoutColors::reset);
    rootLogger.info("=======================================");
    rootLogger.info(" ");

    GpuStopWatch timer;
    TestKernel<PREC, onDevice, HaloDepth> Kernel;

    Gaugefield<PREC, onDevice, HaloDepth> gauge( commBase);
    gauge.one();

    fillIndices<PREC, All, HaloDepth, onDevice>(gauge);

    commBase.globalBarrier();
    timer.reset();
    timer.start();
    gauge.updateAll();
    timer.stop();

    if (!forceHalos) {
        check(CheckIndices<PREC, All, HaloDepth, onDevice>(gauge, LatticeDimensions((NodeDim[0] != 1) ? HaloDepth : 0,
                                                                        (NodeDim[1] != 1) ? HaloDepth : 0,
                                                                        (NodeDim[2] != 1) ? HaloDepth : 0,
                                                                        (NodeDim[3] != 1) ? HaloDepth : 0))
                                                                                ,"Just HaloUpdate");
    } else {
        check(CheckIndices<PREC, All, HaloDepth, onDevice>(gauge, LatticeDimensions(HaloDepth, HaloDepth, HaloDepth, HaloDepth))
                ,"Just HaloUpdate");
    }
    rootLogger.info("Halo update " ,  timer);
    rootLogger.info(" ");

    gauge.one();

    commBase.globalBarrier();
    timer.reset();
    timer.start();
    gauge.template iterateOverBulkAllMu(Kernel);
    gauge.updateAll();
    timer.stop();

    if (!forceHalos) {
        check(CheckIndices<PREC, All, HaloDepth, onDevice>(gauge, LatticeDimensions((NodeDim[0] != 1) ? HaloDepth : 0,
                                                                                    (NodeDim[1] != 1) ? HaloDepth : 0,
                                                                                    (NodeDim[2] != 1) ? HaloDepth : 0,
                                                                                    (NodeDim[3] != 1) ? HaloDepth : 0))
                ,"First index computation then Halo update");
    } else {
        check(CheckIndices<PREC, All, HaloDepth, onDevice>(gauge, LatticeDimensions(HaloDepth, HaloDepth, HaloDepth, HaloDepth))
                ,"First index computation then Halo update");
    }
    rootLogger.info("First index computation then Halo update " ,  timer);
    rootLogger.info(" ");
}



int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    HaloTestParam param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/HaloTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 4;

    rootLogger.info("Initialize Lattice");
    initIndexer(HaloDepth,param, commBase, param.forceHalos());

    run_func<HaloDepth,true>(commBase,  param.nodeDim(), param.forceHalos());
    run_func<HaloDepth,false>(commBase, param.nodeDim(), param.forceHalos());

    return 0;
}



