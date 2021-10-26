/* 
 * main_SpinorHaloTest.cu                                                               
 * 
 * Lukas Mazur, 9 Oct 2017
 * 
 */

#include "../SIMULATeQCD.h"
#include "testing.h"
#include <sstream>
#include "HaloTestParam.h"

#define PREC double

template<Layout LatLayout, size_t HaloDepth>
size_t getGlobalIndex(LatticeDimensions coord, size_t stack) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    LatticeData lat = GInd::getLatData();
    LatticeDimensions globCoord = lat.globalPos(coord);

    return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
           globCoord[3] * lat.globLX * lat.globLY * lat.globLZ +
           stack * lat.globLX * lat.globLY * lat.globLZ * lat.globLT;
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void fillIndices(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinor) {
    typedef GIndexer<All, HaloDepth> GInd;
    typedef GIndexer<Even, HaloDepth> GIndEven;
    typedef GIndexer<Odd, HaloDepth> GIndOdd;
    Spinorfield<floatT, false, LatLayout, HaloDepth, NStacks> spinorHost(spinor.getComm() );
    spinorHost = spinor;
    gVect3arrayAcc<floatT> spinorAcc = spinorHost.getAccessor();
    for (size_t x = 0; x < GInd::getLatData().lx; x++)
        for (size_t y = 0; y < GInd::getLatData().ly; y++)
            for (size_t z = 0; z < GInd::getLatData().lz; z++)
                for (size_t t = 0; t < GInd::getLatData().lt; t++) {

                    bool par = (bool) ((x + y + z + t) % 2);
                    bool even = (LatLayout == Even) && !par;
                    bool odd = (LatLayout == Odd) && par;

                    for (size_t stack = 0; stack < NStacks; stack++){

                        if (LatLayout == All || even || odd) {
                            LatticeDimensions localCoord = LatticeDimensions(x, y, z, t);
                            size_t globIndex = getGlobalIndex<LatLayout, HaloDepth>(localCoord, stack);
                            gVect3<floatT> tmp((floatT) globIndex);

                            if (LatLayout == All) {
                                gSiteStack site = GInd::getSiteStack(x, y, z, t, stack);
                                spinorAcc.setElement(site, tmp);
                            } else if (LatLayout == Even) {
                                gSiteStack site = GIndEven::getSiteStack(x, y, z, t, stack);
                                spinorAcc.setElement(site, tmp);
                            } else if (LatLayout == Odd) {
                                gSiteStack site = GIndOdd::getSiteStack(x, y, z, t, stack);
                                spinorAcc.setElement(site, tmp);
                            }
                        }
                    }
                }
    spinor = spinorHost;
}

template<class floatT>
bool compareGVect3(gVect3<floatT> a, gVect3<floatT> b) {
    floatT tol = 10e-13;

    for (int i = 0; i < 3; i++) {
        GCOMPLEX(floatT) diff = a(i) - b(i);
        if (fabs(diff.cREAL) > tol) return false;
    }
    return true;
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
bool CheckIndices(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinor, LatticeDimensions Halo) {
    typedef GIndexer<All, HaloDepth> GInd;
    typedef GIndexer<Even, HaloDepth> GIndEven;
    typedef GIndexer<Odd, HaloDepth> GIndOdd;
    Spinorfield<floatT, false, LatLayout, HaloDepth, NStacks> spinorHost(spinor.getComm());
    spinorHost = spinor;
    gVect3arrayAcc<floatT> spinorAcc = spinorHost.getAccessor();

    bool passed = true;

    for (int x = -Halo[0]; x < (int) GInd::getLatData().lx + Halo[0]; x++)
        for (int y = -Halo[1]; y < (int) GInd::getLatData().ly + Halo[1]; y++)
            for (int z = -Halo[2]; z < (int) GInd::getLatData().lz + Halo[2]; z++)
                for (int t = -Halo[3]; t < (int) GInd::getLatData().lt + Halo[3]; t++) {
                    bool par = (bool) ((abs(x) + abs(y) + abs(z) + abs(t)) % 2);
                    bool even = (LatLayout == Even) && !par;
                    bool odd = (LatLayout == Odd) && par;

                    for (size_t stack = 0; stack < NStacks; stack++){

                        if (LatLayout == All || even || odd) {
                            LatticeDimensions localCoord = LatticeDimensions(x, y, z, t);
                            size_t globIndex = getGlobalIndex<LatLayout, HaloDepth>(localCoord, stack);
                            gVect3<floatT> tmpA((floatT) globIndex);

                            gVect3<floatT> tmpB;

                            if (LatLayout == All) {
                                gSiteStack site = GInd::getSiteStack(x, y, z, t, stack);
                                tmpB = spinorAcc.getElement(site);
                            } else if (LatLayout == Even) {
                                gSiteStack site = GIndEven::getSiteStack(x, y, z, t, stack);
                                tmpB = spinorAcc.getElement(site);
                            } else if (LatLayout == Odd) {
                                gSiteStack site = GIndOdd::getSiteStack(x, y, z, t, stack);
                                tmpB = spinorAcc.getElement(site);
                            }
                            if (!compareGVect3(tmpA, tmpB)) {
                                passed = false;
                            }
                        }
                    }
                }
    return passed;
}



template< size_t HaloDepth, bool onDevice, size_t NStacks>
bool run_func(CommunicationBase& commBase, const int NodeDim[4], bool forceHalos){

    std::stringstream output;
    output << "Test on " << (onDevice ? "GPU" : "CPU") << " with " << NStacks << " Stacks";
    rootLogger.info() << " ";
    rootLogger.info() << " ";
    rootLogger.info() << "==========================================";
    rootLogger.info() << CoutColors::yellowBold << "       " << output.str() << CoutColors::reset;
    rootLogger.info() << "==========================================";
    rootLogger.info() << " ";


/// ================ Even Test ================ ///
    rootLogger.info() << "----------------------------------------";
    rootLogger.info() << "           ";
    rootLogger.info() << "Initialize Lattice with even indices";
    const Layout LatLayout2 = Even;

    rootLogger.info() << "Initialize Spinorfield";
    Spinorfield<PREC, onDevice, LatLayout2, HaloDepth, NStacks> spinor2(commBase);

    rootLogger.info() << "Fill indices";
    fillIndices<PREC, onDevice, LatLayout2, HaloDepth>(spinor2);


    rootLogger.info() << "Update Halos";
    spinor2.updateAll();

    rootLogger.info() << "Check indices";
    bool EvenTest;
    if (!forceHalos) {
        EvenTest = CheckIndices<PREC, onDevice, LatLayout2, HaloDepth>(spinor2,
                                                             LatticeDimensions((NodeDim[0] != 1) ? HaloDepth : 0,
                                                                               (NodeDim[1] != 1) ? HaloDepth : 0,
                                                                               (NodeDim[2] != 1) ? HaloDepth : 0,
                                                                               (NodeDim[3] != 1) ? HaloDepth : 0));
    } else {

        EvenTest = CheckIndices<PREC, onDevice, LatLayout2, HaloDepth>(spinor2, LatticeDimensions(HaloDepth, HaloDepth, HaloDepth,
                                                                                        HaloDepth));
    }


/// ================ Odd Test ================ ///
    rootLogger.info() << "----------------------------------------";
    rootLogger.info() << "           ";
    rootLogger.info() << "Initialize Lattice with odd indices";
    const Layout LatLayout3 = Odd;

    rootLogger.info() << "Initialize Spinorfield";
    Spinorfield<PREC, onDevice, LatLayout3, HaloDepth, NStacks> spinor3(commBase);

    rootLogger.info() << "Fill indices";
    fillIndices<PREC, onDevice, LatLayout3, HaloDepth>(spinor3);


    rootLogger.info() << "Update Halos";
    spinor3.updateAll();

    rootLogger.info() << "Check indices";

    bool OddTest;
    if (!forceHalos) {
        OddTest = CheckIndices<PREC, onDevice, LatLayout3, HaloDepth>(spinor3,
                                                            LatticeDimensions((NodeDim[0] != 1) ? HaloDepth : 0,
                                                                              (NodeDim[1] != 1) ? HaloDepth : 0,
                                                                              (NodeDim[2] != 1) ? HaloDepth : 0,
                                                                              (NodeDim[3] != 1) ? HaloDepth : 0));
    } else {
        OddTest = CheckIndices<PREC, onDevice, LatLayout3, HaloDepth>(spinor3, LatticeDimensions(HaloDepth, HaloDepth, HaloDepth,
                                                                                       HaloDepth));
    }



/// ================ Full Test ================ ///
    rootLogger.info() << "----------------------------------------";
    rootLogger.info() << "           ";
    rootLogger.info() << "Initialize Lattice with all indices";

    const Layout LatLayout = All;

    rootLogger.info() << "Initialize Spinorfield";
    Spinorfield<PREC, onDevice, LatLayout, HaloDepth, NStacks> spinor(commBase);
    Gaugefield<PREC, onDevice, HaloDepth> gauge( commBase);

    rootLogger.info() << "Fill indices";
    fillIndices<PREC, onDevice, LatLayout, HaloDepth>(spinor);


    rootLogger.info() << "Update Halos";
    spinor.updateAll();

    rootLogger.info() << "Check indices";

    bool fullTest;
    if (!forceHalos) {
        fullTest = CheckIndices<PREC, onDevice, LatLayout, HaloDepth>(spinor, LatticeDimensions((NodeDim[0] != 1) ? HaloDepth : 0,
                                                                                                (NodeDim[1] != 1) ? HaloDepth : 0,
                                                                                                (NodeDim[2] != 1) ? HaloDepth : 0,
                                                                                                (NodeDim[3] != 1) ? HaloDepth
                                                                                                                  : 0));
    } else {
        fullTest = CheckIndices<PREC, onDevice, LatLayout, HaloDepth>(spinor, LatticeDimensions(HaloDepth, HaloDepth, HaloDepth,
                                                                                                HaloDepth));
    }

    check(fullTest && EvenTest && OddTest,output.str());
    return 0;
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    HaloTestParam param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/SpinorHaloTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 2;

    initIndexer(HaloDepth,param, commBase, param.forceHalos());

    run_func<HaloDepth,true, 1>(commBase, param.nodeDim(),param.forceHalos());
    run_func<HaloDepth,true, 8>(commBase, param.nodeDim(),param.forceHalos());
    run_func<HaloDepth,true, 14>(commBase, param.nodeDim(),param.forceHalos());

    run_func<HaloDepth,false, 1>(commBase, param.nodeDim(),param.forceHalos());
    run_func<HaloDepth,false, 8>(commBase, param.nodeDim(),param.forceHalos());
    return 0;
}
