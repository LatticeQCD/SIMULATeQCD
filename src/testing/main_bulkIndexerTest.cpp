/*
 * main_BulkIndexerTest.cpp
 *
 * Philipp Scior, 11 Oct 2018
 *
 * Test that the indexer is working correctly.
 *
 */

#include "../simulateqcd.h"

#define MY_BLOCKSIZE 256

__device__ __host__ bool operator!=(const sitexyzt &lhs, const sitexyzt &rhs){
    bool ret = (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.t == rhs.t);
    return !ret;
}

template<int mu_steps, int nu_steps, int rho_steps, Layout LatLayout, size_t HaloDepth>
__device__ __host__ sitexyzt testMove(const gSite &s, const int mu, const int nu, const int rho, const GIndexer<LatLayout, HaloDepth> Indexer){

    int x = s.coordFull.x;
    int y = s.coordFull.y;
    int z = s.coordFull.z;
    int t = s.coordFull.t;

    if (mu_steps > 0) {
        switch (mu) {
            case 0:
                //                      x = (x+mu_steps) % size.lx();
                x = x + mu_steps;
                if (x >= (int)Indexer.getLatData().lxFull)
                    x -= Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+mu_steps) % size.ly();
                y = y + mu_steps;
                if (y >= (int)Indexer.getLatData().lyFull)
                    y -= Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+mu_steps) % size.lz();
                z = z + mu_steps;
                if (z >= (int)Indexer.getLatData().lzFull)
                    z -= Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+mu_steps) % size.lt();
                t = t + mu_steps;
                if (t >= (int)Indexer.getLatData().ltFull)
                    t -= Indexer.getLatData().ltFull;
                break;
        }
    }
    if (mu_steps < 0) {
        switch (mu) {
            case 0:
                //                      x = (x+mu_steps) % size.lx();
                x = x + mu_steps;
                if (x < 0)
                    x += Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+mu_steps) % size.ly();
                y = y + mu_steps;
                if (y < 0)
                    y += Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+mu_steps) % size.lz();
                z = z + mu_steps;
                if (z < 0)
                    z += Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+mu_steps) % size.lt();
                t = t + mu_steps;
                if (t < 0)
                    t += Indexer.getLatData().ltFull;
                break;
        }
    }
    if (nu_steps > 0) {
        switch (nu) {
            case 0:
                //                      x = (x+nu_steps) % size.lx();
                x = x + nu_steps;
                if (x >= (int)Indexer.getLatData().lxFull)
                    x -= Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+nu_steps) % size.ly();
                y = y + nu_steps;
                if (y >= (int)Indexer.getLatData().lyFull)
                    y -= Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+nu_steps) % size.lz();
                z = z + nu_steps;
                if (z >= (int)Indexer.getLatData().lzFull)
                    z -= Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+nu_steps) % size.lt();
                t = t + nu_steps;
                if (t >= (int)Indexer.getLatData().ltFull)
                    t -= Indexer.getLatData().ltFull;
                break;
        }
    }
    if (nu_steps < 0) {
        switch (nu) {
            case 0:
                //                      x = (x+nu_steps) % size.lx();
                x = x + nu_steps;
                if (x < 0)
                    x += Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+nu_steps) % size.ly();
                y = y + nu_steps;
                if (y < 0)
                    y += Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+nu_steps) % size.lz();
                z = z + nu_steps;
                if (z < 0)
                    z += Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+nu_steps) % size.lt();
                t = t + nu_steps;
                if (t < 0)
                    t += Indexer.getLatData().ltFull;
                break;
        }
    }
    if (rho_steps > 0) {
        switch (rho) {
            case 0:
                //                      x = (x+rho_steps) % size.lx();
                x = x + rho_steps;
                if (x >= (int)Indexer.getLatData().lxFull)
                    x -= Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+rho_steps) % size.ly();
                y = y + rho_steps;
                if (y >= (int)Indexer.getLatData().lyFull)
                    y -= Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+rho_steps) % size.lz();
                z = z + rho_steps;
                if (z >= (int)Indexer.getLatData().lzFull)
                    z -= Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+rho_steps) % size.lt();
                t = t + rho_steps;
                if (t >= (int)Indexer.getLatData().ltFull)
                    t -= Indexer.getLatData().ltFull;
                break;
        }
    }
    if (rho_steps < 0) {
        switch (rho) {
            case 0:
                //                      x = (x+rho_steps) % size.lx();
                x = x + rho_steps;
                if (x < 0)
                    x += Indexer.getLatData().lxFull;
                break;
            case 1:
                //                      y = (y+rho_steps) % size.ly();
                y = y + rho_steps;
                if (y < 0)
                    y += Indexer.getLatData().lyFull;
                break;
            case 2:
                //                      z = (z+rho_steps) % size.lz();
                z = z + rho_steps;
                if (z < 0)
                    z += Indexer.getLatData().lzFull;
                break;
            case 3:
                //                      t = (t+rho_steps) % size.lt();
                t = t + rho_steps;
                if (t < 0)
                    t += Indexer.getLatData().ltFull;
                break;
        }
    }
    sitexyzt fullcoords(x,y,z,t);
    return Indexer.fullCoordToCoord(fullcoords);
}

template<Layout LatLayout, size_t HaloDepth>
bool checkSites(const GIndexer<LatLayout, HaloDepth> Indexer) {

    bool ret_val = true;

    for (size_t x = 0; x<Indexer.getLatData().lx; x++)
        for (size_t y=0; y<Indexer.getLatData().ly; y++)
            for (size_t z=0; z<Indexer.getLatData().lz; z++)
                for (size_t t=0; t<Indexer.getLatData().lt; t++) {

                    bool oddness = (isOdd(x) ^ isOdd(y)) ^ (isOdd(z) ^ isOdd(t));

		            sitexyzt testsite(x,y,z,t);
                    gSite site=Indexer.getSite(x,y,z,t);

                    if(LatLayout==All){
                        if(Indexer.indexToCoord(site.isite) != testsite ||
                           Indexer.fullCoordToCoord(Indexer.indexToCoord_Full(site.isiteFull)) != testsite){
                            ret_val = false;
                        }
                        if(Indexer.getSite(site.isite).coord != testsite)
                            ret_val=false;
                    }
                    else if(LatLayout ==Even && !oddness){
                        if(Indexer.indexToCoord_eo(site.isite, 0) != testsite ||
                           Indexer.fullCoordToCoord(Indexer.indexToCoord_Full_eo(site.isiteFull, 0)) != testsite){
                            ret_val = false;
                        }
                        if(Indexer.getSite(site.isite).coord != testsite)
                            ret_val=false;
                    }
                    else if(LatLayout == Even && oddness){
                        if(Indexer.indexToCoord_eo(site.isite, 1) != testsite ||
                           Indexer.fullCoordToCoord(Indexer.indexToCoord_Full_eo(site.isiteFull, 1)) != testsite){
                            ret_val = false;
                        }
                    }
                    else if(LatLayout == Odd && oddness){
                        if(Indexer.indexToCoord_eo(site.isite, 1) != testsite ||
                           Indexer.fullCoordToCoord(Indexer.indexToCoord_Full_eo(site.isiteFull, 1)) != testsite){
                            ret_val = false;
                        }
                        if(Indexer.getSite(site.isite).coord != testsite)
                            ret_val=false;
                    }
                    else if(LatLayout == Odd && !oddness){
                        if(Indexer.indexToCoord_eo(site.isite, 0) != testsite ||
                           Indexer.fullCoordToCoord(Indexer.indexToCoord_Full_eo(site.isiteFull, 0)) != testsite){
                            ret_val = false;
                        }
                    }
		}
    return ret_val;
}

template<Layout LatLayout, size_t HaloDepth>
bool checkSitesSpatial(const GIndexer<LatLayout, HaloDepth> Indexer) {

    bool ret_val = true;
    size_t t=0; /// Spatial Kernel to be run only on t=0 timeslice.

    for(size_t x=0; x<Indexer.getLatData().lx; x++)
    for(size_t y=0; y<Indexer.getLatData().ly; y++)
    for(size_t z=0; z<Indexer.getLatData().lz; z++){

        bool oddness = (isOdd(x) ^ isOdd(y)) ^ (isOdd(z) ^ isOdd(t));

        sitexyzt testsite(x,y,z,t);
        gSite site=Indexer.getSiteSpatial(x,y,z,t);

        if(LatLayout==All){
            if(Indexer.indexToCoord_Spatial(site.isite) != testsite ||
               Indexer.fullCoordToCoord(Indexer.indexToCoord_SpatialFull(site.isiteFull)) != testsite){
                ret_val=false;
            }
            if(Indexer.getSiteSpatial(site.isite).coord != testsite)
                ret_val=false;
        }
        else if(LatLayout == Even && !oddness){
            if(Indexer.indexToCoord_Spatial_eo(site.isite, 0) != testsite ||
               Indexer.fullCoordToCoord(Indexer.indexToCoord_SpatialFull_eo(site.isiteFull, 0)) != testsite){
                ret_val=false;
            }
            if(Indexer.getSiteSpatial(site.isite).coord != testsite)
                ret_val=false;
            }
        else if(LatLayout == Even && oddness){
            if(Indexer.indexToCoord_Spatial_eo(site.isite, 1) != testsite ||
               Indexer.fullCoordToCoord(Indexer.indexToCoord_SpatialFull_eo(site.isiteFull, 1)) != testsite){
                ret_val=false;
            }
        }
        else if(LatLayout == Odd && oddness){
            if(Indexer.indexToCoord_Spatial_eo(site.isite, 1) != testsite ||
               Indexer.fullCoordToCoord(Indexer.indexToCoord_SpatialFull_eo(site.isiteFull, 1)) != testsite){
                ret_val=false;
            }
            if(Indexer.getSiteSpatial(site.isite).coord != testsite)
                ret_val=false;
        }
        else if(LatLayout == Odd && !oddness){
            if(Indexer.indexToCoord_Spatial_eo(site.isite, 0) != testsite ||
               Indexer.fullCoordToCoord(Indexer.indexToCoord_SpatialFull_eo(site.isiteFull, 0)) != testsite){
                ret_val=false;
            }
        }
    }
    return ret_val;
}

template<Layout LatLayout, size_t HaloDepth>
bool checkIndex(const GIndexer<LatLayout, HaloDepth> Indexer) {

    bool ret_val= true;

    for (size_t x = 0; x<Indexer.getLatData().lx; x++)
        for (size_t y=0; y<Indexer.getLatData().ly; y++)
            for (size_t z=0; z<Indexer.getLatData().lz; z++)
                for (size_t t=0; t<Indexer.getLatData().lt; t++) {

                    sitexyzt testsite(x,y,z,t);
                    sitexyzt fullsite = Indexer.coordToFullCoord(testsite);
                    gSite site = Indexer.getSite(x,y,z,t);

                    bool oddness = (isOdd(x) ^ isOdd(y)) ^ (isOdd(z) ^ isOdd(t));

                    for (int mu = 0; mu < 4; ++mu)
                    {
                        gSiteMu link = Indexer.getSiteMu( site.isite, mu); // This is dangerous and does not work for even odd
                        if (link.mu!=mu)
                            ret_val=false;
                        if (Indexer.coordMuToIndexMu_Full(fullsite.x, fullsite.y, fullsite.z, fullsite.t, mu) != Indexer.getSiteMu(site, mu).indexMuFull)
                            ret_val=false;

                        if (LatLayout==All){
                            if (link.indexMuFull != Indexer.getSiteMu(site, mu).indexMuFull)
                                ret_val=false;
                        }
                        if (LatLayout==Even && !oddness){
                            if (link.indexMuFull != Indexer.getSiteMu(site, mu).indexMuFull)
                                ret_val=false;
                        }
                        if (LatLayout==Odd && oddness){
                            if (link.indexMuFull != Indexer.getSiteMu(site, mu).indexMuFull)
                                ret_val=false;
                        }
                    }
                }
    return ret_val;
}

template<Layout LatLayout, size_t HaloDepth>
bool checkMoves(const GIndexer<LatLayout, HaloDepth> Indexer) {

    bool ret_val = true;

    for (size_t x = 0; x<Indexer.getLatData().lx; x++)
        for (size_t y=0; y<Indexer.getLatData().ly; y++)
            for (size_t z=0; z<Indexer.getLatData().lz; z++)
                for (size_t t=0; t<Indexer.getLatData().lt; t++) {

                    gSite site = Indexer.getSite(x,y,z,t);

                    for (int mu = 0; mu < 4; ++mu){
                        for( int nu = 0; nu < 4; ++nu){
                            for (int rho = 0; rho < 4; ++rho){

                                sitexyzt mycoords = testMove<1,6,9, LatLayout, HaloDepth>(site, mu, nu , rho, Indexer);
                                gSite moved_site = Indexer.template  site_move<1,6,9>(site, mu, nu, rho);

                                if(mycoords!=moved_site.coord)
                                    ret_val = false;

                                gSite updownsite = Indexer.site_dn_dn_dn_dn(Indexer.site_up_up_up_up(site, mu, nu, rho, rho)
                                    , mu, nu, rho, rho );

                                if(site.coord!=updownsite.coord)
                                    ret_val=false;

                                gSite around = Indexer.site_up_up_dn_dn(site, mu, nu, nu, mu);

                                if (site.coord!=around.coord)
                                    ret_val=false;
                            }

                            sitexyzt mycoords = testMove<2,8,0, LatLayout, HaloDepth>(site, mu, nu, 0, Indexer);
                            gSite moved_site = Indexer.template site_move<2,8>(site, mu, nu);

                            if(mycoords!=moved_site.coord)
                                ret_val = false;
                        }

                        sitexyzt mycoords = testMove<5,0,0, LatLayout, HaloDepth>(site, mu, 0, 0, Indexer);
                        gSite moved_site = Indexer.template site_move<5>(site, mu);

                            if(mycoords!=moved_site.coord)
                                ret_val = false;
                    }
                }
    return ret_val;
}


template<class floatT, Layout LatLayout, size_t HaloDepth>
struct do_tests_on_device
{
    do_tests_on_device(){}

    __device__ __host__ floatT operator()(gSite site) {

        typedef GIndexer<LatLayout,HaloDepth> GInd;
        GInd Indexer;

        floatT ret_val = 1.0;

        sitexyzt coordinates = site.coord;
        sitexyzt fullcoords = GInd::coordToFullCoord(coordinates);

        if(GInd::getSite(coordinates.x, coordinates.y, coordinates.z, coordinates.t).isite != site.isite)
            ret_val = 0;

        // bool oddness = (isOdd(coordinates.x) ^ isOdd(coordinates.y)) ^ (isOdd(coordinates.z) ^ isOdd(coordinates.t));

        for (int mu = 0; mu < 4; ++mu)
        {
            gSiteMu link = GInd::getSiteMu( site.isite, mu);

            if (link.mu!=mu)
                ret_val=0;
            if (GInd::coordMuToIndexMu_Full(fullcoords.x, fullcoords.y, fullcoords.z, fullcoords.t, mu) != GInd::getSiteMu(site, mu).indexMuFull)
                ret_val=0;
            if (link.indexMuFull != GInd::getSiteMu(site, mu).indexMuFull)
                ret_val=0;

            for( int nu = 0; nu < 4; ++nu){
                for (int rho = 0; rho < 4; ++rho){

                    sitexyzt mycoords = testMove<1,6,9, LatLayout, HaloDepth>(site, mu, nu , rho, Indexer);
                    gSite moved_site = GInd::template site_move<1,6,9>(site, mu, nu, rho);

                    if(mycoords!=moved_site.coord)
                        ret_val = 0;

                    gSite updownsite = GInd::site_dn_dn_dn_dn(GInd::site_up_up_up_up(site, mu, nu, rho, rho)
                        , mu, nu, rho, rho );

                    if(site.coord!=updownsite.coord)
                        ret_val = 0;

                    gSite around = GInd::site_up_up_dn_dn(site, mu, nu, nu, mu);

                    if (site.coord!=around.coord)
                        ret_val = 0;
                }

                sitexyzt mycoords = testMove<2,8,0, LatLayout, HaloDepth>(site, mu, nu, 0, Indexer);
                gSite moved_site = GInd::template site_move<2,8>(site, mu, nu);

                if(mycoords!=moved_site.coord)
                    ret_val = 0;
            }

            sitexyzt mycoords = testMove<5,0,0, LatLayout, HaloDepth>(site, mu, 0, 0, Indexer);
            gSite moved_site = GInd::template site_move<5>(site, mu);

                if(mycoords!=moved_site.coord)
                    ret_val = 0;
        }
        return ret_val;
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth>
floatT start_tests_on_device(LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<LatLayout,HaloDepth> GInd;
    size_t elems = GInd::getLatData().vol4;
    if(LatLayout==Even || LatLayout==Odd)
        elems = GInd::getLatData().sizeh;

    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<LatLayout, HaloDepth>(do_tests_on_device<floatT, LatLayout, HaloDepth>());

    floatT sum;
    redBase.reduce(sum, elems);

    return sum;
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/bulkIndexerTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 2;

    rootLogger.info("Check indices");
    GIndexer<All,HaloDepth> GInd;
    GIndexer<Even,HaloDepth> GIndEven;
    GIndexer<Odd,HaloDepth> GIndOdd;
    initIndexer(HaloDepth,param,commBase);

/// ================ Full Test ================ ///
    rootLogger.info("----------------------------------------");
    rootLogger.info("           ");
    rootLogger.info("Initialize Lattice with all indices");

    const Layout LatLayout = All;

    bool testDevice = true;

    bool fullTest = checkSites<LatLayout, HaloDepth>(GInd);
    fullTest = fullTest & checkSitesSpatial<LatLayout, HaloDepth>(GInd);
    fullTest = fullTest & checkIndex<LatLayout, HaloDepth>(GInd);
    fullTest = fullTest & checkMoves<LatLayout, HaloDepth>(GInd);

    if(!fullTest)
        rootLogger.error("Test on the host failed");

    LatticeContainer<true,double > redBase(commBase);
    redBase.adjustSize(GInd.getLatData().vol4);

    double sum = 0;
    sum = start_tests_on_device<double, LatLayout, HaloDepth>(redBase);

    if (int(sum) != GInd.getLatData().globalLattice().mult()){
        rootLogger.error("Test on device failed");
        testDevice = false;
    }


/// ================ Even Test ================ ///
    rootLogger.info("----------------------------------------");
    rootLogger.info("           ");
    rootLogger.info("Initialize Lattice with even indices");
    const Layout LatLayout2 = Even;

    bool EvenTest = checkSites<LatLayout2, HaloDepth>(GIndEven);
    EvenTest = checkSitesSpatial<LatLayout2, HaloDepth>(GIndEven);
    EvenTest = EvenTest & checkIndex<LatLayout2, HaloDepth>(GIndEven);
    EvenTest = EvenTest & checkMoves<LatLayout2, HaloDepth>(GIndEven);

    if(!EvenTest)
        rootLogger.error("Test on the host failed");

    sum = start_tests_on_device<double, LatLayout2, HaloDepth>(redBase);

    if (int(sum) != GIndEven.getLatData().globalLattice().mult()/2){
        rootLogger.error("Test on device failed");
        testDevice = false;
    }

/// ================ Odd Test ================ ///
    rootLogger.info("----------------------------------------");
    rootLogger.info("           ");
    rootLogger.info("Initialize Lattice with odd indices");
    const Layout LatLayout3 = Odd;

    bool OddTest = checkSites<LatLayout3, HaloDepth>(GIndOdd);
    OddTest = checkSitesSpatial<LatLayout3, HaloDepth>(GIndOdd);
    OddTest = OddTest & checkIndex<LatLayout3, HaloDepth>(GIndOdd);
    OddTest = OddTest & checkMoves<LatLayout3, HaloDepth>(GIndOdd);

    if(!OddTest)
        rootLogger.error("Test on the host failed");

    sum = start_tests_on_device<double, LatLayout3, HaloDepth>(redBase);

    if (int(sum) != GInd.getLatData().globalLattice().mult()/2){
        rootLogger.error("Test on device failed");
        testDevice = false;
    }

    rootLogger.info("           ");
    if (fullTest && EvenTest && OddTest && testDevice) {
        rootLogger.info("==========================================");
        rootLogger.info("           ");
        rootLogger.info("All Tests passed!");
        rootLogger.info("           ");
        rootLogger.info("==========================================");
    } else {
        rootLogger.error("==========================================");
        rootLogger.error("           ");
        rootLogger.error("At least one test failed!");
        rootLogger.error("           ");
        rootLogger.error("==========================================");
        return -1;
    }

    return 0;
}

