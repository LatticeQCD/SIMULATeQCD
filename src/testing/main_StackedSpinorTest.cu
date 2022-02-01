/* 
 * main_StackedSpinorTest.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "testing.h"

//This Dslash is only valid for HaloSize = 3 or 4. No mixed HaloSize possible
template<class floatT, bool onDevice, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepth, size_t NStacks>
struct QuickDslash {

    gVect3arrayAcc<floatT> spinorIn;
    gaugeAccessor<floatT> gAcc;
    gaugeAccessor<floatT> gAccN;

    QuickDslash(
            Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepth, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepth> &gauge,
            Gaugefield<floatT, onDevice, HaloDepth> &gaugeN) :
            spinorIn(spinorIn.getAccessor()),
            gAcc(gauge.getAccessor()),
            gAccN(gaugeN.getAccessor()) {}


    __device__ __host__ inline auto operator()(gSiteStack site) const
    {
        typedef GIndexer<LatLayout, HaloDepth> GInd;


        gVect3<floatT> Stmp(0.0);

        for (int mu = 0; mu < 4; mu++) {

            Stmp += C_1000 * gAcc.getLink(GInd::getSiteMu(site, mu)) *
                    spinorIn.getElement(GInd::site_up(site, mu));
            Stmp -= C_1000 * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, mu), mu)) *
                    spinorIn.getElement(GInd::site_dn(site, mu));

#define C_3000 (-1./48.0)
            Stmp += C_3000 * gAccN.getLink(GInd::getSiteMu(GInd::site_up(site, mu), mu)) *
                    spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));
            Stmp -= C_3000 * gAccN.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)) *
                    spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
        }
        return Stmp;
    }

    auto getAccessor() const
    {
        return *this;
    }

};




template<class floatT, Layout LatLayout, size_t HaloDepth, size_t NStacks, bool onDevice>
struct FillStacks{
    gVect3arrayAcc<floatT> spinorIn;

    FillStacks(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn) : 
                spinorIn(spinorIn.getAccessor()){}

    __host__ __device__ gVect3<floatT> operator()(gSiteStack site){
        gVect3<floatT> ret = static_cast<floatT>(site.stack + 1) * gvect3_one<floatT>();
        return ret;
    }
};





template<class floatT, Layout LatLayout, size_t HaloDepth, size_t NStacks, bool onDevice>
struct FillStacksLoop{
    gVect3arrayAcc<floatT> spinorIn;
    gVect3<floatT> my_vec;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    FillStacksLoop(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn) : 
                spinorIn(spinorIn.getAccessor()){}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
        my_vec = gvect3_one<floatT>();
    }

    __host__ __device__ gVect3<floatT> operator()(gSite& site, size_t stack){
        gSiteStack siteStack = GInd::getSiteStack(site, stack);
        gVect3<floatT> ret = static_cast<floatT>(siteStack.stack + 1) * my_vec;
        return ret;
    }
};




//Run function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void run_func(CommunicationBase &commBase) {

    //Initialization as usual
    const int HaloDepth = 4;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    //Our reduction base, that we will use for many things.
    LatticeContainer<onDevice,floatT> redBase(commBase);

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth> gauge(commBase);
    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gauge.updateAll();

    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    grnd_state<false> h_rand_ref;
    grnd_state<onDevice> d_rand_ref;

    h_rand.make_rng_state(1337);
    h_rand_ref.make_rng_state(1337);

    d_rand = h_rand;
    d_rand_ref = h_rand_ref;

    StaticArray<Spinorfield<floatT, onDevice, LatLayout, HaloDepth>, NStacks> spinorArray1(commBase);
    StaticArray<Spinorfield<floatT, onDevice, LatLayout, HaloDepth>, NStacks> spinorArray2(commBase);
    StaticArray<Spinorfield<floatT, onDevice, LatLayout, HaloDepth>, NStacks> projectToSpinorArray(commBase);
    StaticArray<Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepth>, NStacks> spinorRHSArray(commBase);

    Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> stackedSpinor1(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> stackedSpinor2(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> stackedProjectToSpinor(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepth, NStacks> stackedSpinorRHS(commBase);
    
    //This will hold the result of the projection
    std::vector<GCOMPLEX(floatT)> res_projected(NStacks);

    //For a reference which we use for testing.
    std::vector<GCOMPLEX(floatT)> ref_projected(NStacks);

    GCOMPLEX(floatT) reference;
    if (LatLayout == All) {
        reference = GInd::getLatData().globvol4 * 3;
    } else {
        reference = GInd::getLatData().globvol4 / 2 * 3;
    }
    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i].iterateWithConst(gvect3_one<floatT>());
        res_projected[i] = spinorArray1[i].dotProduct(spinorArray1[i]);

        compare_relative(res_projected[i], reference, 1e-8, 1e-8,
                "Construct from const static array, Stack = " + std::to_string(i));
    }

    stackedSpinor1 = FillStacks<floatT, LatLayout, HaloDepth, NStacks, onDevice>(stackedSpinor1);

    res_projected = stackedSpinor1.dotProductStacked(stackedSpinor1);
    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], reference * (i + 1) * (i + 1), 1e-8, 1e-8,
                "Construct from operator stacked array, Stack = " + std::to_string(i));
    }

    stackedSpinor1.iterateWithConst(gvect3_one<floatT>());

    res_projected = stackedSpinor1.dotProductStacked(stackedSpinor1);
    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], reference, 1e-8, 1e-8,
                "Construct from const stacked array, Stack = " + std::to_string(i));
    }

    stackedSpinor1.iterateOverFullLoopStack(FillStacksLoop<floatT, LatLayout, HaloDepth, NStacks, onDevice>(stackedSpinor1));

    res_projected = stackedSpinor1.dotProductStacked(stackedSpinor1);
    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], reference * (i + 1) * (i + 1), 1e-8, 1e-8,
                "Construct from operator loop stacked array, Stack = " + std::to_string(i));
    }

    stackedSpinor1.gauss(d_rand.state);
    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i].gauss(d_rand_ref.state);
        ref_projected[i] = spinorArray1[i].dotProduct(spinorArray1[i]);
    }

    res_projected = stackedSpinor1.dotProductStacked(stackedSpinor1);
    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "Generate Gauss, Stack = " + std::to_string(i));
    }

    stackedSpinor2.gauss(d_rand.state);
    stackedSpinor1.gauss(d_rand.state);
    stackedProjectToSpinor.gauss(d_rand.state);

    for (size_t i = 0; i < NStacks; i++){
        spinorArray2[i].gauss(d_rand_ref.state);
    }
    //We have to split up the loop in order to ensure same random numbers as in stacked spinor
    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i].gauss(d_rand_ref.state);
    }
    for (size_t i = 0; i < NStacks; i++){
        projectToSpinorArray[i].gauss(d_rand_ref.state);
    }

    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i] = 1.234 * spinorArray2[i] + spinorArray1[i];
        ref_projected[i] = spinorArray1[i].dotProduct(spinorArray1[i]);
    }

    stackedSpinor1 = 1.234 * stackedSpinor2 + stackedSpinor1;
    res_projected = stackedSpinor1.dotProductStacked(stackedSpinor1);

    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "axpy, Stack = " + std::to_string(i));
    }

    stackedSpinorRHS.gauss(d_rand.state);
    for (size_t i = 0; i < NStacks; i++){
        spinorRHSArray[i].gauss(d_rand_ref.state);
    }

    stackedSpinor1.template iterateOverBulk<64>(
            QuickDslash<floatT, onDevice, LatLayout, LatLayoutRHS, HaloDepth, NStacks>(stackedSpinorRHS, gauge, gauge));

    res_projected = stackedProjectToSpinor.dotProductStacked(stackedSpinor1);

    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i].iterateOverBulk(
                QuickDslash<floatT, onDevice, LatLayout, LatLayoutRHS, HaloDepth, 1>(spinorRHSArray[i], gauge, gauge));
        ref_projected[i] = projectToSpinorArray[i].dotProduct(spinorArray1[i]);
    }

    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "Dslash Test, Stack = " + std::to_string(i));
    }

    for (size_t i = 0; i < NStacks; i++){
        spinorArray1[i].iterateWithConst(gvect3_one<floatT>());
        spinorArray1[i].copyFromStackToStack(stackedSpinor1, 0, i);
        res_projected[i] = projectToSpinorArray[i].dotProduct(spinorArray1[i]);
    }

    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "Copy partial from stacked to non-stacked, Stack = " + std::to_string(i));
    }

    stackedSpinor1.iterateWithConst(gvect3_one<floatT>());
    for (size_t i = 0; i < NStacks; i++){
        stackedSpinor1.copyFromStackToStack(spinorArray1[i], i, 0);
    }

    res_projected = stackedProjectToSpinor.dotProductStacked(stackedSpinor1);

    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "Copy partial from non-stacked to stacked, Stack = " + std::to_string(i));
    }

    if (NStacks >= 3){
        stackedSpinor1.template iterateOverBulkAtStack<2>(stackedSpinor1 + stackedSpinor2);
        spinorArray1[2].iterateOverBulk(spinorArray1[2] + spinorArray2[2]);
    }

    res_projected = stackedProjectToSpinor.dotProductStacked(stackedSpinor1);
    for (size_t i = 0; i < NStacks; i++){
        ref_projected[i] = projectToSpinorArray[i].dotProduct(spinorArray1[i]);
    }

    for (size_t i = 0; i < NStacks; i++){
        compare_relative(res_projected[i], ref_projected[i], 1e-8, 1e-8,
                "Update only special stack, Stack = " + std::to_string(i));
    }
}


int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv);
        param.readfile(commBase, "../parameter/tests/StackedSpinorTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 4;
        initIndexer(HaloDepth, param, commBase, true);
        stdLogger.setVerbosity(INFO);

        /// Let's force Halos in all directions, otherwise the test doesn't work... (last parameter)
        rootLogger.info("-------------------------------------");
        rootLogger.info("Running on Device");
        rootLogger.info("-------------------------------------");
        rootLogger.info("Testing All - All");
        rootLogger.info("------------------");
        run_func<double, All, All, 4, true>(commBase);
        rootLogger.info("------------------");
        rootLogger.info("Testing Even - Odd");
        rootLogger.info("------------------");
        run_func<double, Even, Odd, 4, true>(commBase);
        rootLogger.info("------------------");
        rootLogger.info("Testing Odd - Even");
        rootLogger.info("------------------");
        run_func<double, Odd, Even, 4, true>(commBase);

        rootLogger.info("-------------------------------------");
        rootLogger.info("Running on Host");
        rootLogger.info("-------------------------------------");
        rootLogger.info("Testing All - All");
        rootLogger.info("------------------");
        run_func<double, All, All, 4, false>(commBase);
        rootLogger.info("------------------");
        rootLogger.info("Testing Even - Odd");
        rootLogger.info("------------------");
        run_func<double, Even, Odd, 4, false>(commBase);
        rootLogger.info("------------------");
        rootLogger.info("Testing Odd - Even");
        rootLogger.info("------------------");
        run_func<double, Odd, Even, 4, false>(commBase);
    }
    catch (const std::runtime_error &error) {
        return 1;
    }
    return 0;
}