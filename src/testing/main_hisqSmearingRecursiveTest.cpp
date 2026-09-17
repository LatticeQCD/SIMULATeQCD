/*
 * main_HisqSmearingTest.cpp
 *
 */

#include "../simulateqcd.h"
#include "../modules/hisq/hisqSmearing.h"
#include "testing.h"
#include "../modules/hisq/staggeredPhasesKernel.h"
#include <string>
#include <vector>

#define PREC double
#define USE_GPU true

// Recursive path functors are shared with production HisqSmearing.
using hisq_smearing::RecursiveFat7Lvl1;
using hisq_smearing::RecursiveFat7Lvl2;

template <
    class floatT,
    bool onDevice,
    size_t HaloDepth,
    CompressionType compIn,
    CompressionType compOut>
void CurrentSmearLvl2(
    Gaugefield<
        floatT,
        onDevice,
        HaloDepth,
        compIn> &gauge_in,

    Gaugefield<
        floatT,
        onDevice,
        HaloDepth,
        compOut> &gauge_out,

    Gaugefield<
        floatT,
        onDevice,
        HaloDepth> &dummy,

    const SmearingParameters<floatT> &p)
{
    staple<floatT, HaloDepth, compIn, 3>
        staple3(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, -5>
        stapleLepage(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 5, 1>
        staple5_1(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 5, 2>
        staple5_2(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 5, 3>
        staple5_3(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 5, 4>
        staple5_4(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 1>
        staple7_1(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 2>
        staple7_2(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 3>
        staple7_3(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 4>
        staple7_4(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 5>
        staple7_5(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 6>
        staple7_6(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 7>
        staple7_7(gauge_in.getAccessor());

    staple<floatT, HaloDepth, compIn, 7, 8>
        staple7_8(gauge_in.getAccessor());

    // 3-link
    dummy.iterateOverBulkAllMu(staple3);

    gauge_out =
        p._c_1 * gauge_in +
        p._c_3 * dummy;

    // Lepage
    dummy.iterateOverBulkAllMu(stapleLepage);

    gauge_out =
        gauge_out +
        p._c_lp * dummy;

    // 5-link
    dummy.iterateOverBulkAllMu(staple5_1);
    gauge_out =
        gauge_out + p._c_5 * dummy;

    dummy.iterateOverBulkAllMu(staple5_2);
    gauge_out =
        gauge_out + p._c_5 * dummy;

    dummy.iterateOverBulkAllMu(staple5_3);
    gauge_out =
        gauge_out + p._c_5 * dummy;

    dummy.iterateOverBulkAllMu(staple5_4);
    gauge_out =
        gauge_out + p._c_5 * dummy;

    // 7-link
    dummy.iterateOverBulkAllMu(staple7_1);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_2);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_3);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_4);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_5);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_6);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_7);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    dummy.iterateOverBulkAllMu(staple7_8);
    gauge_out =
        gauge_out + p._c_7 * dummy;

    gauge_out.updateAll();
}


int main(int argc, char *argv[])
{
    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    StopWatch<true> timer;

    bool runBenchmark = false;
    std::vector<char *> parameterArgv{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark")
            runBenchmark = true;
        else
            parameterArgv.push_back(argv[i]);
    }
    int parameterArgc = static_cast<int>(parameterArgv.size());

    // ============================================================
    // Read benchmark parameters.
    //
    // The parameter-file lattice and gauge file belong to STAGE 2.
    // STAGE 1 always uses the historical 8^3 x 4 regression test.
    // ============================================================

    param.readfile(
        commBase,
        "../parameter/tests/hisqSmearingRecursiveTest.param",
        parameterArgc,
        parameterArgv.data()
    );

    // Save benchmark lattice before temporarily switching to
    // the historical 8^3 x 4 lattice.
    const int benchmarkLat[4] = {
        param.latDim[0],
        param.latDim[1],
        param.latDim[2],
        param.latDim[3]
    };

    if (!param.GaugefileName.isSet())
    {
        rootLogger.error(
            "No Gaugefile specified in hisqSmearingTest.param!"
        );
        return 1;
    }

    const std::string benchmarkGaugeFile =
        param.GaugefileName();

    commBase.init(
        param.nodeDim()
    );

    constexpr size_t HaloDepth = 0;

    // ============================================================
    //
    // STAGE 1
    //
    // Historical 8^3 x 4 regression test.
    //
    // ============================================================

    rootLogger.info(
        "============================================================"
    );
    rootLogger.info(
        "STAGE 1: LEGACY HISQ REGRESSION TEST"
    );
    rootLogger.info(
        "============================================================"
    );

    {
        const int legacyLat[4] = {
            8, 8, 8, 4
        };

        param.latDim.set(
            legacyLat
        );

        rootLogger.info(
            "Legacy lattice = ",
            legacyLat[0], " x ",
            legacyLat[1], " x ",
            legacyLat[2], " x ",
            legacyLat[3]
        );

        initIndexer(
            HaloDepth,
            param,
            commBase
        );

        // ========================================================
        // Gauge fields
        // ========================================================

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_in(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lv2(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_naik(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl1_recursive(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lv2_recursive(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_naik_recursive(commBase);

        Gaugefield<
            PREC,
            false,
            HaloDepth
        > gauge_reference_host(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_reference_device(commBase);

        // ========================================================
        // Smearing objects
        // ========================================================

        HisqSmearing<
            PREC,
            USE_GPU,
            HaloDepth,
            R18,
            R18,
            R18,
            R18
        > smearing(
            gauge_in,
            gauge_Lv2,
            gauge_naik
        );

        HisqSmearing<
            PREC,
            USE_GPU,
            HaloDepth,
            R18,
            R18,
            R18,
            R18
        > recursiveSmearing(
            gauge_in,
            gauge_Lv2_recursive,
            gauge_naik_recursive
        );

        // ========================================================
        // Read historical files
        // ========================================================

        rootLogger.info(
            "Read legacy input: ../test_conf/gauge12750"
        );

        gauge_in.readconf_nersc(
            "../test_conf/gauge12750"
        );

        gauge_in.updateAll();

        rootLogger.info(
            "Read historical reference: "
            "../test_conf/smearing_reference_conf"
        );

        gauge_reference_host.readconf_nersc(
            "../test_conf/smearing_reference_conf"
        );

        gauge_reference_device =
            gauge_reference_host;

        gauge_reference_device.su3latunitarize();

        // ========================================================
        // Current vs recursive full HISQ
        // ========================================================

        smearing.SmearAllLegacy();

        recursiveSmearing.SmearAll();

        constexpr double LEGACY_TOL =
            1e-11;

        bool pass_recursive_lvl2 =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_Lv2,
                gauge_Lv2_recursive,
                LEGACY_TOL
            );

        bool pass_recursive_naik =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_naik,
                gauge_naik_recursive,
                LEGACY_TOL
            );

        if (!(pass_recursive_lvl2 &&
              pass_recursive_naik))
        {
            rootLogger.error(
                "Legacy current-vs-recursive comparison: FAIL"
            );

            rootLogger.error(
                "Level-2 comparison: ",
                pass_recursive_lvl2
            );

            rootLogger.error(
                "Naik comparison: ",
                pass_recursive_naik
            );

            return 1;
        }

        rootLogger.info(
            CoutColors::green,
            "Legacy current-vs-recursive comparison: PASS",
            CoutColors::reset
        );

        // ========================================================
        // Historical stored-reference comparison
        //
        // Keep the original regression logic.
        // ========================================================

        gauge_Lv2.su3latunitarize();

        bool pass_reference =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_Lv2,
                gauge_reference_device
            );

        if (!pass_reference)
        {
            rootLogger.error(
                "Historical HISQ reference comparison: FAIL"
            );

            return 1;
        }

        rootLogger.info(
            CoutColors::green,
            "Historical HISQ reference comparison: PASS",
            CoutColors::reset
        );

        rootLogger.info(
            CoutColors::green,
            "Historical smearing comparison: PASS",
            CoutColors::reset
        );

        // All legacy gauge fields are destroyed here before
        // reinitializing the indexer for the larger lattice.
    }

    // ============================================================
    //
    // STAGE 2
    //
    // Real large-volume gauge configuration.
    //
    // ============================================================

    rootLogger.info(
        "============================================================"
    );
    rootLogger.info(
        "STAGE 2: REALISTIC LARGE-LATTICE BENCHMARK"
    );
    rootLogger.info(
        "============================================================"
    );

    {
        // ========================================================
        // Restore lattice dimensions from parameter file
        // ========================================================

        param.latDim.set(
            benchmarkLat
        );

        rootLogger.info(
            "Benchmark lattice = ",
            benchmarkLat[0], " x ",
            benchmarkLat[1], " x ",
            benchmarkLat[2], " x ",
            benchmarkLat[3]
        );

        rootLogger.info(
            "Benchmark gauge configuration = ",
            benchmarkGaugeFile
        );

        initIndexer(
            HaloDepth,
            param,
            commBase
        );

        // ========================================================
        // Gauge fields
        // ========================================================

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_in(commBase);

        // Current full HISQ outputs
        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lv2(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_naik(commBase);

        // Recursive full outputs
        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl1_recursive_all(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lv2_recursive_all(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_naik_recursive_all(commBase);

        // Isolated L1
        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl1_current(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl1_recursive(commBase);

        // Common projected L1 field
        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_u3(commBase);

        // Isolated L2
        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl2_current(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl2_recursive(commBase);

        Gaugefield<
            PREC,
            true,
            HaloDepth
        > gauge_Lvl2_dummy(commBase);

        // ========================================================
        // Read realistic gauge configuration
        // ========================================================

        rootLogger.info(
            "Read benchmark gauge configuration"
        );

        gauge_in.readconf_nersc(
            benchmarkGaugeFile
        );

        gauge_in.updateAll();

        // ========================================================
        // Construct smearing objects
        // ========================================================

        HisqSmearing<
            PREC,
            USE_GPU,
            HaloDepth,
            R18,
            R18,
            R18,
            R18
        > smearing(
            gauge_in,
            gauge_Lv2,
            gauge_naik
        );

        HisqSmearing<
            PREC,
            USE_GPU,
            HaloDepth,
            R18,
            R18,
            R18,
            R18
        > recursiveSmearing(
            gauge_in,
            gauge_Lv2_recursive_all,
            gauge_naik_recursive_all
        );

        // ========================================================
        //
        // LEVEL 1 CORRECTNESS
        //
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "LEVEL-1 CORRECTNESS"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        SmearingParameters<PREC> lvl1_params =
            getLevel1Params<PREC>();

        RecursiveFat7Lvl1<
            PREC,
            HaloDepth,
            R18
        > recursiveFat7(
            gauge_in.getAccessor(),
            lvl1_params
        );

        smearing.SmearLvl1Legacy(
            gauge_Lvl1_current
        );

        gauge_Lvl1_recursive.iterateOverBulkAllMu(
            recursiveFat7
        );

        gauge_Lvl1_recursive.updateAll();

        constexpr double PATH_TOL =
            1e-11;

        bool pass_lvl1 =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_Lvl1_current,
                gauge_Lvl1_recursive,
                PATH_TOL
            );

        if (!pass_lvl1)
        {
            rootLogger.error(
                "Large-lattice level-1 comparison: FAIL"
            );

            return 1;
        }

        rootLogger.info(
            CoutColors::green,
            "Large-lattice level-1 comparison: PASS",
            CoutColors::reset
        );

        // ========================================================
        //
        // LEVEL 2 CORRECTNESS
        //
        // Both implementations use exactly the same U(3) field.
        //
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "LEVEL-2 CORRECTNESS"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        smearing.ProjectU3(
            gauge_Lvl1_current,
            gauge_u3
        );

        SmearingParameters<PREC> lvl2_params =
            getLevel2Params<PREC>();

        RecursiveFat7Lvl2<
            PREC,
            HaloDepth,
            R18
        > recursiveFat7Lvl2(
            gauge_u3.getAccessor(),
            lvl2_params
        );

        CurrentSmearLvl2<
            PREC,
            true,
            HaloDepth,
            R18,
            R18
        >(
            gauge_u3,
            gauge_Lvl2_current,
            gauge_Lvl2_dummy,
            lvl2_params
        );

        gauge_Lvl2_recursive.iterateOverBulkAllMu(
            recursiveFat7Lvl2
        );

        gauge_Lvl2_recursive.updateAll();

        bool pass_lvl2 =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_Lvl2_current,
                gauge_Lvl2_recursive,
                PATH_TOL
            );

        if (!pass_lvl2)
        {
            rootLogger.error(
                "Large-lattice level-2 comparison: FAIL"
            );

            return 1;
        }

        rootLogger.info(
            CoutColors::green,
            "Large-lattice level-2 comparison: PASS",
            CoutColors::reset
        );

        // ========================================================
        //
        // FULL HISQ CORRECTNESS
        //
        // Here each implementation performs its own U(3)
        // projection, so allow the slightly looser end-to-end
        // numerical tolerance.
        //
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "FULL HISQ CORRECTNESS"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        smearing.SmearAllLegacy();

        recursiveSmearing.SmearAll();

        constexpr double FULL_TOL =
            1e-9;

        bool pass_full_lvl2 =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_Lv2,
                gauge_Lv2_recursive_all,
                FULL_TOL
            );

        bool pass_full_naik =
            compare_fields<
                PREC,
                HaloDepth,
                true,
                R18
            >(
                gauge_naik,
                gauge_naik_recursive_all,
                FULL_TOL
            );

        if (!(pass_full_lvl2 &&
              pass_full_naik))
        {
            rootLogger.error(
                "Large-lattice full HISQ comparison: FAIL"
            );

            rootLogger.error(
                "Level-2 comparison: ",
                pass_full_lvl2
            );

            rootLogger.error(
                "Naik comparison: ",
                pass_full_naik
            );

            return 1;
        }

        rootLogger.info(
            CoutColors::green,
            "Large-lattice full HISQ comparison: PASS",
            CoutColors::reset
        );

        if (!runBenchmark) {
            rootLogger.info(CoutColors::green,
                            "Recursive HISQ smearing regression: PASS",
                            CoutColors::reset);
            return 0;
        }

        // ========================================================
        //
        // PERFORMANCE BENCHMARK
        //
        // ========================================================

        constexpr int NWARM =
            2;

        constexpr int NREP =
            3;

        // ========================================================
        // LEVEL 1 PERFORMANCE
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "LEVEL-1 PERFORMANCE"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        for (int i = 0; i < NWARM; ++i)
        {
            smearing.SmearLvl1Legacy(
                gauge_Lvl1_current
            );
        }

        for (int i = 0; i < NWARM; ++i)
        {
            gauge_Lvl1_recursive.iterateOverBulkAllMu(
                recursiveFat7
            );

            gauge_Lvl1_recursive.updateAll();
        }

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            smearing.SmearLvl1Legacy(
                gauge_Lvl1_current
            );
        }

        timer.stop();

        const double current_lvl1_ms =
            timer.milliseconds();

        rootLogger.info(
            "Current SmearLvl1, ",
            NREP,
            " iterations: ",
            timer
        );

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            gauge_Lvl1_recursive.iterateOverBulkAllMu(
                recursiveFat7
            );

            gauge_Lvl1_recursive.updateAll();
        }

        timer.stop();

        const double recursive_lvl1_ms =
            timer.milliseconds();

        rootLogger.info(
            "Recursive SmearLvl1, ",
            NREP,
            " iterations: ",
            timer
        );

        // ========================================================
        // LEVEL 2 PERFORMANCE
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "LEVEL-2 PERFORMANCE"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        for (int i = 0; i < NWARM; ++i)
        {
            CurrentSmearLvl2<
                PREC,
                true,
                HaloDepth,
                R18,
                R18
            >(
                gauge_u3,
                gauge_Lvl2_current,
                gauge_Lvl2_dummy,
                lvl2_params
            );
        }

        for (int i = 0; i < NWARM; ++i)
        {
            gauge_Lvl2_recursive.iterateOverBulkAllMu(
                recursiveFat7Lvl2
            );

            gauge_Lvl2_recursive.updateAll();
        }

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            CurrentSmearLvl2<
                PREC,
                true,
                HaloDepth,
                R18,
                R18
            >(
                gauge_u3,
                gauge_Lvl2_current,
                gauge_Lvl2_dummy,
                lvl2_params
            );
        }

        timer.stop();

        const double current_lvl2_ms =
            timer.milliseconds();

        rootLogger.info(
            "Current SmearLvl2, ",
            NREP,
            " iterations: ",
            timer
        );

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            gauge_Lvl2_recursive.iterateOverBulkAllMu(
                recursiveFat7Lvl2
            );

            gauge_Lvl2_recursive.updateAll();
        }

        timer.stop();

        const double recursive_lvl2_ms =
            timer.milliseconds();

        rootLogger.info(
            "Recursive SmearLvl2, ",
            NREP,
            " iterations: ",
            timer
        );

        // ========================================================
        // FULL HISQ PERFORMANCE
        // ========================================================

        rootLogger.info(
            "------------------------------------------------------------"
        );
        rootLogger.info(
            "FULL HISQ PERFORMANCE"
        );
        rootLogger.info(
            "------------------------------------------------------------"
        );

        for (int i = 0; i < NWARM; ++i)
        {
            smearing.SmearAllLegacy();
        }

        for (int i = 0; i < NWARM; ++i)
        {
            recursiveSmearing.SmearAll();
        }

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            smearing.SmearAllLegacy();
        }

        timer.stop();

        const double current_full_ms =
            timer.milliseconds();

        rootLogger.info(
            "Current SmearAll, ",
            NREP,
            " iterations: ",
            timer
        );

        timer.reset();
        timer.start();

        for (int i = 0; i < NREP; ++i)
        {
            recursiveSmearing.SmearAll();
        }

        timer.stop();

        const double recursive_full_ms =
            timer.milliseconds();

        rootLogger.info(
            "Recursive SmearAll, ",
            NREP,
            " iterations: ",
            timer
        );

        // ========================================================
        // Summary
        // ========================================================

        const double speedup_lvl1 =
            current_lvl1_ms /
            recursive_lvl1_ms;

        const double speedup_lvl2 =
            current_lvl2_ms /
            recursive_lvl2_ms;

        const double speedup_full =
            current_full_ms /
            recursive_full_ms;

        rootLogger.info(
            "============================================================"
        );
        rootLogger.info(
            "HISQ RECURSIVE SMEARING BENCHMARK SUMMARY"
        );
        rootLogger.info(
            "============================================================"
        );

        rootLogger.info(
            "Benchmark lattice: ",
            benchmarkLat[0], " x ",
            benchmarkLat[1], " x ",
            benchmarkLat[2], " x ",
            benchmarkLat[3]
        );

        rootLogger.info(
            "Gauge configuration: ",
            benchmarkGaugeFile
        );

        rootLogger.info(
            "Level-1 current:   ",
            current_lvl1_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Level-1 recursive: ",
            recursive_lvl1_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Level-1 speedup:   ",
            speedup_lvl1,
            "x"
        );

        rootLogger.info(
            "Level-2 current:   ",
            current_lvl2_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Level-2 recursive: ",
            recursive_lvl2_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Level-2 speedup:   ",
            speedup_lvl2,
            "x"
        );

        rootLogger.info(
            "Full HISQ current:   ",
            current_full_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Full HISQ recursive: ",
            recursive_full_ms / NREP,
            " ms/call"
        );

        rootLogger.info(
            "Full HISQ speedup:   ",
            speedup_full,
            "x"
        );

        rootLogger.info(
            "Full HISQ runtime reduction: ",
            100.0 *
            (
                1.0 -
                recursive_full_ms /
                current_full_ms
            ),
            "%"
        );

        rootLogger.info(
            "============================================================"
        );

        rootLogger.info(
            CoutColors::green,
            "Large-lattice smearing comparison: PASS",
            CoutColors::reset
        );
    }

    rootLogger.info(
        "============================================================"
    );

    rootLogger.info(
        CoutColors::green,
        "Recursive HISQ smearing regression: PASS",
        CoutColors::reset
    );

    rootLogger.info(
        "============================================================"
    );

    return 0;
}
