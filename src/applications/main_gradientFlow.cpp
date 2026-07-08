/*
 * main_gradientFlow.cpp
 *
 * Lukas Mazur
 *
 * Application for computing observables on configurations that have been smoothed using the gradient flow. If you have
 * a new observable that you would like to measure on flowed configurations, please add your code here. Try to follow
 * the example of other observables, e.g. by including an optional flag for measuring your observable, outputting using
 * the FileWriter, and so on.
 *
 */

#include <filesystem> 
#include "../simulateqcd.h"
#include "../modules/gradientFlow/gradientFlowParameters.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include "../modules/observables/topology.h"
#include "../modules/observables/weinberg.h"
#include "../modules/observables/blocking.h"
#include "../modules/observables/energyMomentumTensor.h"
#include "../modules/observables/energyMomentumTensorCorrelator.h"
#include "../modules/observables/colorElectricCorr.h"
#include "../modules/observables/colorMagneticCorr.h"
#include "../modules/gaugeFixing/gfix.h"
#include "../modules/gaugeFixing/polyakovLoopCorrelator.h"

namespace fs = std::filesystem;

#define USE_GPU true
//define precision
#if SINGLEPREC
#define PREC float
#else
#define PREC double
#endif


template<typename floatT, bool onDevice, size_t HaloDepth, RungeKuttaMethod RK_method, Force force>
void run(CommunicationBase &commBase, gradientFlowParam<floatT> &lp) {

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, onDevice, HaloDepth> gauge(commBase);
    gradientFlow<floatT, HaloDepth, RK_method, force> gradFlow(gauge, lp.start_step_size(),
                                                                     lp.measurement_intervall()[0],
                                                                     lp.measurement_intervall()[1],
                                                                     lp.necessary_flow_times.get(),
                                                                     lp.measured_corr_flow_times.get(), lp.accuracy());

    //! check for blocking method
    size_t numBlocks=lp.latDim()[0]/size_t(lp.binsize());

    if ( lp.topCharge_imp_block() || lp.shear_bulk_corr_block() ) {
        if (lp.latDim()[0]%(lp.nodeDim()[0]*lp.binsize()) != 0 || lp.latDim()[1]%(lp.nodeDim()[1]*lp.binsize()) != 0 || lp.latDim()[2]%(lp.nodeDim()[2]*lp.binsize()) != 0) {
            throw std::runtime_error(stdLogger.fatal("bin can not span between gpus"));
        }
        if (lp.latDim()[0]/lp.nodeDim()[0]<lp.binsize() || lp.latDim()[1]/lp.nodeDim()[1]<lp.binsize() || lp.latDim()[2]/lp.nodeDim()[2]<lp.binsize()) {
            throw std::runtime_error(stdLogger.fatal("each gpu should be able to hold at least one block. please check your blocksize and nodeDim"));
        }
    }

    //! -------------------------------prepare file output--------------------------------------------------------------

    std::stringstream prefix;
    // std::stringstream prefix, datName, datNameConf, datNameCloverSlices, datNameTopChSlices, datNameTopChSlices_imp, datNameTopChSlices_imp_imp,
    //         datNameBlockShear, datNameBlockBulk, datName_normEMT, datNameColElecCorrSlices_naive, datNameColMagnCorrSlices_naive,
    //         datNamePolyCorrSinglet, datNamePolyCorrOctet, datNamePolyCorrAverage, datNameColElecCorrSlices_clover,
    //         datNameColMagnCorrSlices_clover, datNameBlockTopCharge, datNameEMTU, datNameEMTE, datNameEMTUTimeSlices, datNameEMTETimeSlices,
    //         // datNameHDF5,
    //         datNameRenormPolySuscA, datNameRenormPolySuscL, datNameRenormPolySuscT,
    //         datNameWeinbergSlices, datNameWeinbergSlices_imp, datNameWeinbergSlices_imp_imp;
    // fill stream with 0's
    // datName.fill('0');
    // get the data file name
    if ( lp.RK_method() == "adaptive_stepsize" || lp.RK_method() == "adaptive_stepsize_allgpu" )
        prefix << lp.force() << "Flow_acc" << std::fixed << std::setprecision(6)
               << lp.accuracy() << "_sts" << std::fixed << std::setprecision(6) << lp.start_step_size();
    else {
        prefix << lp.force() << "Flow";
        if (not lp.ignore_fixed_startstepsize()) {
            prefix << "_sts" << std::fixed << std::setprecision(6) << lp.start_step_size();
        }
    }
    fs::path datName = fs::path(lp.measurements_dir()) / (prefix.str() + lp.fileExt());
    fs::path datNameConf = fs::path(lp.measurements_dir()) / ("conf_" + prefix.str() + lp.fileExt());
    fs::path datNameCloverSlices = fs::path(lp.measurements_dir()) / (prefix.str() + "_CloverTimeSlices" + lp.fileExt());
    fs::path datNameTopChSlices = fs::path(lp.measurements_dir()) / (prefix.str() + "_TopChTimeSlices" + lp.fileExt());
    fs::path datNameTopChSlices_imp = fs::path(lp.measurements_dir()) / (prefix.str() + "_TopChTimeSlicesImp" + lp.fileExt());
    fs::path datNameTopChSlices_imp_imp = fs::path(lp.measurements_dir()) / (prefix.str() + "_TopChTimeSlicesImpImp" + lp.fileExt());
    fs::path datNameWeinbergSlices = fs::path(lp.measurements_dir()) / (prefix.str() + "_WeinbergTimeSlices" + lp.fileExt());
    fs::path datNameWeinbergSlices_imp = fs::path(lp.measurements_dir()) / (prefix.str() + "_WeinbergTimeSlicesImp" + lp.fileExt());
    fs::path datNameWeinbergSlices_imp_imp = fs::path(lp.measurements_dir()) / (prefix.str() + "_WeinbergTimeSlicesImpImp" + lp.fileExt());
    fs::path datNameColElecCorrSlices_naive = fs::path(lp.measurements_dir()) / (prefix.str() + "_ColElecCorrTimeSlices_naive" + lp.fileExt());
    fs::path datNameColMagnCorrSlices_naive = fs::path(lp.measurements_dir()) / (prefix.str() + "_ColMagnCorrTimeSlices_naive" + lp.fileExt());
    fs::path datNamePolyCorrSinglet = fs::path(lp.measurements_dir()) / (prefix.str() + "_PolyakovCorrSinglet" + lp.fileExt());
    fs::path datNamePolyCorrOctet = fs::path(lp.measurements_dir()) / (prefix.str() + "_PolyakovCorrOctet" + lp.fileExt());
    fs::path datNamePolyCorrAverage = fs::path(lp.measurements_dir()) / (prefix.str() + "_PolyakovCorrAverage" + lp.fileExt());
    fs::path datNameRenormPolySuscA = fs::path(lp.measurements_dir()) / (prefix.str() + "_RenormPolySuscA" + lp.fileExt());
    fs::path datNameRenormPolySuscL = fs::path(lp.measurements_dir()) / (prefix.str() + "_RenormPolySuscL" + lp.fileExt());
    fs::path datNameRenormPolySuscT = fs::path(lp.measurements_dir()) / (prefix.str() + "_RenormPolySuscT" + lp.fileExt());
    fs::path datNameBlockTopCharge = fs::path(lp.measurements_dir()) / (prefix.str() + "_BlockTopCharge" + lp.fileExt());
    fs::path datNameBlockShear = fs::path(lp.measurements_dir()) / (prefix.str() + "_BlockShear" + lp.fileExt());
    fs::path datNameBlockBulk = fs::path(lp.measurements_dir()) / (prefix.str() + "_BlockBulk" + lp.fileExt());
    fs::path datName_normEMT = fs::path(lp.measurements_dir()) / (prefix.str() + "_NormEMT" + lp.fileExt());
    fs::path datNameColElecCorrSlices_clover = fs::path(lp.measurements_dir()) / (prefix.str() + "_ColElecCorrTimeSlices_clover" + lp.fileExt());
    fs::path datNameColMagnCorrSlices_clover = fs::path(lp.measurements_dir()) / (prefix.str() + "_ColMagnCorrTimeSlices_clover" + lp.fileExt());
    fs::path datNameEMTU = fs::path(lp.measurements_dir()) / (prefix.str() + "_EMTU" + lp.fileExt());
    fs::path datNameEMTE = fs::path(lp.measurements_dir()) / (prefix.str() + "_EMTE" + lp.fileExt());
    fs::path datNameEMTUTimeSlices = fs::path(lp.measurements_dir()) / (prefix.str() + "_EMTUTimeSlices" + lp.fileExt());
    fs::path datNameEMTETimeSlices = fs::path(lp.measurements_dir()) / (prefix.str() + "_EMTETimeSlices" + lp.fileExt());
    fs::path datNameHDF5 = fs::path(lp.measurements_dir()) / (prefix.str() + lp.fileExt() + ".h5");
    
    FileWriter file(gauge.getComm(), lp);

    if (!lp.useHDF5()) {
        file.createFile(datName.string());

        //! write header
        LineFormatter header = file.header();
        header << "Flow time ";
        if (lp.plaquette()) header << "Plaquette ";
        if (lp.clover()) header << "Clover ";
        if (lp.topCharge()) header << "Top. Charge ";
        if (lp.topCharge_imp() || lp.topCharge_imp_block()) header << "Impr. top. Charge ";
        if (lp.topCharge_imp_imp() || lp.topCharge_imp_imp_block()) header << "O(a^6) Impr. top. Charge ";
        if (lp.weinberg()) header << "Weinberg ";
        if (lp.weinberg_imp()) header << "Impr. Weinberg ";
        if (lp.weinberg_imp_imp()) header << "O(a^6) Impr. Weinberg ";
        header.endLine();
    }

    // TODO: Why always .getComm instead of using the commBase?
    FileWriter file_BlockTopCharge(gauge.getComm(), lp);
    FileWriter file_normEMT(gauge.getComm(), lp);
    FileWriter file_BlockShear(gauge.getComm(), lp);
    FileWriter file_BlockBulk(gauge.getComm(), lp);
    if (lp.shear_bulk_corr_block()) {
        file_normEMT.createFile(datName_normEMT.string());
        file_BlockShear.createFile(datNameBlockShear.string());
        file_BlockBulk.createFile(datNameBlockBulk.string());
        LineFormatter header_normEMT = file_normEMT.header();
        header_normEMT << "#flowtime E U00 U11 U22 U33 U01 U02 U03 U12 U13 U23" << "\n";
        header_normEMT.endLine();
        LineFormatter header_BlockShear = file_BlockShear.header();
        header_BlockShear << "#flowtime tau/a=0: #r/a1 #corr1 #r/a2 #corr2.... tau/a=1: #r/a1 #corr1...." << "\n";
        header_BlockShear.endLine();
        LineFormatter header_BlockBulk = file_BlockBulk.header();
        header_BlockBulk<< "#flowtime tau/a=0: #r/a1 #corr1 #r/a2 #corr2.... tau/a=1: #r/a1 #corr1...." << "\n";
        header_BlockBulk.endLine();
    }

    FileWriter file_EMTE(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTracefull() && !lp.useHDF5()) {
        file_EMTE.createFile(datNameEMTE.string());
        LineFormatter header_EMTE = file_EMTE.header();
        header_EMTE << "#flowtime E" << "\n";
        header_EMTE.endLine();
    }

    FileWriter file_EMTU(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTraceless() && !lp.useHDF5()) {
        file_EMTU.createFile(datNameEMTU.string());
        LineFormatter header_EMTU = file_EMTU.header();
        header_EMTU << "#flowtime U00 U11 U22 U33 U01 U02 U03 U12 U13 U23" << "\n";
        header_EMTU.endLine();
    }

    FileWriter file_EMTUTimeSlices(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTracelessTimeSlices()) {
        file_EMTUTimeSlices.createFile(datNameEMTUTimeSlices.string());
        LineFormatter header_EMTUTimeSlices = file_EMTUTimeSlices.header();
        header_EMTUTimeSlices << "#flowtime U00 U11 U22 U33 U01 U02 U03 U12 U13 U23 for tau=0, ... for tau=1 ..." << "\n";
        header_EMTUTimeSlices.endLine();
    }

    FileWriter file_EMTETimeSlices(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTracefullTimeSlices()) {
        file_EMTETimeSlices.createFile(datNameEMTETimeSlices.string());
        LineFormatter header_EMTETimeSlices = file_EMTETimeSlices.header();
        header_EMTETimeSlices << "#flowtime E for tau=0, ... for tau=1 ..." << "\n";
        header_EMTETimeSlices.endLine();
    }

    if (lp.topCharge_imp_block()) {
        file_BlockTopCharge.createFile(datNameBlockTopCharge.string());
        LineFormatter header_BlockTopCharge = file_BlockTopCharge.header();
        header_BlockTopCharge<< "#flow time tau/a=0: #r/a1 #corr1 #r/a2 #corr2.... tau/a=1: #r/a1 #corr1...." << "\n";
        header_BlockTopCharge.endLine();
    }

    FileWriter filePolyCorrSinglet(gauge.getComm(), lp);
    FileWriter filePolyCorrOctet(gauge.getComm(), lp);
    FileWriter filePolyCorrAverage(gauge.getComm(), lp);
    if (lp.PolyakovLoopCorrelator()) {
        filePolyCorrSinglet.createFile(datNamePolyCorrSinglet.string());
        filePolyCorrOctet.createFile(datNamePolyCorrOctet.string());
        filePolyCorrAverage.createFile(datNamePolyCorrAverage.string());
    }

    FileWriter fileRenormPolySuscA(gauge.getComm(), lp);
    FileWriter fileRenormPolySuscL(gauge.getComm(), lp);
    FileWriter fileRenormPolySuscT(gauge.getComm(), lp);
    if (lp.RenormPolyakovSusc()) {
        fileRenormPolySuscA.createFile(datNameRenormPolySuscA.string());
        fileRenormPolySuscL.createFile(datNameRenormPolySuscL.string());
        fileRenormPolySuscT.createFile(datNameRenormPolySuscT.string());
    }

    FileWriter fileCloverSl(gauge.getComm(), lp);
    if (lp.cloverTimeSlices()) {
        fileCloverSl.createFile(datNameCloverSlices.string());
        LineFormatter headerClSl = fileCloverSl.header();
        headerClSl << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerClSl << "Nt=" + std::to_string(nt) + " ";
        }
        headerClSl.endLine();
    }

    FileWriter fileTopChSl(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices()) {
        fileTopChSl.createFile(datNameTopChSlices.string());
        LineFormatter headerThSl = fileTopChSl.header();
        headerThSl << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl.endLine();
    }

    FileWriter fileTopChSl_imp(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices_imp()) {
        fileTopChSl_imp.createFile(datNameTopChSlices_imp.string());
        LineFormatter headerThSl_imp = fileTopChSl_imp.header();
        headerThSl_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp.endLine();
    }

    FileWriter fileTopChSl_imp_imp(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices_imp_imp()) {
        fileTopChSl_imp_imp.createFile(datNameTopChSlices_imp_imp.string());
        LineFormatter headerThSl_imp_imp = fileTopChSl_imp_imp.header();
        headerThSl_imp_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp_imp.endLine();
    }

    FileWriter fileWeinbergSl(gauge.getComm(), lp);
    if (lp.weinbergTimeSlices()) {
        fileWeinbergSl.createFile(datNameWeinbergSlices.string());
        LineFormatter headerThSl = fileWeinbergSl.header();
        headerThSl << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl.endLine();
    }
    
    FileWriter fileWeinbergSl_imp(gauge.getComm(), lp);
    if (lp.weinbergTimeSlices_imp()) {
        fileWeinbergSl_imp.createFile(datNameWeinbergSlices_imp.string());
        LineFormatter headerThSl_imp = fileWeinbergSl_imp.header();
        headerThSl_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp.endLine();
    }

    FileWriter fileWeinbergSl_imp_imp(gauge.getComm(), lp);
    if (lp.weinbergTimeSlices_imp_imp()) {
        fileWeinbergSl_imp_imp.createFile(datNameWeinbergSlices_imp_imp.string());
        LineFormatter headerThSl_imp_imp = fileWeinbergSl_imp_imp.header();
        headerThSl_imp_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp_imp.endLine();
    }
    
    FileWriter fileColElecCorrSl_naive(gauge.getComm(), lp);
    if (lp.ColorElectricCorrTimeSlices_naive()) {
        fileColElecCorrSl_naive.createFile(datNameColElecCorrSlices_naive.string());
        LineFormatter headerColElecCorrSl_naive = fileColElecCorrSl_naive.header();
        headerColElecCorrSl_naive << "Flow time ";
        headerColElecCorrSl_naive << "Re(PolyLoop) ";
        headerColElecCorrSl_naive << "Im(PolyLoop) ";
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColElecCorrSl_naive << "dt=" + std::to_string(dt) + "_real ";
        }
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColElecCorrSl_naive << "dt=" + std::to_string(dt) + "_imag ";
        }
        headerColElecCorrSl_naive.endLine();
    }

    FileWriter fileColMagnCorrSl_naive(gauge.getComm(), lp);
    if (lp.ColorMagneticCorrTimeSlices_naive()) {
        fileColMagnCorrSl_naive.createFile(datNameColMagnCorrSlices_naive.string());
        LineFormatter headerColMagnCorrSl_naive = fileColMagnCorrSl_naive.header();
        headerColMagnCorrSl_naive << "Flow time ";
        headerColMagnCorrSl_naive << "Re(PolyLoop) ";
        headerColMagnCorrSl_naive << "Im(PolyLoop) ";
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColMagnCorrSl_naive << "dt=" + std::to_string(dt) + "_real ";
        }
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColMagnCorrSl_naive << "dt=" + std::to_string(dt) + "_imag ";
        }
        headerColMagnCorrSl_naive.endLine();
    }

    FileWriter fileColElecCorrSl_clover(gauge.getComm(), lp);
    if (lp.ColorElectricCorrTimeSlices_clover()) {
        fileColElecCorrSl_clover.createFile(datNameColElecCorrSlices_clover.string());
        LineFormatter headerColElecCorrSl_clover = fileColElecCorrSl_clover.header();
        headerColElecCorrSl_clover << "Flow time ";
        headerColElecCorrSl_clover << "Re(PolyLoop) ";
        headerColElecCorrSl_clover << "Im(PolyLoop) ";
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColElecCorrSl_clover << "dt=" + std::to_string(dt) + "_real ";
        }
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColElecCorrSl_clover << "dt=" + std::to_string(dt) + "_imag ";
        }
        headerColElecCorrSl_clover.endLine();
    }

    FileWriter fileColMagnCorrSl_clover(gauge.getComm(), lp);
    if (lp.ColorMagneticCorrTimeSlices_clover()) {
        fileColMagnCorrSl_clover.createFile(datNameColMagnCorrSlices_clover.string());
        LineFormatter headerColMagnCorrSl_clover = fileColMagnCorrSl_clover.header();
        headerColMagnCorrSl_clover << "Flow time ";
        headerColMagnCorrSl_clover << "Re(PolyLoop) ";
        headerColMagnCorrSl_clover << "Im(PolyLoop) ";
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColMagnCorrSl_clover << "dt=" + std::to_string(dt) + "_real ";
        }
        for (int dt = 1; dt <= lp.latDim[3]/2; dt++) {
            headerColMagnCorrSl_clover << "dt=" + std::to_string(dt) + "_imag ";
        }
        headerColMagnCorrSl_clover.endLine();
    }

    // hdf5 file for correlators and other quantities
    HDF5FileWriter<PREC> hdf5File(commBase, lp, datNameHDF5.string());

    hdf5File.writeAttributes(lp);

    //! -------------------------------read in configuration------------------------------------------------------------

    if (lp.use_unit_conf()){
        rootLogger.info("Using unit configuration for tests/benchmarks");
        gauge.one();
    } else {
        if (lp.format() == "nersc") {
            gauge.readconf_nersc(lp.GaugefileName());
        } else if (lp.format() == "ildg") {
            gauge.readconf_ildg(lp.GaugefileName());
        } else if (lp.format() == "milc") {
            gauge.readconf_milc(lp.GaugefileName()); 
        } else if (lp.format() == "openqcd") {
            gauge.readconf_openqcd(lp.GaugefileName());
        } else {
            throw (std::runtime_error(rootLogger.fatal("Invalid specification for format ", lp.format())));
        }
    }
    gauge.updateAll();

    //! -------------------------------set up observable measurement classes--------------------------------------------

    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);
    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);
    Weinberg<floatT, USE_GPU, HaloDepth> weinberg(gauge);
    EnergyMomentumTensor<floatT, USE_GPU, HaloDepth> EMT(gauge);
    EnergyMomentumTensorCorrelator<floatT, HaloDepth> EMTCorr(commBase);

    BlockingMethod<floatT, true, HaloDepth, floatT, topChargeDens_imp<floatT, HaloDepth, true>, CorrType<floatT>> BlockTopChDens(gauge);
    BlockingMethod<floatT, true, HaloDepth, floatT, EMTtrace<floatT, true, HaloDepth>, CorrType<floatT>> BlockBulk(gauge);
    BlockingMethod<floatT, true, HaloDepth, Matrix4x4Sym<floatT>, EMTtraceless<floatT, true, HaloDepth>, CorrType<floatT>> BlockShear(gauge);

    ColorElectricCorr<floatT, USE_GPU, HaloDepth> CEC(gauge);
    ColorMagneticCorr<floatT, USE_GPU, HaloDepth> CMC(gauge);
    PolyakovLoop<floatT, USE_GPU, HaloDepth> poly(gauge);
    GaugeFixing<floatT,true,HaloDepth> gFix(gauge);
    PolyakovLoopCorrelator<floatT,true,HaloDepth> PLC(gauge);
    CorrelatorTools<floatT,true,HaloDepth> corrTools;

    //! -------------------------------variables for the observables----------------------------------------------------

    floatT plaq, clov, topChar, wb, resultEMTE;
    Matrix4x4Sym<floatT> resultEMTU;
    std::vector<floatT> resultClSl, resultThSl, resultThSl_imp, resultThSl_imp_imp, resultEMTETimeSlices;
    std::vector<Matrix4x4Sym<floatT>> resultEMTUTimeSlices;
    std::vector<COMPLEX(floatT)> resultColElecCorSl_naive, resultColMagnCorSl_naive, resultColElecCorSl_clover,
                                  resultColMagnCorSl_clover;

    COMPLEX(floatT) resultPoly;

    std::vector<floatT> vec_plca, vec_plc1, vec_plc8;
    std::vector<int>    vec_factor, vec_weight;
    if (lp.PolyakovLoopCorrelator()) {
        vec_plca   = std::vector<floatT>(corrTools.distmax);
        vec_plc1   = std::vector<floatT>(corrTools.distmax);
        vec_plc8   = std::vector<floatT>(corrTools.distmax);
        vec_factor = std::vector<int>(corrTools.distmax);
        vec_weight = std::vector<int>(corrTools.distmax);
        corrTools.getFactorArray(vec_factor, vec_weight);
    }
    
    std::vector<int> vecCounts;
    if (lp.energyMomentumTensorCorrFunctionsAveragedTau() || lp.energyMomentumTensorCorrFunctionsGeneralTau()) {
        hsize_t r2max = EMTCorr.getR2max();
        vecCounts = std::vector<int>(r2max+1);
        EMTCorr.getR2Counts(vecCounts);
    
        if (lp.useHDF5()) {
            hdf5File.writeR2Counts(vecCounts);
        }
    }

    std::vector<std::vector<COMPLEX(PREC)>> vecEMTCorrAveragedTau;
    if (lp.energyMomentumTensorCorrFunctionsAveragedTau()) {
        hsize_t r2max = EMTCorr.getR2max();
        vecEMTCorrAveragedTau = std::vector<std::vector<COMPLEX(PREC)>>(10, std::vector<COMPLEX(PREC)>(r2max+1));
    }
    
    std::vector<std::vector<std::vector<COMPLEX(PREC)>>> vecEMTCorrGeneralTau;
    if (lp.energyMomentumTensorCorrFunctionsGeneralTau()) {
        hsize_t r2max = EMTCorr.getR2max();
        vecEMTCorrGeneralTau = std::vector<std::vector<std::vector<COMPLEX(PREC)>>>(14, std::vector<std::vector<COMPLEX(PREC)>>(lp.latDim()[3], std::vector<COMPLEX(PREC)>(r2max+1)));
        rootLogger.info("Time extend of the lattice is " + std::to_string(lp.latDim()[3]));
    }

    std::vector<Matrix4x4Sym<floatT>> EMTUBlock(numBlocks*numBlocks*numBlocks*lp.latDim()[3]);
    std::vector<floatT> EMTEBlock(numBlocks*numBlocks*numBlocks*lp.latDim()[3]);
    std::vector<floatT> ShearCorr, BulkCorr;
    floatT EnergyDensity;
    Matrix4x4Sym<floatT> EMTensorTraceless;

    std::vector<floatT> TopChargeBlock(numBlocks*numBlocks*numBlocks*lp.latDim()[3]);
    std::vector<floatT> TopChargeDensCorr;
    size_t Rsq_size = (numBlocks/2+1)*(numBlocks/2+1)*3;
    floatT TopologicalCharge;

    floatT norm, suscL, suscT;
    COMPLEX(floatT) suscA;

    //! -------------------------------flow the field until max flow time-----------------------------------------------

    std::stringstream logStream;
    StopWatch<true> timer;
    timer.start();
    floatT flow_time = 0.;
    bool continueFlow = true;

    while (continueFlow) {

        //! -------------------------------prepare log output-----------------------------------------------------------

        logStream.str("");
        logStream << std::fixed << std::setprecision(7) << "   t = " << flow_time << ": ";

        //! -------------------------------calculate observables on flowed field----------------------------------------

        if (lp.save_conf() && gradFlow.checkIfMeasuredTime()){
            gauge.writeconf_nersc( datNameConf.string() + "_FT" + std::to_string(flow_time));
        }

        LineFormatter newLine = file.tag("");
        
        // write flow time
        if (lp.useHDF5()) {
            hdf5File.writeObservable<HDF5_Observable::FlowTime>(flow_time);
            if (gradFlow.checkIfMeasuredTime()) {
                hdf5File.writeObservable<HDF5_Observable::FlowTimeMeasured>(flow_time);
            }
        } else {
            newLine << flow_time;
        }

        if (lp.plaquette()) {
            plaq = gAction.plaquette();

            if (lp.useHDF5()) {
                hdf5File.writeObservable<HDF5_Observable::Plaquette>(plaq);
            } else {
                newLine << plaq;
            }

            logStream << std::fixed << std::setprecision(6) << "   Plaquette = " << plaq;
        }

        if (lp.cloverTimeSlices()) {
            LineFormatter newLineCl = fileCloverSl.tag("");
            gAction.cloverTimeSlices(resultClSl);
            newLineCl << flow_time;
            for (auto &elem : resultClSl) {
                newLineCl << elem;
            }
            gAction.dontRecomputeField();
        }

        if (lp.clover()) {
            clov = gAction.clover();
            
            logStream << std::fixed << std::setprecision(6) << "   Clover = " << clov;
            if (lp.useHDF5()) {
                hdf5File.writeObservable<HDF5_Observable::Clover>(clov);
            } else {
                newLine << clov;
            }

            gAction.recomputeField();
        }

        if (lp.topChargeTimeSlices()) {
            LineFormatter newLineTh = fileTopChSl.tag("");
            topology.topChargeTimeSlices(resultThSl);
            newLineTh << flow_time;
            for (auto &elem : resultThSl) {
                newLineTh << elem;
            }
            topology.dontRecomputeField();
        }

        if (lp.topCharge()) {
            topChar = topology.topCharge();
            logStream << std::scientific << std::setprecision(14) << "   topCharge = " << topChar;
            // logStream << std::fixed << std::setprecision(6) << "   topCharge = " << topChar;
            topology.recomputeField();
            
            if (lp.useHDF5()) {
                hdf5File.writeObservable<HDF5_Observable::TopCharge>(topChar);
            } else {
                newLine << topChar;
            }
        }

        if (lp.topChargeTimeSlices_imp()) {
            LineFormatter newLineTh = fileTopChSl_imp.tag("");
            topology.template topChargeTimeSlices<true,false>(resultThSl_imp);
            newLineTh << flow_time;
            logStream << "   topCharge_imp TimeSlices = ";
            logStream << std::scientific << std::setprecision(14) << resultThSl_imp[0];
            for (auto &elem : resultThSl_imp) {
                newLineTh << elem;
            }
            topology.dontRecomputeField();
        }

        if (lp.topCharge_imp() && !lp.topCharge_imp_block()) {
            topChar = topology.template topCharge<true,false>();
            
            logStream << std::scientific << std::setprecision(14) << "   topCharge_imp = " << topChar;
            // logStream << std::fixed << std::setprecision(6) << "   topCharge_imp = " << topChar;
            if (lp.useHDF5()) {
                hdf5File.writeObservable<HDF5_Observable::TopChargeImp>(topChar);
            } else {
                newLine << topChar;
            }
            
            topology.recomputeField();
        }

        if (lp.topChargeTimeSlices_imp_imp()) {
            LineFormatter newLineTh = fileTopChSl_imp_imp.tag("");
//            std::cout << "topChargeTimeSlices_imp_imp" << std::endl;
            topology.template topChargeTimeSlices<false,true>(resultThSl_imp_imp);
            newLineTh << flow_time;
            logStream << "   topCharge_imp_imp TimeSlices = ";
            logStream << std::scientific << std::setprecision(14) << resultThSl_imp_imp[0];
            for (auto &elem : resultThSl_imp_imp) {
                newLineTh << elem;
            }
            topology.dontRecomputeField();
        }

        if (lp.topCharge_imp_imp()) {
            topChar = topology.template topCharge<false,true>();
            logStream << std::scientific << std::setprecision(14) << "   topCharge_imp_imp = " << topChar;
            newLine << topChar;
            topology.recomputeField();
        }

        if (lp.topCharge_imp_block()) {
            TopologicalCharge = BlockTopChDens.updateBlock(TopChargeBlock, lp.binsize());
            TopChargeDensCorr = BlockTopChDens.getCorr(TopChargeBlock, lp.binsize());

            LineFormatter newLine_BlockTopCharge = file_BlockTopCharge.tag("");
            newLine_BlockTopCharge << flow_time << " ";

            for (size_t i=0;i<TopChargeDensCorr.size();i++) {
                if (fabs(TopChargeDensCorr[i])>1e-50) {
                    newLine_BlockTopCharge << sqrt(i%Rsq_size)*lp.binsize() << " " << std::scientific << std::setprecision(15) << TopChargeDensCorr[i] <<" ";
                }
            }
            newLine_BlockTopCharge << "\n";

            logStream << std::fixed << std::setprecision(6) << "   topCharge_imp = " << TopologicalCharge;
            newLine << TopologicalCharge;
        }

      
        if (lp.weinbergTimeSlices()) {
            LineFormatter newLineTh = fileWeinbergSl.tag("");
            weinberg.WBTimeSlices(resultThSl);
            newLineTh << flow_time;
            for (auto &elem : resultThSl) {
                newLineTh << elem;
            }
            weinberg.dontRecomputeField();
        }

        if (lp.weinberg()) {
            wb = weinberg.WB();
            logStream << std::scientific << std::setprecision(14) << "   Weinberg = " << wb;
//            logStream << std::fixed << std::setprecision(6) << "   topCharge = " << topChar;
            newLine << wb;
            weinberg.recomputeField();
        }

        if (lp.weinbergTimeSlices_imp()) {
            LineFormatter newLineTh = fileWeinbergSl_imp.tag("");
            weinberg.template WBTimeSlices<true,false>(resultThSl_imp);
            newLineTh << flow_time;
            logStream << "   Weinberg_imp TimeSlices = ";
            logStream << std::scientific << std::setprecision(14) << resultThSl_imp[0];
            for (auto &elem : resultThSl_imp) {
                newLineTh << elem;
            }
            weinberg.dontRecomputeField();
        }

        if (lp.weinberg_imp()) {
            wb = weinberg.template WB<true,false>();
            logStream << std::scientific << std::setprecision(14) << "   Weinberg_imp = " << topChar;
//            logStream << std::fixed << std::setprecision(6) << "   topCharge_imp = " << topChar;
            newLine << wb;
            weinberg.recomputeField();
        }

        if (lp.weinbergTimeSlices_imp_imp()) {
            LineFormatter newLineTh = fileWeinbergSl_imp_imp.tag("");
            weinberg.template WBTimeSlices<false,true>(resultThSl_imp_imp);
            newLineTh << flow_time;
            logStream << "   Weinberg_imp_imp TimeSlices = ";
            logStream << std::scientific << std::setprecision(14) << resultThSl_imp_imp[0];
            for (auto &elem : resultThSl_imp_imp) {
                newLineTh << elem;
            }
            weinberg.dontRecomputeField();
        }

        if (lp.weinberg_imp_imp()) {
            wb = weinberg.template WB<false,true>();
            logStream << std::scientific << std::setprecision(14) << "   Weinberg_imp_imp = " << topChar;
            newLine << wb;
            weinberg.recomputeField();
        }

        if (lp.energyMomentumTensorTracefull()) {
            EMT.EMTEAveraged(resultEMTE);
            if (lp.useHDF5()) {
                hdf5File.writeObservable<HDF5_Observable::EMTE>(resultEMTE);
            } else {
                LineFormatter newLineEMTE = file_EMTE.tag("");
                newLineEMTE << flow_time << " ";
                newLineEMTE << std::scientific << std::setprecision(15) << resultEMTE << " ";
            }
        }

        if (lp.energyMomentumTensorTraceless()) {
            EMT.EMTUAveraged(resultEMTU);

            if (lp.useHDF5()) {
                hdf5File.writeEMTU(resultEMTU.toStdVector());
            } else {
                LineFormatter newLineEMTU = file_EMTU.tag("");
                newLineEMTU << flow_time << " ";
                newLineEMTU << std::scientific << std::setprecision(15) << resultEMTU.elems[0] << " "
                            << resultEMTU.elems[1] << " " << resultEMTU.elems[2] << " " << resultEMTU.elems[3] << " "
                            << resultEMTU.elems[4] << " " << resultEMTU.elems[5] << " " << resultEMTU.elems[6] << " "
                            << resultEMTU.elems[7] << " " << resultEMTU.elems[8] << " " << resultEMTU.elems[9] << " ";
            }
        }

        if (lp.energyMomentumTensorTracelessTimeSlices() && gradFlow.checkIfMeasuredTime()) {
            LineFormatter newLineEMTUTimeSlices = file_EMTUTimeSlices.tag("");
            EMT.EMTUTimeSlices(resultEMTUTimeSlices);
            newLineEMTUTimeSlices << flow_time << " ";
            for (auto &elem : resultEMTUTimeSlices) {
                newLineEMTUTimeSlices << std::scientific << std::setprecision(15) << elem.elems[0] << " "
                                  << elem.elems[1] << " " << elem.elems[2] << " " << elem.elems[3] << " "
                                  << elem.elems[4] << " " << elem.elems[5] << " " << elem.elems[6] << " "
                                  << elem.elems[7] << " " << elem.elems[8] << " " << elem.elems[9] << " ";
            }
        }

        if (lp.energyMomentumTensorTracefullTimeSlices() && gradFlow.checkIfMeasuredTime()) {
            LineFormatter newLineEMTETimeSlices = file_EMTETimeSlices.tag("");
            EMT.EMTETimeSlices(resultEMTETimeSlices);
            newLineEMTETimeSlices << flow_time << " ";
            for (auto &elem : resultEMTETimeSlices) {
                newLineEMTETimeSlices << std::scientific << std::setprecision(15) << elem << " ";
            }
        }

        if (lp.energyMomentumTensorCorrFunctionsAveragedTau() && gradFlow.checkIfMeasuredTime()) {
            StopWatch<true> emtCorrTimer;
            StopWatch<true> hdf5Timer;

            emtCorrTimer.start();
            EMTCorr.EMTCorrGFunctionsAveragedTau(gauge, vecEMTCorrAveragedTau);
            emtCorrTimer.stop();
            rootLogger.debug("EMTCorrGFunctions took ", emtCorrTimer.seconds(), "s.");

            EMTCorr.checkEMTCorrGFunctionsAveragedTau(vecEMTCorrAveragedTau, vecCounts);
            
            // write data in hdf5 file anyway (regardless of useHDF5 setting)
            hdf5Timer.start();
            hdf5File.writeEMTCorrAveragedTauData(vecEMTCorrAveragedTau, vecCounts);
            hdf5Timer.stop();
            rootLogger.debug("writeEMTCorrData took  ", hdf5Timer.seconds(), "s.");
        }

        if (lp.energyMomentumTensorCorrFunctionsGeneralTau() && gradFlow.checkIfMeasuredTime()) {
            StopWatch<true> emtCorrTimer;
            StopWatch<true> hdf5Timer;

            emtCorrTimer.start();
            if (lp.energyMomentumTensorCorrFunctionsGeneralTauMatsubara()) {
                EMTCorr.template EMTCorrGFunctionsGeneralTau<true>(gauge, vecEMTCorrGeneralTau);
            } else {
                EMTCorr.template EMTCorrGFunctionsGeneralTau<false>(gauge, vecEMTCorrGeneralTau);
            }
            emtCorrTimer.stop();
            rootLogger.debug("EMTCorrGFunctions took ", emtCorrTimer.seconds(), "s.");
            
            // check for Matsubara modes is not implemented (yet?)
            if (!lp.energyMomentumTensorCorrFunctionsGeneralTauMatsubara()) {
                EMTCorr.template checkEMTCorrGFunctionsGeneralTau<false>(vecEMTCorrGeneralTau, vecCounts);
            }

            // write data in hdf5 file anyway (regardless of useHDF5 setting)
            hdf5Timer.start();
            if (lp.energyMomentumTensorCorrFunctionsGeneralTauMatsubara()) {
                hdf5File.template writeEMTCorrGeneralTauData<true>(vecEMTCorrGeneralTau, vecCounts);
            } else {
                hdf5File.template writeEMTCorrGeneralTauData<false>(vecEMTCorrGeneralTau, vecCounts);
            }
            hdf5Timer.stop();
            rootLogger.debug("writeEMTCorrData took  ", hdf5Timer.seconds(), "s.");
        }

        if (lp.shear_bulk_corr_block() && gradFlow.checkIfMeasuredTime()) {

            EnergyDensity = BlockBulk.updateBlock(EMTEBlock, lp.binsize());
            BulkCorr = BlockBulk.getCorr(EMTEBlock, lp.binsize());

            EMTensorTraceless = BlockShear.updateBlock(EMTUBlock, lp.binsize());
            ShearCorr = BlockShear.getCorr(EMTUBlock, lp.binsize());

            LineFormatter newLine_BlockShear = file_BlockShear.tag("");
            newLine_BlockShear << flow_time << " ";

            LineFormatter newLine_BlockBulk = file_BlockBulk.tag("");
            newLine_BlockBulk << flow_time << " ";

            for (size_t i=0;i<BulkCorr.size();i++) {
                if (fabs(BulkCorr[i])>1e-50) {//skip empty(zero) entries
                    newLine_BlockBulk << sqrt(i%Rsq_size)*lp.binsize() << " " << std::scientific << std::setprecision(15) << BulkCorr[i] <<" ";
                    newLine_BlockShear << sqrt(i%Rsq_size)*lp.binsize() << " " << std::scientific << std::setprecision(15) << ShearCorr[i] <<" ";
                }
            }
            newLine_BlockShear << "\n";
            newLine_BlockBulk << "\n";

            LineFormatter newLine_normEMT = file_normEMT.tag("");
            newLine_normEMT << flow_time << " ";
            newLine_normEMT << std::scientific << std::setprecision(15) << EnergyDensity << " " << EMTensorTraceless.elems[0] << " "
                            << EMTensorTraceless.elems[1] << " " << EMTensorTraceless.elems[2] << " " << EMTensorTraceless.elems[3] << " "
                            << EMTensorTraceless.elems[4] << " " << EMTensorTraceless.elems[5] << " " << EMTensorTraceless.elems[6] << " "
                            << EMTensorTraceless.elems[7] << " " << EMTensorTraceless.elems[8] << " " << EMTensorTraceless.elems[9] <<"\n";
        }

        if (lp.ColorElectricCorrTimeSlices_naive() && gradFlow.checkIfMeasuredTime()) {
            //! print naive discretization for ce
            LineFormatter newLineColEl_naive = fileColElecCorrSl_naive.tag("");
            resultPoly = poly.getPolyakovLoop();
            resultColElecCorSl_naive = CEC.getColorElectricCorr_naive();
            newLineColEl_naive << flow_time;
            newLineColEl_naive << real(resultPoly);
            newLineColEl_naive << imag(resultPoly);
            for (auto &elem : resultColElecCorSl_naive) {
                newLineColEl_naive << real(elem);
            }
            for (auto &elem : resultColElecCorSl_naive) {
                newLineColEl_naive << imag(elem);
            }
        }

        if (lp.ColorElectricCorrTimeSlices_clover() && gradFlow.checkIfMeasuredTime()) {
            //! print clover discretization for ce
            LineFormatter newLineColEl_clover = fileColElecCorrSl_clover.tag("");
            resultPoly = poly.getPolyakovLoop();
            resultColElecCorSl_clover = CEC.getColorElectricCorr_clover();
            newLineColEl_clover << flow_time;
            newLineColEl_clover << real(resultPoly);
            newLineColEl_clover << imag(resultPoly);
            for (auto &elem : resultColElecCorSl_clover) {
                newLineColEl_clover << real(elem);
            }
            for (auto &elem : resultColElecCorSl_clover) {
                newLineColEl_clover << imag(elem);
            }
        }

        if (lp.ColorMagneticCorrTimeSlices_naive() && gradFlow.checkIfMeasuredTime()) {
            //! print naive discretization for cm
            LineFormatter newLineColMa_naive = fileColMagnCorrSl_naive.tag("");
            resultPoly = poly.getPolyakovLoop();
            resultColMagnCorSl_naive = CMC.getColorMagneticCorr_naive();
            newLineColMa_naive << flow_time;
            newLineColMa_naive << real(resultPoly);
            newLineColMa_naive << imag(resultPoly);
            for (auto &elem : resultColMagnCorSl_naive) {
                newLineColMa_naive << real(elem);
            }
            for (auto &elem : resultColMagnCorSl_naive) {
                newLineColMa_naive << imag(elem);
            }
        }

        if (lp.ColorMagneticCorrTimeSlices_clover() && gradFlow.checkIfMeasuredTime()) {
            //! print clover discretization for cm
            LineFormatter newLineColMa_clover = fileColMagnCorrSl_clover.tag("");
            resultPoly = poly.getPolyakovLoop();
            resultColMagnCorSl_clover = CMC.getColorMagneticCorr_clover();
            newLineColMa_clover << flow_time;
            newLineColMa_clover << real(resultPoly);
            newLineColMa_clover << imag(resultPoly);
            for (auto &elem : resultColMagnCorSl_clover) {
                newLineColMa_clover << real(elem);
            }
            for (auto &elem : resultColMagnCorSl_clover) {
                newLineColMa_clover << imag(elem);
            }
        }

        if ((lp.PolyakovLoopCorrelator() && gradFlow.checkIfMeasuredTime())) {
            Gaugefield<floatT, false, HaloDepth> gauge_host(gauge.getComm());
            gauge_host = gauge;
            int ngfstep=0;
            floatT gftheta=1e10;
            while ((ngfstep < lp.GaugeFixNMax()) && (gftheta > lp.GaugeFixTol())) {
                // Compute starting GF functional and update the lattice.
                gFix.gaugefixOR();
                // Due to the nature of the update, we have to re-unitarize every so often.
                if ((ngfstep % lp.GaugeFixNUnitarize()) == 0) {
                    gauge.su3latunitarize();
                }
                // Re-calculate theta to determine whether we are sufficiently fixed.
                gftheta = gFix.getTheta();
                ngfstep += 1;
            }
            gauge.su3latunitarize(); // One final re-unitarization.
            PLC.PLCtoArrays(vec_plca, vec_plc1, vec_plc8, vec_factor, vec_weight, true);
            LineFormatter newLineplca = filePolyCorrAverage.tag("");
            LineFormatter newLineplc1 = filePolyCorrSinglet.tag("");
            LineFormatter newLineplc8 = filePolyCorrOctet.tag("");
            newLineplca << flow_time;
            newLineplc1 << flow_time;
            newLineplc8 << flow_time;
            // Write final results to output file. Not every r^2 is possible on a lattice; this
            // construction ensures only those possible distances are output.
            for (int dx=0 ; dx<corrTools.distmax ; dx++) {
                if (vec_factor[dx]>0) {
                    newLineplca << vec_plca[dx];
                    newLineplc1 << vec_plc1[dx];
                    newLineplc8 << vec_plc8[dx];
                }
            }
            gauge = gauge_host;
        }

        if (lp.RenormPolyakovSusc()) {

            // All susceptibility objects, instantiated here to save memory when RenormPolyakovSusc==False.
            CorrField<false,SU3<floatT>> thermalWilsonLine(gauge.getComm(), corrTools.vol3);
            Correlator<false,COMPLEX(floatT)> ABareSusc(gauge.getComm(), corrTools.USr2max);
            Correlator<false,floatT> LBareSusc(gauge.getComm(), corrTools.USr2max);
            Correlator<false,floatT> TBareSusc(gauge.getComm(), corrTools.USr2max);
            Correlator<false,floatT> CPUnorm(gauge.getComm(), corrTools.USr2max);
            LatticeContainerAccessor _thermalWilsonLine(thermalWilsonLine.getAccessor());
            LatticeContainerAccessor _ABareSusc(ABareSusc.getAccessor());
            LatticeContainerAccessor _LBareSusc(LBareSusc.getAccessor());
            LatticeContainerAccessor _TBareSusc(TBareSusc.getAccessor());
            LatticeContainerAccessor _CPUnorm(CPUnorm.getAccessor());

            // Get thermal Wilson line, the object out of which the susceptibilities is constructed.
            poly.PloopInArray(_thermalWilsonLine);
            resultPoly = poly.getPolyakovLoop();

            // Calculate susceptibilities.
            rootLogger.info("Remove contact term from chi_A, chi_L, chi_T...");
            corrTools.template correlateAt<SU3<floatT>,COMPLEX(floatT),trAxtrBt<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, ABareSusc, false, lp.normFileDir());
            corrTools.template correlateAt<SU3<floatT>,floatT,trReAxtrReB<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, LBareSusc, true, lp.normFileDir());
            corrTools.template correlateAt<SU3<floatT>,floatT,trImAxtrImB<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, TBareSusc, true, lp.normFileDir());

            // Output.
            LineFormatter newLinePolySuscA = fileRenormPolySuscA.tag("");
            LineFormatter newLinePolySuscL = fileRenormPolySuscL.tag("");
            LineFormatter newLinePolySuscT = fileRenormPolySuscT.tag("");
            newLinePolySuscA << flow_time;
            newLinePolySuscL << flow_time;
            newLinePolySuscT << flow_time;
            for(int ir2=0; ir2<corrTools.USr2max+1; ir2++) {
                _CPUnorm.getValue<floatT>(ir2,norm);
                if(norm > 0) {
                    _ABareSusc.getValue<COMPLEX(floatT)>(ir2,suscA);
                    _LBareSusc.getValue<floatT>(ir2,suscL);
                    _TBareSusc.getValue<floatT>(ir2,suscT);
                    newLinePolySuscA << real(suscA);
                    newLinePolySuscL << suscL;
                    newLinePolySuscT << suscT;
                }
            }
            newLinePolySuscA << resultPoly;
            newLinePolySuscL << resultPoly;
            newLinePolySuscT << resultPoly;
        }

        rootLogger.info(logStream.str());

        continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached
        if (continueFlow){
            flow_time += gradFlow.updateFlow(); //! integrate flow equation up to next flow time
            gauge.updateAll();

            gAction.recomputeField();
            topology.recomputeField();
        }

    }
    timer.stop();
    rootLogger.info("complete time = " ,  timer.minutes() ,  " min");
}


int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(INFO);
        CommunicationBase commBase(&argc, &argv);
        gradientFlowParam<PREC> lp;
        lp.readfile(commBase, "../parameter/applications/gradientFlow.param", argc, argv);
        commBase.init(lp.nodeDim());

        /// Convert input strings to enum for switching
        Force input_force = Force_map[lp.force()];
        RungeKuttaMethod input_RK_method = RK_map[lp.RK_method()];

        if (input_RK_method == fixed_stepsize && lp.ignore_fixed_startstepsize() && lp.necessary_flow_times.isSet()) {
            rootLogger.info("Ignoring fixed start_step_size. "
                                 "Stepsizes are dynamically deduced from necessary_flow_times.");
            lp.start_step_size.set(lp.measurement_intervall()[1]);
        }

        size_t input_HaloDepth = 1;
        if (input_force == wilson && (lp.topCharge_imp() || lp.topChargeTimeSlices_imp() || lp.topCharge_imp_imp() || lp.topChargeTimeSlices_imp_imp())) {
            input_HaloDepth = 2;
        } else if (input_force == zeuthen ) {
            input_HaloDepth = 3;
        }

        //! loop over all templates and choose the one specified by the user
        static_for<1, 4>::apply([&](auto i){
            const auto HaloDepth = static_cast<size_t>(i);
            static_for<0, 3>::apply([&](auto j){
                const auto RKmethod = static_cast<RungeKuttaMethod>(static_cast<int>(j));
                static_for<0, 2>::apply([&](auto k){
                    const auto myforce = static_cast<Force>(static_cast<int>(k));
                    if ( myforce == input_force && RKmethod == input_RK_method && HaloDepth == input_HaloDepth ) {
                        rootLogger.info("Initializing gradientFlow with RK_method=", RungeKuttaMethods[j], ", Force=", Forces[k]);
                        run<PREC, USE_GPU, HaloDepth, RKmethod, myforce>(commBase, lp);
                    }
                });
            });
        });
    }
    catch(const std::runtime_error &error) {
        return 1;
    }
    return 0;
}

