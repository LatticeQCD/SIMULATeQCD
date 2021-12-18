/* 
 * main_gradientFlow.cu                                                               
 *
 * Lukas Mazur
 *
 * Application for computing observables on configurations that have been smoothed using the gradient flow. If you have
 * a new observable that you would like to measure on flowed configurations, please add your code here. Try to follow
 * the example of other observables, e.g. by including an optional flag for measuring your observable, outputting using
 * the FileWriter, and so on.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gradientFlow/gradientFlow.h"
#include "../modules/observables/Topology.h"
#include "../modules/observables/Blocking.h"
#include "../modules/observables/EnergyMomentumTensor.h"
#include "../modules/observables/ColorElectricCorr.h"
#include "../modules/observables/ColorMagneticCorr.h"
#include "../modules/gaugeFixing/gfix.h"
#include "../modules/gaugeFixing/PolyakovLoopCorrelator.h"
#include <cstdio>

#define USE_GPU true
//define precision
#if SINGLEPREC
#define PREC float
#else
#define PREC double
#endif


template<class floatT>
struct gradientFlowParam : LatticeParameters {

    //! ---------------------------------basic options you probably care about------------------------------------------
    Parameter<std::string> measurements_dir; //! where the output gets stored
    Parameter<std::string> force; //! wilson or zeuthen flow
    Parameter<std::string> RK_method; //! RK_method = {fixed_stepsize, adaptive_stepsize, adaptive_stepsize_allgpu}
    Parameter<floatT> start_step_size;
    Parameter<floatT> accuracy; //! only used for adaptive stepsize. difference between 2nd and 3rd order RK
    DynamicParameter<floatT> necessary_flow_times; //! these flow times will not be skipped over in the integration
    Parameter<floatT, 2> measurement_intervall; //! measurement_intervall[0]: start, [1]: stop

    //! ---------------------------------which observables should be measured on the flowed configuration?--------------

    Parameter<bool> plaquette;
    Parameter<bool> clover;
    Parameter<bool> cloverTimeSlices;
    Parameter<bool> topCharge;
    Parameter<bool> topChargeTimeSlices;
    Parameter<bool> topCharge_imp;
    Parameter<bool> topChargeTimeSlices_imp;
    Parameter<bool> ColorElectricCorrTimeSlices_naive;
    Parameter<bool> ColorElectricCorrTimeSlices_clover;
    Parameter<bool> ColorMagneticCorrTimeSlices_naive;
    Parameter<bool> ColorMagneticCorrTimeSlices_clover;
    Parameter<bool> RenormPolyakovSusc;

    Parameter<bool> topCharge_imp_block;
    Parameter<bool> shear_bulk_corr_block;
    Parameter<bool> energyMomentumTensorTracelessTimeSlices;
    Parameter<bool> energyMomentumTensorTracefullTimeSlices;
    Parameter<int> binsize; //! the binsize used in the blocking method 

    Parameter<bool> PolyakovLoopCorrelator;
    Parameter<floatT> GaugeFixTol;
    Parameter<int> GaugeFixNMax;
    Parameter<int> GaugeFixNUnitarize;

    Parameter<std::string> normFileDir; //! Normalization file needed for correlator class calcuations.

    //! ---------------------------------advanced options---------------------------------------------------------------

    Parameter<bool> use_unit_conf; //! for testing (or benchmarking purposes using fixed stepsize)
    Parameter<bool> save_conf;
    //! ignore start_step_size and integrate to the necessary_flow_times without steps in between.
    //! only useful when using RK_method fixed_stepsize
    Parameter<bool> ignore_fixed_startstepsize;

    gradientFlowParam() {
        addDefault(force, "force", std::string("zeuthen"));

        add(start_step_size, "start_step_size");

        addDefault(RK_method, "RK_method", std::string("adaptive_stepsize"));
        addDefault(accuracy, "accuracy", floatT(1e-5));

        addDefault(binsize, "binsize", 8);

        add(measurements_dir, "measurements_dir");

        addOptional(necessary_flow_times, "necessary_flow_times");
        addDefault(ignore_fixed_startstepsize, "ignore_start_step_size", false);

        addDefault(save_conf, "save_configurations", false);

        addDefault(use_unit_conf, "use_unit_conf", false);

        add(measurement_intervall, "measurement_intervall");

        addDefault(plaquette, "plaquette", true);
        addDefault(clover, "clover", false);
        addDefault(cloverTimeSlices, "cloverTimeSlices", false);
        addDefault(topCharge, "topCharge", false);
        addDefault(topChargeTimeSlices, "topChargeTimeSlices", false);
        addDefault(topCharge_imp, "topCharge_imp", false);
        addDefault(topChargeTimeSlices_imp, "topChargeTimeSlices_imp", false);
        addDefault(topCharge_imp_block, "topCharge_imp_block", false);
        addDefault(shear_bulk_corr_block, "shear_bulk_corr_block", false);
        addDefault(energyMomentumTensorTracelessTimeSlices, "energyMomentumTensorTracelessTimeSlices", false);
        addDefault(energyMomentumTensorTracefullTimeSlices, "energyMomentumTensorTracefullTimeSlices", false);
        addDefault(ColorElectricCorrTimeSlices_naive, "ColorElectricCorrTimeSlices_naive", false);
        addDefault(ColorElectricCorrTimeSlices_clover, "ColorElectricCorrTimeSlices_clover", false);
        addDefault(ColorMagneticCorrTimeSlices_naive, "ColorMagneticCorrTimeSlices_naive", false);
        addDefault(ColorMagneticCorrTimeSlices_clover, "ColorMagneticCorrTimeSlices_clover", false);
        addDefault(RenormPolyakovSusc, "RenormPolyakovSusc", false);

        addDefault(PolyakovLoopCorrelator, "PolyakovLoopCorrelator", false);
        addDefault(GaugeFixTol, "GaugeFixTol", floatT(1e-6));
        addDefault(GaugeFixNMax, "GaugeFixNMax", 9000);
        addDefault(GaugeFixNUnitarize, "GaugeFixNUnitarize", 20);

        addDefault(normFileDir, "normFileDir", std::string("./"));
    }
};


template<class floatT, size_t HaloDepth, typename gradFlowClass>
void run(gradFlowClass &gradFlow, Gaugefield<floatT, USE_GPU, HaloDepth> &gauge, gradientFlowParam<floatT> &lp) {

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

    std::stringstream prefix, datName, datNameConf, datNameCloverSlices, datNameTopChSlices, datNameTopChSlices_imp,
            datNameBlockShear, datNameBlockBulk, datName_normEMT, datNameColElecCorrSlices_naive, datNameColMagnCorrSlices_naive,
            datNamePolyCorrSinglet, datNamePolyCorrOctet, datNamePolyCorrAverage, datNameColElecCorrSlices_clover,
            datNameColMagnCorrSlices_clover, datNameBlockTopCharge, datNameEMTUTimeSlices, datNameEMTETimeSlices,
            datNameRenormPolySuscA, datNameRenormPolySuscL, datNameRenormPolySuscT;
    // fill stream with 0's
    datName.fill('0');
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
    datName << lp.measurements_dir() << prefix.str() << lp.fileExt();
    datNameConf << lp.measurements_dir()<< "conf_" << prefix.str() << lp.fileExt();
    datNameCloverSlices << lp.measurements_dir() << prefix.str() << "_CloverTimeSlices" << lp.fileExt();
    datNameTopChSlices << lp.measurements_dir() << prefix.str() << "_TopChTimeSlices" << lp.fileExt();
    datNameTopChSlices_imp << lp.measurements_dir() << prefix.str() << "_TopChTimeSlicesImp" << lp.fileExt();
    datNameColElecCorrSlices_naive << lp.measurements_dir() << prefix.str() << "_ColElecCorrTimeSlices_naive" << lp.fileExt();
    datNameColMagnCorrSlices_naive << lp.measurements_dir() << prefix.str() << "_ColMagnCorrTimeSlices_naive" << lp.fileExt();
    datNamePolyCorrSinglet << lp.measurements_dir() << prefix.str() << "_PolyakovCorrSinglet" << lp.fileExt();
    datNamePolyCorrOctet << lp.measurements_dir() << prefix.str() << "_PolyakovCorrOctet" << lp.fileExt();
    datNamePolyCorrAverage << lp.measurements_dir() << prefix.str() << "_PolyakovCorrAverage" << lp.fileExt();
    datNameRenormPolySuscA << lp.measurements_dir() << prefix.str() << "_RenormPolySuscA" << lp.fileExt();
    datNameRenormPolySuscL << lp.measurements_dir() << prefix.str() << "_RenormPolySuscL" << lp.fileExt();
    datNameRenormPolySuscT << lp.measurements_dir() << prefix.str() << "_RenormPolySuscT" << lp.fileExt();
    datNameBlockTopCharge << lp.measurements_dir() << prefix.str() << "_BlockTopCharge" << lp.fileExt();
    datNameBlockShear << lp.measurements_dir() << prefix.str() << "_BlockShear" << lp.fileExt();
    datNameBlockBulk << lp.measurements_dir() << prefix.str() << "_BlockBulk" << lp.fileExt();
    datName_normEMT << lp.measurements_dir() << prefix.str() << "_NormEMT" << lp.fileExt();
    datNameColElecCorrSlices_clover << lp.measurements_dir() << prefix.str() << "_ColElecCorrTimeSlices_clover" << lp.fileExt();
    datNameColMagnCorrSlices_clover << lp.measurements_dir() << prefix.str() << "_ColMagnCorrTimeSlices_clover" << lp.fileExt();
    datNameEMTUTimeSlices << lp.measurements_dir() << prefix.str() << "_EMTUTimeSlices" << lp.fileExt();
    datNameEMTETimeSlices << lp.measurements_dir() << prefix.str() << "_EMTETimeSlices" << lp.fileExt();
    FileWriter file(gauge.getComm(), lp, datName.str());

    //! write header
    LineFormatter header = file.header();
    header << "Flow time ";
    if (lp.plaquette()) header << "Plaquette ";
    if (lp.clover()) header << "Clover ";
    if (lp.topCharge()) header << "Top. Charge ";
    if (lp.topCharge_imp() || lp.topCharge_imp_block()) header << "Impr. top. Charge ";
    header.endLine();

    FileWriter file_BlockTopCharge(gauge.getComm(), lp);
    FileWriter file_normEMT(gauge.getComm(), lp);
    FileWriter file_BlockShear(gauge.getComm(), lp);
    FileWriter file_BlockBulk(gauge.getComm(), lp);
    if (lp.shear_bulk_corr_block()) {
        file_normEMT.createFile(datName_normEMT.str());
        file_BlockShear.createFile(datNameBlockShear.str());
        file_BlockBulk.createFile(datNameBlockBulk.str());
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

    FileWriter file_EMTUTimeSlices(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTracelessTimeSlices()) {
        file_EMTUTimeSlices.createFile(datNameEMTUTimeSlices.str());
        LineFormatter header_EMTUTimeSlices = file_EMTUTimeSlices.header();
        header_EMTUTimeSlices << "#flowtime U00 U11 U22 U33 U01 U02 U03 U12 U13 U23 for tau=0, ... for tau=1 ..." << "\n";
        header_EMTUTimeSlices.endLine();
    }

    FileWriter file_EMTETimeSlices(gauge.getComm(), lp);
    if (lp.energyMomentumTensorTracefullTimeSlices()) {
        file_EMTETimeSlices.createFile(datNameEMTETimeSlices.str());
        LineFormatter header_EMTETimeSlices = file_EMTETimeSlices.header();
        header_EMTETimeSlices << "#flowtime E for tau=0, ... for tau=1 ..." << "\n";
        header_EMTETimeSlices.endLine();
    }

    if (lp.topCharge_imp_block()) {
        file_BlockTopCharge.createFile(datNameBlockTopCharge.str());
        LineFormatter header_BlockTopCharge = file_BlockTopCharge.header();
        header_BlockTopCharge<< "#flow time tau/a=0: #r/a1 #corr1 #r/a2 #corr2.... tau/a=1: #r/a1 #corr1...." << "\n";
        header_BlockTopCharge.endLine();
    }

    FileWriter filePolyCorrSinglet(gauge.getComm(), lp);
    FileWriter filePolyCorrOctet(gauge.getComm(), lp);
    FileWriter filePolyCorrAverage(gauge.getComm(), lp);
    if (lp.PolyakovLoopCorrelator()) {
        filePolyCorrSinglet.createFile(datNamePolyCorrSinglet.str());
        filePolyCorrOctet.createFile(datNamePolyCorrOctet.str());
        filePolyCorrAverage.createFile(datNamePolyCorrAverage.str());
    }

    FileWriter fileRenormPolySuscA(gauge.getComm(), lp);
    FileWriter fileRenormPolySuscL(gauge.getComm(), lp);
    FileWriter fileRenormPolySuscT(gauge.getComm(), lp);
    if (lp.RenormPolyakovSusc()) {
        fileRenormPolySuscA.createFile(datNameRenormPolySuscA.str());
        fileRenormPolySuscL.createFile(datNameRenormPolySuscL.str());
        fileRenormPolySuscT.createFile(datNameRenormPolySuscT.str());
    }

    FileWriter fileCloverSl(gauge.getComm(), lp);
    if (lp.cloverTimeSlices()) {
        fileCloverSl.createFile(datNameCloverSlices.str());
        LineFormatter headerClSl = fileCloverSl.header();
        headerClSl << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerClSl << "Nt=" + std::to_string(nt) + " ";
        }
        headerClSl.endLine();
    }

    FileWriter fileTopChSl(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices()) {
        fileTopChSl.createFile(datNameTopChSlices.str());
        LineFormatter headerThSl = fileTopChSl.header();
        headerThSl << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl.endLine();
    }

    FileWriter fileTopChSl_imp(gauge.getComm(), lp);
    if (lp.topChargeTimeSlices_imp()) {
        fileTopChSl_imp.createFile(datNameTopChSlices_imp.str());
        LineFormatter headerThSl_imp = fileTopChSl_imp.header();
        headerThSl_imp << "Flow time ";
        for (int nt = 0; nt < lp.latDim[3]; nt++) {
            headerThSl_imp << "Nt=" + std::to_string(nt) + " ";
        }
        headerThSl_imp.endLine();
    }

    FileWriter fileColElecCorrSl_naive(gauge.getComm(), lp);
    if (lp.ColorElectricCorrTimeSlices_naive()) {
        fileColElecCorrSl_naive.createFile(datNameColElecCorrSlices_naive.str());
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
        fileColMagnCorrSl_naive.createFile(datNameColMagnCorrSlices_naive.str());
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
        fileColElecCorrSl_clover.createFile(datNameColElecCorrSlices_clover.str());
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
        fileColMagnCorrSl_clover.createFile(datNameColMagnCorrSlices_clover.str());
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

    //! -------------------------------read in configuration------------------------------------------------------------

    if (lp.use_unit_conf()){
        rootLogger.info("Using unit configuration for tests/benchmarks");
        gauge.one();
    } else {
        rootLogger.info("Read configuration");
        gauge.readconf_nersc(lp.GaugefileName());
    }
    gauge.updateAll();

    //! -------------------------------set up observable measurement classes--------------------------------------------

    GaugeAction<floatT, USE_GPU, HaloDepth> gAction(gauge);
    Topology<floatT, USE_GPU, HaloDepth> topology(gauge);
    EnergyMomentumTensor<floatT, USE_GPU, HaloDepth> EMT(gauge);

    BlockingMethod<floatT, true, HaloDepth, floatT, topChargeDens_imp<floatT, HaloDepth, true>, CorrType<floatT>> BlockTopChDens(gauge, lp.binsize());
    BlockingMethod<floatT, true, HaloDepth, floatT, EMTtrace<floatT, true, HaloDepth>, CorrType<floatT>> BlockBulk(gauge, lp.binsize());
    BlockingMethod<floatT, true, HaloDepth, Matrix4x4Sym<floatT>, EMTtraceless<floatT, true, HaloDepth>, CorrType<floatT>> BlockShear(gauge, lp.binsize());

    ColorElectricCorr<floatT, USE_GPU, HaloDepth> CEC(gauge);
    ColorMagneticCorr<floatT, USE_GPU, HaloDepth> CMC(gauge);
    PolyakovLoop<floatT, USE_GPU, HaloDepth> poly(gauge);
    GaugeFixing<floatT,true,HaloDepth> gFix(gauge);
    PolyakovLoopCorrelator<floatT,true,HaloDepth> PLC(gauge);
    CorrelatorTools<floatT,true,HaloDepth> corrTools;

    //! -------------------------------variables for the observables----------------------------------------------------

    floatT plaq, clov, topChar;
    std::vector<floatT> resultClSl, resultThSl, resultThSl_imp, resultEMTETimeSlices;
    std::vector<Matrix4x4Sym<floatT>> resultEMTUTimeSlices;
    std::vector<GCOMPLEX(floatT)> resultColElecCorSl_naive, resultColMagnCorSl_naive, resultColElecCorSl_clover,
                                  resultColMagnCorSl_clover;

    GCOMPLEX(floatT) resultPoly;

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
    GCOMPLEX(floatT) suscA;

    //! -------------------------------flow the field until max flow time-----------------------------------------------

    std::stringstream logStream;
    StopWatch timer;
    timer.start();
    floatT flow_time = 0.;
    bool continueFlow = true;

    while (continueFlow) {

        continueFlow = gradFlow.continueFlow(); //! check if the max flow time has been reached

        //! -------------------------------prepare log output-----------------------------------------------------------

        logStream.str("");
        logStream << std::fixed << std::setprecision(7) << "   t = " << flow_time << ": ";
        LineFormatter newLine = file.tag("");
        newLine << flow_time;

        //! -------------------------------calculate observables on flowed field----------------------------------------

        if (lp.save_conf() && gradFlow.checkIfnecessaryTime()){
            gauge.writeconf_nersc( datNameConf.str() + "_FT" + std::to_string(flow_time));
        }

        if (lp.plaquette()) {
            plaq = gAction.plaquette();
            logStream << std::fixed << std::setprecision(6) << "   Plaquette = " << plaq;
            newLine << plaq;
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
            newLine << clov;
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
            logStream << std::fixed << std::setprecision(6) << "   topCharge = " << topChar;
            newLine << topChar;
            topology.recomputeField();
        }

        if (lp.topChargeTimeSlices_imp()) {
            LineFormatter newLineTh = fileTopChSl_imp.tag("");
            topology.template topChargeTimeSlices<true>(resultThSl_imp);
            newLineTh << flow_time;
            for (auto &elem : resultThSl_imp) {
                newLineTh << elem;
            }
            topology.dontRecomputeField();
        }

        if (lp.topCharge_imp() && !lp.topCharge_imp_block()) {
            topChar = topology.template topCharge<true>();
            logStream << std::fixed << std::setprecision(6) << "   topCharge_imp = " << topChar;
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

        if (lp.energyMomentumTensorTracelessTimeSlices() && gradFlow.checkIfnecessaryTime()) {
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

        if (lp.energyMomentumTensorTracefullTimeSlices() && gradFlow.checkIfnecessaryTime()) {
            LineFormatter newLineEMTETimeSlices = file_EMTETimeSlices.tag("");
            EMT.EMTETimeSlices(resultEMTETimeSlices);
            newLineEMTETimeSlices << flow_time << " ";
            for (auto &elem : resultEMTETimeSlices) {
                newLineEMTETimeSlices << std::scientific << std::setprecision(15) << elem << " ";
            }
        }

        if (lp.shear_bulk_corr_block() && gradFlow.checkIfnecessaryTime()) {

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

        if (lp.ColorElectricCorrTimeSlices_naive() && gradFlow.checkIfnecessaryTime()) {
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

        if (lp.ColorElectricCorrTimeSlices_clover() && gradFlow.checkIfnecessaryTime()) {
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
   
        if (lp.ColorMagneticCorrTimeSlices_naive() && gradFlow.checkIfnecessaryTime()) {
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

        if (lp.ColorMagneticCorrTimeSlices_clover() && gradFlow.checkIfnecessaryTime()) {
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

        if ((lp.PolyakovLoopCorrelator() && gradFlow.checkIfnecessaryTime())) {
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
            CorrField<false,GSU3<floatT>> thermalWilsonLine(gauge.getComm(), corrTools.vol3);
            Correlator<false,GCOMPLEX(floatT)> ABareSusc(gauge.getComm(), corrTools.USr2max);
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
            corrTools.template correlateAt<GSU3<floatT>,GCOMPLEX(floatT),trAxtrBt<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, ABareSusc, false, lp.normFileDir());
            corrTools.template correlateAt<GSU3<floatT>,floatT,trReAxtrReB<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, LBareSusc, true, lp.normFileDir());
            corrTools.template correlateAt<GSU3<floatT>,floatT,trImAxtrImB<floatT>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, TBareSusc, true, lp.normFileDir());

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
                    _ABareSusc.getValue<GCOMPLEX(floatT)>(ir2,suscA);
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
        flow_time += gradFlow.updateFlow(); //! integrate flow equation up to next flow time
        gauge.updateAll();

        gAction.recomputeField();
        topology.recomputeField();
    }
    timer.stop();
    rootLogger.info("complete time = " ,  timer.ms()/60. ,  " min");
}


template<class floatT, bool onDevice, const size_t HaloDepth, RungeKuttaMethod input_RK_method, template<class, const size_t, RungeKuttaMethod> class gradFlowClass >
void init_flow(CommunicationBase &commBase, gradientFlowParam<floatT> &lp) {

    initIndexer(HaloDepth, lp, commBase);
    Gaugefield<floatT, onDevice, HaloDepth> gauge(commBase);
    gradFlowClass<floatT, HaloDepth, input_RK_method> gradFlow(gauge, lp.start_step_size(),
                                                                 lp.measurement_intervall()[0],
                                                                 lp.measurement_intervall()[1],
                                                                 lp.necessary_flow_times.get(), lp.accuracy());
    run<floatT, HaloDepth, gradFlowClass<floatT, HaloDepth, input_RK_method>>(gradFlow, gauge, lp);
}

int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(DEBUG);
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

        /// Set HaloDepth. The ifdefs can reduce compile time (only define what you need in CMakeLists).
        /// Wilson flow with topological charge (correlator) needs HaloDepth=2, without 1.
        /// Zeuthen flow always needs 3.
        switch (input_force) {
#ifdef WILSON_FLOW
            case wilson: {
                if (lp.topCharge_imp() || lp.topChargeTimeSlices_imp()) {
                    const size_t HaloDepth = 2;
                    switch (input_RK_method) {
#ifdef FIXED_STEPSIZE
                        case fixed_stepsize:
                            init_flow<PREC, USE_GPU, HaloDepth, fixed_stepsize, wilsonFlow>(commBase, lp);
                            break;
#endif
#ifdef ADAPTIVE_STEPSIZE
                        case adaptive_stepsize:
                            init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize, wilsonFlow>(commBase, lp);
                            break;
                        case adaptive_stepsize_allgpu:
                            init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize_allgpu, wilsonFlow>(commBase, lp);
                            break;
#endif
                        default:
                            throw std::runtime_error(stdLogger.fatal("Invalid RK_method. Did you set the compile definitions accordingly?"));
                    }
                } else {
                    const size_t HaloDepth = 1;
                    switch (input_RK_method) {
#ifdef FIXED_STEPSIZE
                        case fixed_stepsize:
                            init_flow<PREC, USE_GPU, HaloDepth, fixed_stepsize, wilsonFlow>(commBase, lp);
                            break;
#endif
#ifdef ADAPTIVE_STEPSIZE
                        case adaptive_stepsize:
                            init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize, wilsonFlow>(commBase, lp);
                            break;
                        case adaptive_stepsize_allgpu:
                            init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize_allgpu, wilsonFlow>(commBase, lp);
                            break;
#endif
                        default:
                            throw std::runtime_error(stdLogger.fatal("Invalid RK_method. Did you set the compile definitions accordingly?"));
                    }
                }
                break;
            }
#endif
#ifdef ZEUTHEN_FLOW
            case zeuthen: {
                const size_t HaloDepth = 3;
                switch (input_RK_method) {
#ifdef FIXED_STEPSIZE
                    case fixed_stepsize:
                        init_flow<PREC, USE_GPU, HaloDepth, fixed_stepsize, zeuthenFlow>(commBase, lp);
                        break;
#endif
#ifdef ADAPTIVE_STEPSIZE
                    case adaptive_stepsize:
                        init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize, zeuthenFlow>(commBase, lp);
                        break;
                    case adaptive_stepsize_allgpu:
                        init_flow<PREC, USE_GPU, HaloDepth, adaptive_stepsize_allgpu, zeuthenFlow>(commBase, lp);
                        break;
#endif
                    default:
                        throw std::runtime_error(stdLogger.fatal("Invalid RK_method. Did you set the compile definitions accordingly?"));
                }
                break;
            }
#endif
            default:
                throw std::runtime_error(stdLogger.fatal("Invalid force. Did you set the compile definitions accordingly?"));
        }
    }
    catch (const std::runtime_error &error) {
        return -1;
    }
    return 0;
}

