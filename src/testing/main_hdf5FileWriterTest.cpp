/*
 * main_hdf5FileWriterTest.cpp
 *
 * Jonas Winter
 *
 * Test the HDF5FileWriter: write synthetic data for every public writer method, read the file back with H5Cpp and
 * compare shapes, values and attributes.
 * Also check that the writer releases all HDF5 objects when it goes out of scope.
 *
 * Covered functions of HDF5FileWriter:
 *     - constructor and destructor                  (no file, dataset, group or attribute left open afterwards)
 *     - writeAttributes(lp)                         (gauge file, force, RK method, step size, accuracy)
 *     - writeR2CountsAndValues(counts)              (r2_values, r2_counts)
 *     - writeObservable<HDF5_Observable::X>         for X = FlowTime, FlowTimeMeasured, Plaquette, Clover, TopCharge,
 *                                                           TopChargeImp, EMTE
 *     - writeEMTU
 *     - writeTopChargeGeneralTauData                (G_Qtop_general_tau, r^2-filtered)
 *     - writeEMTCorrAveragedTauData                 (G_EMT_averaged_tau, r^2-filtered, component names)
 *     - writeEMTCorrGeneralTauData                  (G_EMT_general_tau, r^2-filtered, component names,
 *                                                    Matsubara modes attribute)
 *
 * Not covered: flush(), and writing from multiple ranks (only root writes).
 * Single rank only.
 *
 */

#include "../simulateqcd.h"
#include "../base/IO/hdf5fileWriter.h"
#include <array>
#include <complex>
#include <cstdio>

#define PREC double
#define HaloDepth 0

typedef std::complex<double> cplx;
typedef TensorDecomposition<PREC, HaloDepth> TD;

constexpr int numberOfFlowTimes = 3;
const std::string fileName = "hdf5FileWriterTest.h5";
const std::string groupFlow = "/gradient_flow_measurements";
const std::string groupCorr = "/gradient_flow_measurements/correlators";


// ------------------------------------------------------------------------------- synthetic data

// unique value for every (quantity, flow time, component, tau, r^2), so that misplaced entries are detected
cplx syntheticValue(int quantity, int step, int component, int tau, int r2) {
    const double re = 1e4 * quantity + 1e3 * step + 20.0 * component + tau + 0.01 * r2;
    return cplx(re, -re - 0.5);
}

enum Quantity {
    qFlowTime, qFlowTimeMeasured, qPlaquette, qClover, qTopCharge, qTopChargeImp, qEMTE, qEMTU, qTopChargeCorr, qEMTCorrAveraged, qEMTCorrGeneral
};

// the general-tau channels that are odd in tau: in Matsubara mode their imaginary part is stored
bool isTauOdd(Projector projector) {
    return projector == Projector::MT || projector == Projector::sm || projector == Projector::lm || projector == Projector::wm;
}

// Matsubara flag passed to writeEMTCorrGeneralTauData at a given flow-time step: alternate to test both rules in one file
bool matsubaraAtStep(int step) {
    return step % 2 == 1;
}

std::string projectorName(Projector projector) {
    static const std::array<std::string, 14> names{{
        "TT", "RT", "MT", "UT", "ss", "ll", "ww", "mm", "sl", "sw", "sm", "lw", "lm", "wm"
    }};
    return names[(int) projector];
}


// ------------------------------------------------------------------------------- reading back

struct Dataset {
    std::vector<hsize_t> dims;
    std::vector<double> values;
};

Dataset readDataset(H5File &file, const std::string &path) {
    DataSet dataSet = file.openDataSet(path);
    DataSpace space = dataSet.getSpace();
    Dataset result;
    result.dims.resize(space.getSimpleExtentNdims());
    space.getSimpleExtentDims(result.dims.data(), NULL);
    hsize_t size = 1;
    for (hsize_t d : result.dims) size *= d;
    result.values.resize(size);
    if (size > 0) dataSet.read(result.values.data(), PredType::NATIVE_DOUBLE);
    return result;
}

std::vector<int> readIntDataset(H5File &file, const std::string &path) {
    DataSet dataSet = file.openDataSet(path);
    std::vector<int> result(dataSet.getSpace().getSimpleExtentNpoints());
    if (!result.empty()) dataSet.read(result.data(), PredType::NATIVE_INT);
    return result;
}

std::vector<std::string> readStringArrayAttribute(H5Object &object, const std::string &name) {
    Attribute attribute = object.openAttribute(name);
    StrType strType(PredType::C_S1, H5T_VARIABLE);
    std::vector<char *> buffer(attribute.getSpace().getSimpleExtentNpoints());
    attribute.read(strType, buffer.data());
    std::vector<std::string> result;
    for (char *s : buffer) {
        result.push_back(std::string(s));
        H5free_memory(s);
    }
    return result;
}

std::string readFixedStringAttribute(H5Object &object, const std::string &name) {
    Attribute attribute = object.openAttribute(name);
    H5std_string result;
    attribute.read(StrType(PredType::C_S1, 256), result);
    return std::string(result.c_str());
}

template<typename T>
std::vector<T> readNumericAttribute(H5Object &object, const std::string &name, const PredType &type) {
    Attribute attribute = object.openAttribute(name);
    std::vector<T> result(attribute.getSpace().getSimpleExtentNpoints());
    attribute.read(type, result.data());
    return result;
}


// ------------------------------------------------------------------------------- bookkeeping

struct TestContext {
    bool lerror = false;

    void check(bool condition, const std::string &text) {
        if (condition) {
            rootLogger.info(CoutColors::green, "TEST PASSED: ", CoutColors::reset, text);
        } else {
            rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, text);
            lerror = true;
        }
    }

    void checkDims(const Dataset &dataset, const std::vector<hsize_t> &expected, const std::string &name) {
        std::stringstream got, want;
        for (hsize_t d : dataset.dims) got << d << " ";
        for (hsize_t d : expected) want << d << " ";
        check(dataset.dims == expected, name + " has shape ( " + want.str() + ") [got ( " + got.str() + ")]");
    }

    void checkValues(double maxDeviation, const std::string &name) {
        std::stringstream text;
        text << name << " holds the written values (max. dev. " << maxDeviation << ")";
        check(maxDeviation == 0.0, text.str());
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    gradientFlowParam<PREC> lp;
    lp.readfile(commBase, "../parameter/tests/hdf5FileWriterTest.param", argc, argv);
    commBase.init(lp.nodeDim());
    initIndexer(HaloDepth, lp, commBase);

    if (commBase.getNumberProcesses() != 1) {
        rootLogger.error("hdf5FileWriterTest only supports a single rank.");
        return -1;
    }

    TestContext ctx;

    const int r2max = TD::getR2max();
    const int tauMax = TD::getTauMax();
    std::vector<int> counts(r2max + 1);
    TD::getR2Counts(counts);
    std::vector<int> hitR2Values;
    for (int r2 = 0; r2 <= r2max; r2++) {
        if (counts[r2] != 0) hitR2Values.push_back(r2);
    }
    const hsize_t hitR2 = hitR2Values.size();
    const hsize_t N = numberOfFlowTimes;

    // ---------------------------------------------------------------------------- write

    {
        HDF5FileWriter<PREC> writer(commBase, lp, fileName);
        writer.writeAttributes(lp);
        writer.writeR2CountsAndValues(counts);

        for (int step = 0; step < numberOfFlowTimes; step++) {
            writer.writeObservable<HDF5_Observable::FlowTime>(syntheticValue(qFlowTime, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::FlowTimeMeasured>(syntheticValue(qFlowTimeMeasured, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::Plaquette>(syntheticValue(qPlaquette, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::Clover>(syntheticValue(qClover, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::TopCharge>(syntheticValue(qTopCharge, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::TopChargeImp>(syntheticValue(qTopChargeImp, step, 0, 0, 0).real());
            writer.writeObservable<HDF5_Observable::EMTE>(syntheticValue(qEMTE, step, 0, 0, 0).real());

            std::vector<PREC> emtu(10);
            for (int i = 0; i < 10; i++) emtu[i] = syntheticValue(qEMTU, step, i, 0, 0).real();
            writer.writeEMTU(emtu);

            std::vector<std::vector<COMPLEX(PREC)>> topChargeCorr(tauMax + 1, std::vector<COMPLEX(PREC)>(r2max + 1));
            for (int tau = 0; tau <= tauMax; tau++)
            for (int r2 = 0; r2 <= r2max; r2++) {
                cplx v = syntheticValue(qTopChargeCorr, step, 0, tau, r2);
                topChargeCorr[tau][r2] = COMPLEX(PREC)(v.real(), v.imag());
            }
            writer.writeTopChargeGeneralTauData(topChargeCorr, counts, matsubaraAtStep(step));

            std::vector<std::vector<COMPLEX(PREC)>> emtCorrAveraged(10, std::vector<COMPLEX(PREC)>(r2max + 1));
            for (int i = 0; i < 10; i++)
            for (int r2 = 0; r2 <= r2max; r2++) {
                cplx v = syntheticValue(qEMTCorrAveraged, step, i, 0, r2);
                emtCorrAveraged[i][r2] = COMPLEX(PREC)(v.real(), v.imag());
            }
            writer.writeEMTCorrAveragedTauData(emtCorrAveraged, counts);

            std::vector<std::vector<std::vector<COMPLEX(PREC)>>> emtCorrGeneral(14, std::vector<std::vector<COMPLEX(PREC)>>(tauMax + 1, std::vector<COMPLEX(PREC)>(r2max + 1)));
            for (int i = 0; i < 14; i++)
            for (int tau = 0; tau <= tauMax; tau++)
            for (int r2 = 0; r2 <= r2max; r2++) {
                cplx v = syntheticValue(qEMTCorrGeneral, step, i, tau, r2);
                emtCorrGeneral[i][tau][r2] = COMPLEX(PREC)(v.real(), v.imag());
            }
            writer.writeEMTCorrGeneralTauData(emtCorrGeneral, counts, matsubaraAtStep(step));
        }
    }

    // ---------------------------------------------------------------------------- lifetime (finding B13)

    // datatypes are left out: H5Cpp keeps transient copies of all predefined types open for the whole program
    const ssize_t openObjects = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_FILE | H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_ATTR);
    ctx.check(openObjects == 0, "no HDF5 object is open after the writer went out of scope [open objects: " + std::to_string(openObjects) + "]");

    // ---------------------------------------------------------------------------- read back

    Exception::dontPrint();
    try {
        H5File file(fileName, H5F_ACC_RDONLY);

        // scalar flow-time datasets
        const std::array<std::pair<std::string, Quantity>, 7> scalarDatasets{{
            {groupFlow + "/flow_time", qFlowTime},
            {groupCorr + "/flow_time", qFlowTimeMeasured},
            {groupFlow + "/plaquette", qPlaquette},
            {groupFlow + "/clover", qClover},
            {groupFlow + "/topological_charge", qTopCharge},
            {groupFlow + "/topological_charge_improved", qTopChargeImp},
            {groupFlow + "/EMTE", qEMTE}
        }};
        for (const auto &entry : scalarDatasets) {
            Dataset dataset = readDataset(file, entry.first);
            ctx.checkDims(dataset, {N}, entry.first);
            if (dataset.dims != std::vector<hsize_t>{N}) continue;
            double deviation = 0.0;
            for (int step = 0; step < numberOfFlowTimes; step++) {
                deviation = std::max(deviation, std::abs(dataset.values[step] - syntheticValue(entry.second, step, 0, 0, 0).real()));
            }
            ctx.checkValues(deviation, entry.first);
        }

        // EMTU: (10, N)
        {
            Dataset dataset = readDataset(file, groupFlow + "/EMTU");
            ctx.checkDims(dataset, {10, N}, "EMTU");
            if (dataset.dims == std::vector<hsize_t>{10, N}) {
                double deviation = 0.0;
                for (int i = 0; i < 10; i++)
                for (int step = 0; step < numberOfFlowTimes; step++) {
                    deviation = std::max(deviation, std::abs(dataset.values[i * N + step] - syntheticValue(qEMTU, step, i, 0, 0).real()));
                }
                ctx.checkValues(deviation, "EMTU");
            }
        }

        // r^2 axis: exactly the non-empty shells
        {
            std::vector<int> r2Values = readIntDataset(file, groupCorr + "/r2_values");
            std::vector<int> r2Counts = readIntDataset(file, groupCorr + "/r2_counts");
            std::vector<int> expectedCounts;
            for (int r2 : hitR2Values) expectedCounts.push_back(counts[r2]);
            ctx.check(r2Values == hitR2Values, "r2_values lists exactly the non-empty r^2 shells");
            ctx.check(r2Counts == expectedCounts, "r2_counts holds the site counts of the non-empty r^2 shells");
        }

        // topological-charge correlator: (N, tauMax + 1, hitR2), always the real part
        {
            Dataset dataset = readDataset(file, groupCorr + "/G_Qtop_general_tau");
            const std::vector<hsize_t> expectedDims = {N, (hsize_t) tauMax + 1, hitR2};
            ctx.checkDims(dataset, expectedDims, "G_Qtop_general_tau");
            if (dataset.dims == expectedDims) {
                double deviation = 0.0;
                for (int step = 0; step < numberOfFlowTimes; step++)
                for (int tau = 0; tau <= tauMax; tau++)
                for (hsize_t h = 0; h < hitR2; h++) {
                    const double value = dataset.values[(step * (tauMax + 1) + tau) * hitR2 + h];
                    deviation = std::max(deviation, std::abs(value - syntheticValue(qTopChargeCorr, step, 0, tau, hitR2Values[h]).real()));
                }
                ctx.checkValues(deviation, "G_Qtop_general_tau (real part, r^2-filtered)");
            }
        }

        // EMT correlator, averaged tau: (10, N, hitR2), real part
        {
            Dataset dataset = readDataset(file, groupCorr + "/G_EMT_averaged_tau");
            const std::vector<hsize_t> expectedDims = {10, N, hitR2};
            ctx.checkDims(dataset, expectedDims, "G_EMT_averaged_tau");
            if (dataset.dims == expectedDims) {
                double deviation = 0.0;
                for (int i = 0; i < 10; i++)
                for (int step = 0; step < numberOfFlowTimes; step++)
                for (hsize_t h = 0; h < hitR2; h++) {
                    const double value = dataset.values[(i * N + step) * hitR2 + h];
                    deviation = std::max(deviation, std::abs(value - syntheticValue(qEMTCorrAveraged, step, i, 0, hitR2Values[h]).real()));
                }
                ctx.checkValues(deviation, "G_EMT_averaged_tau (real part, r^2-filtered)");
            }
        }

        // EMT correlator, general tau: (14, N, tauMax + 1, hitR2); in Matsubara mode the tau-odd channels store the imaginary part
        {
            Dataset dataset = readDataset(file, groupCorr + "/G_EMT_general_tau");
            const std::vector<hsize_t> expectedDims = {14, N, (hsize_t) tauMax + 1, hitR2};
            ctx.checkDims(dataset, expectedDims, "G_EMT_general_tau");
            if (dataset.dims == expectedDims) {
                double deviation = 0.0;
                for (int i = 0; i < 14; i++)
                for (int step = 0; step < numberOfFlowTimes; step++)
                for (int tau = 0; tau <= tauMax; tau++)
                for (hsize_t h = 0; h < hitR2; h++) {
                    const double value = dataset.values[((i * N + step) * (tauMax + 1) + tau) * hitR2 + h];
                    const cplx v = syntheticValue(qEMTCorrGeneral, step, i, tau, hitR2Values[h]);
                    const bool storeImag = matsubaraAtStep(step) && isTauOdd(allProjectorsGeneralTau[i]);
                    deviation = std::max(deviation, std::abs(value - (storeImag ? v.imag() : v.real())));
                }
                ctx.checkValues(deviation, "G_EMT_general_tau (Re, or Im for tau-odd channels in Matsubara mode; r^2-filtered)");
            }
        }

        // attributes of the file
        ctx.check(readNumericAttribute<int>(file, "N", PredType::NATIVE_INT) == std::vector<int>(lp.latDim(), lp.latDim() + 4), "attribute N = lattice extents");
        ctx.check(readNumericAttribute<int>(file, "nodes", PredType::NATIVE_INT) == std::vector<int>(lp.nodeDim(), lp.nodeDim() + 4), "attribute nodes = node layout");
        ctx.check(readNumericAttribute<int>(file, "GPU topology", PredType::NATIVE_INT) == std::vector<int>(lp.gpuTopo(), lp.gpuTopo() + 4), "attribute GPU topology");
        ctx.check(readNumericAttribute<int>(file, "configuration Number", PredType::NATIVE_INT) == std::vector<int>{lp.confnumber()}, "attribute configuration Number");
        ctx.check(readNumericAttribute<double>(file, "beta", PredType::NATIVE_DOUBLE) == std::vector<double>{lp.beta()}, "attribute beta");
        ctx.check(readFixedStringAttribute(file, "gauge file") == lp.GaugefileName(), "attribute gauge file");

        // attributes of the gradient-flow group
        Group flowGroup = file.openGroup(groupFlow);
        ctx.check(readFixedStringAttribute(flowGroup, "gradient flow force") == lp.force(), "attribute gradient flow force");
        ctx.check(readFixedStringAttribute(flowGroup, "Runge-Kutta fixed/adaptive step size") == lp.RK_method(), "attribute Runge-Kutta fixed/adaptive step size");
        ctx.check(readNumericAttribute<double>(flowGroup, "start step size", PredType::NATIVE_DOUBLE) == std::vector<double>{lp.start_step_size()}, "attribute start step size");
        ctx.check(readNumericAttribute<double>(flowGroup, "adaptive step size accuracy", PredType::NATIVE_DOUBLE) == std::vector<double>{lp.accuracy()}, "attribute adaptive step size accuracy");

        // attributes of the EMTU dataset: index pairs in Matrix4x4SymComplex order
        DataSet emtu = file.openDataSet(groupFlow + "/EMTU");
        ctx.check(readStringArrayAttribute(emtu, "index pairs") == std::vector<std::string>{"00", "11", "22", "33", "01", "02", "03", "12", "13", "23"},
            "attribute EMTU/index pairs");
        ctx.check(readStringArrayAttribute(emtu, "index names") == std::vector<std::string>{"xx", "yy", "zz", "tt", "xy", "xz", "xt", "yz", "yt", "zt"},
            "attribute EMTU/index names");

        // component names follow the projector order used by the tensor decomposition.
        // In the averaged-tau list, mm is stored under the name "UR" (see tensorDecomposition.h, Projector).
        std::vector<std::string> expectedAveraged, expectedGeneral;
        for (Projector projector : allProjectorsAveragedTau) expectedAveraged.push_back(projector == Projector::mm ? "UR" : projectorName(projector));
        for (Projector projector : allProjectorsGeneralTau) expectedGeneral.push_back(projectorName(projector));

        DataSet averaged = file.openDataSet(groupCorr + "/G_EMT_averaged_tau");
        DataSet general = file.openDataSet(groupCorr + "/G_EMT_general_tau");
        ctx.check(readStringArrayAttribute(averaged, "component names") == expectedAveraged, "G_EMT_averaged_tau/component names match allProjectorsAveragedTau");
        ctx.check(readStringArrayAttribute(general, "component names") == expectedGeneral, "G_EMT_general_tau/component names match allProjectorsGeneralTau");

        hbool_t matsubara = 0;
        general.openAttribute("Matsubara modes").read(PredType::NATIVE_HBOOL, &matsubara);
        ctx.check((bool) matsubara == lp.energyMomentumTensorCorrFunctionsGeneralTauMatsubara(), "attribute G_EMT_general_tau/Matsubara modes");

        file.close();
    } catch (const Exception &error) {
        rootLogger.error(CoutColors::red, "TEST FAILED: ", CoutColors::reset, "reading ", fileName, " threw: ", error.getDetailMsg());
        ctx.lerror = true;
    }

    std::remove(fileName.c_str());

    if (ctx.lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green, "passed!", CoutColors::reset);
    }

    return 0;
}
