/* 
 * hdf5fileWriter.h
 *
 * File for writing gradient flow data to HDF5 files
 *
 * All HDF5 objects are held by value; the H5Cpp handles close themselves on destruction,
 * so the file is closed when the writer goes out of scope.
 * Only the root rank creates and writes the file.
 *
 */

#pragma once
#include "H5Cpp.h"
#include <vector>
#include <string>
#include "../communication/communicationBase.h"
#include "../latticeParameters.h"
#include "../../modules/tensor_decomposition/tensorDecomposition.h"
#include "../../modules/gradientFlow/gradientFlowParameters.h"

using namespace H5;

template<class floatT>
struct ComplexData {
    floatT real;
    floatT imag;
};

enum class HDF5_Observable {
    FlowTime,
    FlowTimeMeasured,
    Plaquette,
    Clover,
    TopCharge,
    TopChargeImp,
    EMTU,
    EMTE,
    EMTCorrAveragedTau,
    EMTCorrGeneralTau
};


template<class floatT>
class HDF5FileWriter {
    private:
        CommunicationBase& _commBase;
        LatticeParameters& _latParams;

        H5File _file;
        const H5std_string _fileName;
        Group groupGradFlowMeasurements;
        const H5std_string groupNameGradFlowMeasurements;
        Group groupCorr;
        const H5std_string groupNameCorr;

        hsize_t r2max;
        hsize_t tauMax;
        hsize_t hitR2;
        
        // dimensions for quantities with:
        // scalar,                  flow-time dependent,                e.g. Q(tau_F)
        hsize_t initDimsFlowTimeQuantity[1] = {0};
        hsize_t maxDimsFlowTimeQuantity[1] = {H5S_UNLIMITED};
        hsize_t chunkSizeFlowTimeQuantity[1] = {1};
        // scalar,                  r^2 dependent,                      e.g. N_counts(r^2)
        hsize_t initDimsR2Counts[1]; // = {r2max + 1}
        hsize_t maxDimsR2Counts[1]; // = {r2max + 1}
        // 4x4Sym (10 components),  flow-time dependent,                e.g. T_munu(tau_F)
        hsize_t initDims4x4Sym[2] = {10, 0};
        hsize_t maxDims4x4Sym[2] = {10, H5S_UNLIMITED};
        hsize_t chunkSize4x4Sym[2] = {10, 1};
        // no tensor structure,     flow-time, tau and r^2 dependent,        e.g. G_Q(tau_F, tau, r^2)
        hsize_t initDimsTopChargeCorrGeneralTau[3];                     // = {tau_F, tau_max + 1, r2max + 1}
        hsize_t maxDimsTopChargeCorrGeneralTau[3];                      // = {H5S_UNLIMITED, tau_max + 1, r2max + 1}
        hsize_t chunkSizeTopChargeCorrGeneralTau[3];                    // = {1, tau_max + 1, r2max + 1}
        // 10 tensor functions,     flow-time and r^2 dependent,        e.g. G_LL(tau_F, r^2)
        hsize_t initDimsEMTCorrAveragedTau[3];                          // = {10, 0, r2max + 1}
        hsize_t maxDimsEMTCorrAveragedTau[3];                           // = {10, H5S_UNLIMITED, r2max + 1}
        hsize_t chunkSizeEMTCorrAveragedTau[3];                         // = {1, 1, r2max + 1}
        // 14 tensor functions,     flow-time, tau and r^2 dependent,   e.g. G_MT(tau_F, tau, r^2)
        hsize_t initDimsEMTCorrGeneralTau[4];                           // = {14, 0, N_t, r2max + 1}
        hsize_t maxDimsEMTCorrGeneralTau[4];                            // = {14, H5S_UNLIMITED, N_t, r2max + 1}
        hsize_t chunkSizeEMTCorrGeneralTau[4];                          // = {1, 1, N_t, r2max + 1}
        
        const H5::DataType* hdf5FloatT = nullptr;

        const H5std_string dataSetNameFlowTime;
        const H5std_string dataSetNameFlowTimeNecessary;
        const H5std_string dataSetNamePlaquette, dataSetNameClover;
        const H5std_string dataSetNameTopCharge, dataSetNameTopChargeImp;
        const H5std_string dataSetNameR2Counts, dataSetNameR2Values;
        const H5std_string dataSetNameEMTE, dataSetNameEMTU;
        const H5std_string dataSetNameTopChargeCorrGeneralTau;
        const H5std_string dataSetNameEMTCorrAveragedTau;
        const H5std_string dataSetNameEMTCorrGeneralTau;

        DataSet dataSetFlowTime;
        DataSet dataSetFlowTimeMeasured;
        DataSet dataSetPlaquette, dataSetClover;
        DataSet dataSetTopCharge, dataSetTopChargeImp;
        DataSet dataSetR2Counts, dataSetR2Values;
        DataSet dataSetEMTE, dataSetEMTU;
        DataSet dataSetTopChargeCorrGeneralTau;
        DataSet dataSetEMTCorrAveragedTau;
        DataSet dataSetEMTCorrGeneralTau;

    public:
        // standard constructor
        HDF5FileWriter(CommunicationBase& commBase, LatticeParameters& latParams, const std::string& fileName) :
            _commBase(commBase), _latParams(latParams), _fileName(fileName),
            groupNameGradFlowMeasurements("/gradient_flow_measurements"),
            groupNameCorr("correlators"),
            dataSetNameFlowTime("flow_time"), dataSetNameFlowTimeNecessary("flow_time"),
            dataSetNamePlaquette("plaquette"), dataSetNameTopCharge("topological_charge"),
            dataSetNameClover("clover"), dataSetNameTopChargeImp("topological_charge_improved"),
            dataSetNameR2Counts("r2_counts"), dataSetNameR2Values("r2_values"),
            dataSetNameEMTE("EMTE"), dataSetNameEMTU("EMTU"),
            dataSetNameTopChargeCorrGeneralTau("G_Qtop_general_tau"),
            dataSetNameEMTCorrAveragedTau("G_EMT_averaged_tau"),
            dataSetNameEMTCorrGeneralTau("G_EMT_general_tau")
        {
            
            // fix PredType based on floatT
            if constexpr (std::is_same_v<floatT, double>) {
                hdf5FloatT = &PredType::NATIVE_DOUBLE;
            } else if constexpr (std::is_same_v<floatT, float>) {
                hdf5FloatT = &PredType::NATIVE_FLOAT;
            }

            r2max = TensorDecomposition<floatT, 0>::getR2max();
            tauMax = TensorDecomposition<floatT, 0>::getTauMax();
            hitR2 = TensorDecomposition<floatT, 0>::getNumberOfHitR2();

            // set initial dimension: one for component functions, one for flow time, one for tau, hitR2 for separations
            initDimsR2Counts[0] = hitR2;
            initDimsTopChargeCorrGeneralTau[0] = 0;
            initDimsTopChargeCorrGeneralTau[1] = tauMax + 1;
            initDimsTopChargeCorrGeneralTau[2] = hitR2;
            initDimsEMTCorrAveragedTau[0] = 10;
            initDimsEMTCorrAveragedTau[1] = 0;
            initDimsEMTCorrAveragedTau[2] = hitR2;
            initDimsEMTCorrGeneralTau[0] = 14;
            initDimsEMTCorrGeneralTau[1] = 0;
            initDimsEMTCorrGeneralTau[2] = tauMax + 1;
            initDimsEMTCorrGeneralTau[3] = hitR2;

            // set maximum dimensions: one for component functions, one for flow time, one for tau, hitR2 for separations
            maxDimsR2Counts[0] = hitR2;
            maxDimsTopChargeCorrGeneralTau[0] = H5S_UNLIMITED;
            maxDimsTopChargeCorrGeneralTau[1] = tauMax + 1;
            maxDimsTopChargeCorrGeneralTau[2] = hitR2;
            maxDimsEMTCorrAveragedTau[0] = 10;
            maxDimsEMTCorrAveragedTau[1] = H5S_UNLIMITED;
            maxDimsEMTCorrAveragedTau[2] = hitR2;
            maxDimsEMTCorrGeneralTau[0] = 14;
            maxDimsEMTCorrGeneralTau[1] = H5S_UNLIMITED;
            maxDimsEMTCorrGeneralTau[2] = tauMax + 1;
            maxDimsEMTCorrGeneralTau[3] = hitR2;

            // set chunk size: one for component functions, one for flow time, one for tau, hitR2 for separations
            chunkSizeTopChargeCorrGeneralTau[0] = 1;
            chunkSizeTopChargeCorrGeneralTau[1] = 1;
            chunkSizeTopChargeCorrGeneralTau[2] = hitR2;
            chunkSizeEMTCorrAveragedTau[0] = 1;
            chunkSizeEMTCorrAveragedTau[1] = 1;
            chunkSizeEMTCorrAveragedTau[2] = hitR2;
            chunkSizeEMTCorrGeneralTau[0] = 1;
            chunkSizeEMTCorrGeneralTau[1] = 1;
            chunkSizeEMTCorrGeneralTau[2] = 1;
            chunkSizeEMTCorrGeneralTau[3] = hitR2;

            // only the root rank creates the file; all other ranks keep empty handles and never touch them
            if (!IamRoot()) return;

            // create the HDF5 file
            _file = H5File(_fileName, H5F_ACC_TRUNC);

            // create groups
            groupGradFlowMeasurements = _file.createGroup(groupNameGradFlowMeasurements);
            groupCorr = groupGradFlowMeasurements.createGroup(groupNameCorr);

            // create dataSpaces
            DataSpace dataSpaceFlowTimeQuantity(1, initDimsFlowTimeQuantity, maxDimsFlowTimeQuantity);
            DataSpace dataSpaceR2Counts(1, initDimsR2Counts, maxDimsR2Counts);
            DataSpace dataSpace4x4Sym(2, initDims4x4Sym, maxDims4x4Sym);
            DataSpace dataSpaceTopChargeCorrGeneralTau(3, initDimsTopChargeCorrGeneralTau, maxDimsTopChargeCorrGeneralTau);
            DataSpace dataSpaceEMTCorrAveragedTau(3, initDimsEMTCorrAveragedTau, maxDimsEMTCorrAveragedTau);
            DataSpace dataSpaceEMTCorrGeneralTau(4, initDimsEMTCorrGeneralTau, maxDimsEMTCorrGeneralTau);

            // create dataset property list and set the chunking
            DSetCreatPropList propListFlowTimeQuantity;
            propListFlowTimeQuantity.setChunk(1, chunkSizeFlowTimeQuantity);
            DSetCreatPropList propList4x4Sym;
            propList4x4Sym.setChunk(2, chunkSize4x4Sym);
            DSetCreatPropList propListTopChargeCorrGeneralTau;
            propListTopChargeCorrGeneralTau.setChunk(3, chunkSizeTopChargeCorrGeneralTau);
            DSetCreatPropList propListEMTCorrAveragedTau;
            propListEMTCorrAveragedTau.setChunk(3, chunkSizeEMTCorrAveragedTau);
            DSetCreatPropList propListEMTCorrGeneralTau;
            propListEMTCorrGeneralTau.setChunk(4, chunkSizeEMTCorrGeneralTau);

            // create compound data type for storing complex numbers
            // compTypeComplex = new CompType(sizeof(ComplexData<floatT>));
            // compTypeComplex->insertMember("real", HOFFSET(ComplexData<floatT>, real), *hdf5FloatT);
            // compTypeComplex->insertMember("imag", HOFFSET(ComplexData<floatT>, imag), *hdf5FloatT);

            // create datasets
            dataSetFlowTime = groupGradFlowMeasurements.createDataSet(dataSetNameFlowTime, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetPlaquette = groupGradFlowMeasurements.createDataSet(dataSetNamePlaquette, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetClover = groupGradFlowMeasurements.createDataSet(dataSetNameClover, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetTopCharge = groupGradFlowMeasurements.createDataSet(dataSetNameTopCharge, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetTopChargeImp = groupGradFlowMeasurements.createDataSet(dataSetNameTopChargeImp, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetEMTE = groupGradFlowMeasurements.createDataSet(dataSetNameEMTE, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetEMTU = groupGradFlowMeasurements.createDataSet(dataSetNameEMTU, *hdf5FloatT, dataSpace4x4Sym, propList4x4Sym);

            dataSetFlowTimeMeasured = groupCorr.createDataSet(dataSetNameFlowTimeNecessary, *hdf5FloatT, dataSpaceFlowTimeQuantity, propListFlowTimeQuantity);
            dataSetR2Counts = groupCorr.createDataSet(dataSetNameR2Counts, PredType::NATIVE_INT, dataSpaceR2Counts);
            dataSetR2Values = groupCorr.createDataSet(dataSetNameR2Values, PredType::NATIVE_INT, dataSpaceR2Counts);
            dataSetTopChargeCorrGeneralTau = groupCorr.createDataSet(dataSetNameTopChargeCorrGeneralTau, *hdf5FloatT, dataSpaceTopChargeCorrGeneralTau, propListTopChargeCorrGeneralTau);
            dataSetEMTCorrAveragedTau = groupCorr.createDataSet(dataSetNameEMTCorrAveragedTau, *hdf5FloatT, dataSpaceEMTCorrAveragedTau, propListEMTCorrAveragedTau);
            dataSetEMTCorrGeneralTau = groupCorr.createDataSet(dataSetNameEMTCorrGeneralTau, *hdf5FloatT, dataSpaceEMTCorrGeneralTau, propListEMTCorrGeneralTau);

        }

        bool IamRoot() {
            return _commBase.IamRoot();
        }

        // write everything buffered so far to disk, so that a killed run leaves a readable file
        void flush() {
            if (IamRoot()) {
                _file.flush(H5F_SCOPE_GLOBAL);
            }
        }

        void writeBoolAttribute(
            H5Object& object,
            const std::string& attributeName,
            bool value
        ) {
            DataSpace dataSpaceScalar(H5S_SCALAR);

            hbool_t hvalue = value;

            object.createAttribute(attributeName, PredType::NATIVE_HBOOL, dataSpaceScalar).write(PredType::NATIVE_HBOOL, &hvalue);
        }

        void writeStringArrayAttribute(
            H5Object& object,
            const std::string& attributeName,
            const std::vector<std::string>& strings
        ) {
            StrType strType(PredType::C_S1, H5T_VARIABLE);

            hsize_t dims[1] = {strings.size()};
            DataSpace dataSpace(1, dims);

            std::vector<const char*> chars;
            chars.reserve(strings.size());
            for (auto& s : strings) chars.push_back(s.c_str());

            object.createAttribute(attributeName, strType, dataSpace).write(strType, chars.data());
        }

        void writeAttributes(gradientFlowParam<floatT> &parameters) {
            if (IamRoot()) {
                hsize_t latDimDims[1] = {4};
                hsize_t indicesDims[1] = {10};
                DataSpace dataSpaceScalar = DataSpace(H5S_SCALAR);
                DataSpace dataSpaceLatDim(1, latDimDims);
                DataSpace dataSpaceIndices(1, indicesDims);

                _file.createAttribute("N", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.latDim);
                _file.createAttribute("nodes", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.nodeDim);
                _file.createAttribute("GPU topology", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.gpuTopo);
                _file.createAttribute("configuration Number", PredType::NATIVE_INT, dataSpaceScalar).write(PredType::NATIVE_INT, &parameters.confnumber.ref());
                _file.createAttribute("beta", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.beta.ref());
                _file.createAttribute("gauge file", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.GaugefileName.ref());

                groupGradFlowMeasurements.createAttribute("gradient flow force", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.force.ref());
                groupGradFlowMeasurements.createAttribute("Runge-Kutta fixed/adaptive step size", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.RK_method.ref());
                groupGradFlowMeasurements.createAttribute("start step size", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.start_step_size.ref());
                groupGradFlowMeasurements.createAttribute("adaptive step size accuracy", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.accuracy.ref());

                std::vector<std::string> EMTNumbers = {
                    "00", "11", "22", "33",
                    "01", "02", "03",
                    "12", "13",
                    "23"
                };
                std::vector<std::string> EMTUNames = {
                    "xx", "yy", "zz", "tt",
                    "xy", "xz", "xt",
                    "yz", "yt",
                    "zt"
                };

                std::vector<std::string> emtCorrComponentFunctionNamesAveragedTau = {
                    "TT", "RT",
                    "UT", "UR",
                    "ss", "ll", "ww",
                    "sl", "sw", "lw"
                };

                std::vector<std::string> emtCorrComponentFunctionNamesGeneralTau = {
                    "TT",
                    "RT", "MT", "UT",
                    "ss", "ll", "ww", "mm",
                    "sl", "sw", "sm", "lw", "lm", "wm"
                };

                this->writeStringArrayAttribute(dataSetEMTU, "index pairs", EMTNumbers);
                this->writeStringArrayAttribute(dataSetEMTU, "index names", EMTUNames);
                this->writeStringArrayAttribute(dataSetEMTCorrAveragedTau, "component names", emtCorrComponentFunctionNamesAveragedTau);
                this->writeStringArrayAttribute(dataSetEMTCorrGeneralTau, "component names", emtCorrComponentFunctionNamesGeneralTau);

                this->writeBoolAttribute(dataSetEMTCorrGeneralTau, "Matsubara modes", parameters.energyMomentumTensorCorrFunctionsGeneralTauMatsubara());
            }
        }

        void writeR2CountsAndValues(const std::vector<int>& vecCounts) {
            if (IamRoot()) {
                std::vector<int> filteredR2Values = std::vector<int>();
                std::vector<int> filteredCounts = std::vector<int>();

                for (int r2 = 0; r2 < vecCounts.size(); r2++) {
                    if (vecCounts[r2] != 0) {
                        filteredCounts.push_back(vecCounts[r2]);
                        filteredR2Values.push_back(r2);
                    }
                }

                dataSetR2Values.write(filteredR2Values.data(), PredType::NATIVE_INT);
                dataSetR2Counts.write(filteredCounts.data(), PredType::NATIVE_INT);
            }
        }

        void writeFlowTimeQuantity(DataSet &dataSet, const floatT flowTimeQuantity) {
            if (!IamRoot()) return;

            // get dataspace of dataset
            DataSpace fileSpaceFlowTimeQuantity = dataSet.getSpace();

            // get rank
            int rank = fileSpaceFlowTimeQuantity.getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceFlowTimeQuantity.getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[1] = {currentDims[0]};
            hsize_t amount[1] = {1};
            hsize_t newsize[1] = {currentDims[0]+1};
            dataSet.extend(newsize);

            // select hyperslab in file
            fileSpaceFlowTimeQuantity = dataSet.getSpace();
            fileSpaceFlowTimeQuantity.selectHyperslab(H5S_SELECT_SET, amount, offset);

            // create memory space
            DataSpace memorySpace(1, amount, NULL);

            dataSet.write(&flowTimeQuantity, *hdf5FloatT, memorySpace, fileSpaceFlowTimeQuantity);
        }

        template<HDF5_Observable obs>
        void writeObservable(const floatT value) {
            switch (obs) {
                case HDF5_Observable::FlowTime:
                    writeFlowTimeQuantity(dataSetFlowTime, value);
                    break;
                case HDF5_Observable::FlowTimeMeasured:
                    writeFlowTimeQuantity(dataSetFlowTimeMeasured, value);
                    break;
                case HDF5_Observable::Plaquette:
                    writeFlowTimeQuantity(dataSetPlaquette, value);
                    break;
                case HDF5_Observable::Clover:
                    writeFlowTimeQuantity(dataSetClover, value);
                    break;
                case HDF5_Observable::TopCharge:
                    writeFlowTimeQuantity(dataSetTopCharge, value);
                    break;
                case HDF5_Observable::TopChargeImp:
                    writeFlowTimeQuantity(dataSetTopChargeImp, value);
                    break;
                case HDF5_Observable::EMTE:
                    writeFlowTimeQuantity(dataSetEMTE, value);
                    break;
            }
        }

        void write4x4Sym(DataSet &dataSet, const std::vector<floatT>& vec4x4SymComponents) {
            if (!IamRoot()) return;

            // get dataspace of dataset
            DataSpace fileSpace4x4Sym = dataSet.getSpace();

            // get rank
            int rank = fileSpace4x4Sym.getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpace4x4Sym.getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[2] = {0, currentDims[1]};
            hsize_t amount[2] = {10, 1};
            hsize_t newsize[2] = {10, currentDims[1]+1};
            dataSet.extend(newsize);

            // select hyperslab in file
            fileSpace4x4Sym = dataSet.getSpace();
            fileSpace4x4Sym.selectHyperslab(H5S_SELECT_SET, amount, offset);

            // create memory space
            DataSpace memorySpace(2, amount, NULL);

            dataSet.write(vec4x4SymComponents.data(), *hdf5FloatT, memorySpace, fileSpace4x4Sym);
        }

        void writeEMTU(const std::vector<floatT>& vecEMTUComponents) {
            write4x4Sym(dataSetEMTU, vecEMTUComponents);
        }

        void writeTopChargeGeneralTauData(
            const std::vector<std::vector<COMPLEX(floatT)>>& vecTopChargeCorrComplex,
            const std::vector<int>& vecR2Counts,
            const bool matsubara
        ) {
            if (!IamRoot()) return;

            // create vector of floatT instead of COMPLEX(floatT) to store real or imag parts
            std::vector<std::vector<floatT>> vecTopChargeCorrComplexTransformed(tauMax + 1);
            for (int t = 0; t < tauMax + 1; t++)
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                if (vecR2Counts[r2] != 0) {
                    vecTopChargeCorrComplexTransformed[t].push_back(real(vecTopChargeCorrComplex[t][r2]));
                }
            }

            // flatten array
            std::vector<floatT> vecTopChargeCorrComplexTransformedFlat((tauMax + 1) * (hitR2));
            for (int t = 0; t < tauMax + 1; t++)
            for (int r2 = 0; r2 < hitR2; r2++) {
                vecTopChargeCorrComplexTransformedFlat[t * hitR2 + r2] = vecTopChargeCorrComplexTransformed[t][r2];
            }

            // get dataspace of dataset
            DataSpace fileSpaceCorr = dataSetTopChargeCorrGeneralTau.getSpace();

            // get rank
            int rank = fileSpaceCorr.getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceCorr.getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[3] = {currentDims[0], 0, 0};
            hsize_t amount[3] = {1, currentDims[1], currentDims[2]};
            hsize_t newsize[3] = {currentDims[0]+1, currentDims[1], currentDims[2]};
            dataSetTopChargeCorrGeneralTau.extend(newsize);
            
            // select hyperslab in file
            fileSpaceCorr = dataSetTopChargeCorrGeneralTau.getSpace();
            fileSpaceCorr.selectHyperslab(H5S_SELECT_SET, amount, offset);
            
            // create memory space
            DataSpace memorySpace(3, amount, NULL);
            
            // write complex data
            dataSetTopChargeCorrGeneralTau.write(vecTopChargeCorrComplexTransformedFlat.data(), *hdf5FloatT, memorySpace, fileSpaceCorr);
        }

        void writeEMTCorrAveragedTauData(
            const std::vector<std::vector<COMPLEX(floatT)>>& vecEMTcorrComplex,
            const std::vector<int>& vecR2Counts
        ) {
            if (!IamRoot()) return;

            // create vector of floatT instead of COMPLEX(floatT) to store just the real parts
            std::vector<std::vector<floatT>> vecEMTCorrComplexTransformed(10);
            for (int i = 0; i < 10; i++)
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                if (vecR2Counts[r2] != 0) {
                    vecEMTCorrComplexTransformed[i].push_back(real(vecEMTcorrComplex[i][r2]));
                }
            }

            // flatten array
            std::vector<floatT> vecEMTCorrComplexDataTransformedFlat(10 * (hitR2));
            for (int i = 0; i < 10; i++)
            for (int r2 = 0; r2 < hitR2; r2++) {
                vecEMTCorrComplexDataTransformedFlat[i * (hitR2) + r2] = vecEMTCorrComplexTransformed[i][r2];
            }

            // get dataspace of dataset
            DataSpace fileSpaceCorr = dataSetEMTCorrAveragedTau.getSpace();

            // get rank
            int rank = fileSpaceCorr.getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceCorr.getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[3] = {0, currentDims[1], 0};
            hsize_t amount[3] = {currentDims[0], 1, currentDims[2]};
            hsize_t newsize[3] = {currentDims[0], currentDims[1]+1, currentDims[2]};
            dataSetEMTCorrAveragedTau.extend(newsize);
            
            // select hyperslab in file
            fileSpaceCorr = dataSetEMTCorrAveragedTau.getSpace();
            fileSpaceCorr.selectHyperslab(H5S_SELECT_SET, amount, offset);
            
            // create memory space
            DataSpace memorySpace(3, amount, NULL);
            
            // write complex data
            dataSetEMTCorrAveragedTau.write(vecEMTCorrComplexDataTransformedFlat.data(), *hdf5FloatT, memorySpace, fileSpaceCorr);
        }

        void writeEMTCorrGeneralTauData(
            const std::vector<std::vector<std::vector<COMPLEX(floatT)>>>& vecEMTcorrComplex,
            const std::vector<int>& vecR2Counts,
            const bool matsubara
        ) {
            if (!IamRoot()) return;

            // create vector of floatT instead of COMPLEX(floatT) to store real or imag parts
            std::vector<std::vector<std::vector<floatT>>> vecEMTCorrComplexTransformed(14, std::vector<std::vector<floatT>>(tauMax + 1));
            for (int i = 0; i < 14; i++)
            for (int t = 0; t < tauMax + 1; t++)
            for (int r2 = 0; r2 < r2max + 1; r2++) {
                if (vecR2Counts[r2] != 0) {
                    if (matsubara && (i == 2 || i == 10 || i == 12 || i == 13)) {
                        vecEMTCorrComplexTransformed[i][t].push_back(imag(vecEMTcorrComplex[i][t][r2]));
                    } else {
                        vecEMTCorrComplexTransformed[i][t].push_back(real(vecEMTcorrComplex[i][t][r2]));
                    }
                }
            }

            // flatten array
            std::vector<floatT> vecEMTCorrComplexDataTransformedFlat(14 * (tauMax + 1) * (hitR2));
            for (int i = 0; i < 14; i++)
            for (int t = 0; t < tauMax + 1; t++)
            for (int r2 = 0; r2 < hitR2; r2++) {
                vecEMTCorrComplexDataTransformedFlat[(i * (tauMax + 1) + t) * (hitR2) + r2] = vecEMTCorrComplexTransformed[i][t][r2];
            }

            // get dataspace of dataset
            DataSpace fileSpaceCorr = dataSetEMTCorrGeneralTau.getSpace();

            // get rank
            int rank = fileSpaceCorr.getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceCorr.getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[4] = {0, currentDims[1], 0, 0};
            hsize_t amount[4] = {currentDims[0], 1, currentDims[2], currentDims[3]};
            hsize_t newsize[4] = {currentDims[0], currentDims[1]+1, currentDims[2], currentDims[3]};
            dataSetEMTCorrGeneralTau.extend(newsize);
            
            // select hyperslab in file
            fileSpaceCorr = dataSetEMTCorrGeneralTau.getSpace();
            fileSpaceCorr.selectHyperslab(H5S_SELECT_SET, amount, offset);
            
            // create memory space
            DataSpace memorySpace(4, amount, NULL);
            
            // write complex data
            dataSetEMTCorrGeneralTau.write(vecEMTCorrComplexDataTransformedFlat.data(), *hdf5FloatT, memorySpace, fileSpaceCorr);
        }

};
