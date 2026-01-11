//
// hdf5fileWriter.h
//
// File for writing gradient flow data to HDF5 files
//

#include "H5Cpp.h"
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
    EMTCorr
};


template<class floatT>
class HDF5FileWriter {
    private:
        CommunicationBase& _commBase;
        LatticeParameters& _latParams;

        H5File* _file;
        const H5std_string _fileName;
        Group* groupGradFlowMeasurements;
        const H5std_string groupNameGradFlowMeasurements;
        Group* groupEMTCorr;
        const H5std_string groupNameEMTCorr;

        hsize_t r2max;
        
        // dimensions for quantities with:
        // scalar,                  flow-time dependent,            e.g. Q(tau_F)
        hsize_t initDimsFlowTimeQuantity[1] = {0};
        hsize_t maxDimsFlowTimeQuantity[1] = {H5S_UNLIMITED};
        hsize_t chunkSizeFlowTimeQuantity[1] = {1};
        // scalar,                  r^2 dependent,                  e.g. N_counts(r^2)
        hsize_t initDimsR2Counts[1]; // = {r2max + 1}
        hsize_t maxDimsR2Counts[1]; // = {r2max + 1}
        // 4x4Sym (10 components),  flow-time dependent,            e.g. T_munu(tau_F)
        hsize_t initDims4x4Sym[2] = {0, 10};
        hsize_t maxDims4x4Sym[2] = {H5S_UNLIMITED, 10};
        hsize_t chunkSize4x4Sym[2] = {1, 10};
        // 10 tensor functions,     flow-time and r^2 dependent,    e.g. G_LL(tau_F, r^2)
        hsize_t initDimsEMTCorr[3]; // = {10, 0, r2max + 1}
        hsize_t maxDimsEMTCorr[3]; // = {10, H5S_UNLIMITED, r2max + 1}
        hsize_t chunkSizeEMTCorr[3]; // = {1, 1, r2max + 1}
        
        const H5::DataType* hdf5FloatT = nullptr;
        CompType *compTypeComplex;

        DataSpace *dataSpaceFlowTimeQuantity;
        DataSpace *dataSpaceR2Counts;
        DataSpace *dataSpace4x4Sym;
        DataSpace *dataSpaceEMTCorr;

        const H5std_string dataSetNameFlowTime;
        const H5std_string dataSetNameFlowTimeNecessary;
        const H5std_string dataSetNamePlaquette, dataSetNameClover;
        const H5std_string dataSetNameTopCharge, dataSetNameTopChargeImp;
        const H5std_string dataSetNameR2Counts;
        const H5std_string dataSetNameEMTE, dataSetNameEMTU;
        const H5std_string dataSetNameEMTCorr;

        DataSet *dataSetFlowTime;
        DataSet *dataSetFlowTimeMeasured;
        DataSet *dataSetPlaquette, *dataSetClover;
        DataSet *dataSetTopCharge, *dataSetTopChargeImp;
        DataSet *dataSetR2Counts;
        DataSet *dataSetEMTE, *dataSetEMTU;
        DataSet *dataSetEMTCorr;

    public:
        // standard constructor
        HDF5FileWriter(CommunicationBase& commBase, LatticeParameters& latParams, const std::string& fileName) :
            _commBase(commBase), _latParams(latParams), _fileName(fileName),
            groupNameGradFlowMeasurements("/gradient_flow_measurements"),
            groupNameEMTCorr("EMT_correlator"),
            dataSetNameFlowTime("flow_time"), dataSetNameFlowTimeNecessary("measured_flow_time"),
            dataSetNamePlaquette("plaquette"), dataSetNameTopCharge("topological_charge"),
            dataSetNameClover("clover"), dataSetNameTopChargeImp("topological_charge_improved"),
            dataSetNameR2Counts("r2_counts"), dataSetNameEMTE("EMTE"), dataSetNameEMTU("EMTU"), dataSetNameEMTCorr("G") {
            
            // fix PredType based on floatT
            if constexpr (std::is_same_v<floatT, double>) {
                hdf5FloatT = &PredType::NATIVE_DOUBLE;
            } else if constexpr (std::is_same_v<floatT, float>) {
                hdf5FloatT = &PredType::NATIVE_FLOAT;
            }

            r2max = TensorDecomposition<floatT, 0>::getR2max();

            // create the HDF5 file
            _file = new H5File(_fileName, H5F_ACC_TRUNC);

            // create groups
            groupGradFlowMeasurements = new Group(_file->createGroup(groupNameGradFlowMeasurements));
            groupEMTCorr = new Group(groupGradFlowMeasurements->createGroup(groupNameEMTCorr));

            // set initial dimension to zero for flow time, r2max+1 for separations
            initDimsR2Counts[0] = r2max + 1;
            initDimsEMTCorr[0] = 10;
            initDimsEMTCorr[1] = 0;
            initDimsEMTCorr[2] = r2max + 1;

            // set maximum dimensions, unlimited for flow time, r2max+1 for separations
            maxDimsR2Counts[0] = r2max + 1;
            maxDimsEMTCorr[0] = 10;
            maxDimsEMTCorr[1] = H5S_UNLIMITED;
            maxDimsEMTCorr[2] = r2max + 1;

            // set chunk size, one for flow time, r2max+1 for separations
            chunkSizeEMTCorr[0] = 1;
            chunkSizeEMTCorr[1] = 1;
            chunkSizeEMTCorr[2] = r2max + 1;

            // create dataSpaces
            dataSpaceFlowTimeQuantity = new DataSpace(1, initDimsFlowTimeQuantity, maxDimsFlowTimeQuantity);
            dataSpaceR2Counts = new DataSpace(1, initDimsR2Counts, maxDimsR2Counts);
            dataSpace4x4Sym = new DataSpace(2, initDims4x4Sym, maxDims4x4Sym);
            dataSpaceEMTCorr = new DataSpace(3, initDimsEMTCorr, maxDimsEMTCorr);
            
            // create dataset property list and set the chunking
            DSetCreatPropList propListFlowTimeQuantity;
            propListFlowTimeQuantity.setChunk(1, chunkSizeFlowTimeQuantity);
            DSetCreatPropList propList4x4Sym;
            propList4x4Sym.setChunk(2, chunkSize4x4Sym);
            DSetCreatPropList propListEMTCorr;
            propListEMTCorr.setChunk(3, chunkSizeEMTCorr);

            // create compound data type for storing complex numbers
            compTypeComplex = new CompType(sizeof(ComplexData<floatT>));
            compTypeComplex->insertMember("real", HOFFSET(ComplexData<floatT>, real), *hdf5FloatT);
            compTypeComplex->insertMember("imag", HOFFSET(ComplexData<floatT>, imag), *hdf5FloatT);

            // create datasets
            dataSetFlowTime = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameFlowTime, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetPlaquette = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNamePlaquette, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetClover = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameClover, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetTopCharge = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameTopCharge, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetTopChargeImp = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameTopChargeImp, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetEMTE = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameEMTE, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetEMTU = new DataSet(groupGradFlowMeasurements->createDataSet(dataSetNameEMTU, *hdf5FloatT, *dataSpace4x4Sym, propList4x4Sym));
            
            dataSetFlowTimeMeasured = new DataSet(groupEMTCorr->createDataSet(dataSetNameFlowTimeNecessary, *hdf5FloatT, *dataSpaceFlowTimeQuantity, propListFlowTimeQuantity));
            dataSetR2Counts = new DataSet(groupEMTCorr->createDataSet(dataSetNameR2Counts, PredType::NATIVE_INT, *dataSpaceR2Counts));
            dataSetEMTCorr = new DataSet(groupEMTCorr->createDataSet(dataSetNameEMTCorr, *compTypeComplex, *dataSpaceEMTCorr, propListEMTCorr));

        }

        bool IamRoot() {
            return _commBase.IamRoot();
        }

        void writeAttributes(gradientFlowParam<floatT> &parameters) {
            if (IamRoot()) {
                hsize_t latDimDims[1] = {4};
                hsize_t indicesDims[1] = {10};
                DataSpace dataSpaceScalar = DataSpace(H5S_SCALAR);
                DataSpace dataSpaceLatDim(1, latDimDims);
                DataSpace dataSpaceIndices(1, indicesDims);

                _file->createAttribute("N", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.latDim);
                _file->createAttribute("nodes", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.nodeDim);
                _file->createAttribute("GPU topology", PredType::NATIVE_INT, dataSpaceLatDim).write(PredType::NATIVE_INT, parameters.gpuTopo);
                _file->createAttribute("configuration Number", PredType::NATIVE_INT, dataSpaceScalar).write(PredType::NATIVE_INT, &parameters.confnumber.ref());
                _file->createAttribute("beta", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.beta.ref());
                _file->createAttribute("gauge file", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.GaugefileName.ref());

                groupGradFlowMeasurements->createAttribute("gradient flow force", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.force.ref());
                groupGradFlowMeasurements->createAttribute("Runge-Kutta fixed/adaptive step size", StrType(PredType::C_S1, 256), dataSpaceScalar).write(StrType(PredType::C_S1, 256), parameters.RK_method.ref());
                groupGradFlowMeasurements->createAttribute("start step size", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.start_step_size.ref());
                groupGradFlowMeasurements->createAttribute("adaptive step size accuracy", PredType::NATIVE_DOUBLE, dataSpaceScalar).write(PredType::NATIVE_DOUBLE, &parameters.accuracy.ref());

                const char* EMTNumbers[10] = {
                    "00", "11", "22", "33"
                    "01", "02", "03",
                    "12", "13"
                    "23"
                };
                const char* EMTUNames[10] = {
                    "xx", "yy", "zz", "tt"
                    "xy", "xz", "xt",
                    "yz", "yt"
                    "zt"
                };
                const char* tensorComponentsNames[10] = {
                    "TT", "LL",
                    "T", "L",
                    "SS", "LL", "WW",
                    "SL", "SW", "LW"
                };

                StrType strType(PredType::C_S1, 256);

                dataSetEMTU->createAttribute("index pairs", strType, dataSpaceIndices).write(strType, EMTNumbers);
                dataSetEMTU->createAttribute("index names", strType, dataSpaceIndices).write(strType, EMTUNames);
                dataSetEMTCorr->createAttribute("component names", strType, dataSpaceIndices).write(strType, tensorComponentsNames);
            }
        }

        void writeR2Counts(const std::vector<int>& vecCounts) {
            if (IamRoot()) {
                dataSetR2Counts->write(vecCounts.data(), PredType::NATIVE_INT);
            }
        }

        void writeFlowTimeQuantity(DataSet &dataSet, const floatT flowTimeQuantity) {
            // get dataspace of dataset
            DataSpace *fileSpaceFlowTimeQuantity = new DataSpace(dataSet.getSpace());

            // get rank
            int rank = fileSpaceFlowTimeQuantity->getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceFlowTimeQuantity->getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[1] = {currentDims[0]};
            hsize_t amount[1] = {1};
            hsize_t newsize[1] = {currentDims[0]+1};
            dataSet.extend(newsize);

            // select hyperslab in file
            fileSpaceFlowTimeQuantity = new DataSpace(dataSet.getSpace());
            fileSpaceFlowTimeQuantity->selectHyperslab(H5S_SELECT_SET, amount, offset);

            // create memory space
            DataSpace *memorySpace = new DataSpace(1, amount, NULL);

            if (IamRoot()) {
                dataSet.write(&flowTimeQuantity, *hdf5FloatT, *memorySpace, *fileSpaceFlowTimeQuantity);
            }
        }

        template<HDF5_Observable obs>
        void writeObservable(const floatT value) {
            switch (obs) {
                case HDF5_Observable::FlowTime:
                    writeFlowTimeQuantity(*dataSetFlowTime, value);
                    break;
                case HDF5_Observable::FlowTimeMeasured:
                    writeFlowTimeQuantity(*dataSetFlowTimeMeasured, value);
                    break;
                case HDF5_Observable::Plaquette:
                    writeFlowTimeQuantity(*dataSetPlaquette, value);
                    break;
                case HDF5_Observable::Clover:
                    writeFlowTimeQuantity(*dataSetClover, value);
                    break;
                case HDF5_Observable::TopCharge:
                    writeFlowTimeQuantity(*dataSetTopCharge, value);
                    break;
                case HDF5_Observable::TopChargeImp:
                    writeFlowTimeQuantity(*dataSetTopChargeImp, value);
                    break;
                case HDF5_Observable::EMTE:
                    writeFlowTimeQuantity(*dataSetEMTE, value);
                    break;
            }
        }

        void write4x4Sym(DataSet &dataSet, const std::vector<floatT>& vec4x4SymComponents) {
            // get dataspace of dataset
            DataSpace *fileSpace4x4Sym = new DataSpace(dataSet.getSpace());

            // get rank
            int rank = fileSpace4x4Sym->getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpace4x4Sym->getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[2] = {currentDims[0], 0};
            hsize_t amount[2] = {1, 10};
            hsize_t newsize[2] = {currentDims[0]+1, 10};
            dataSet.extend(newsize);

            // select hyperslab in file
            fileSpace4x4Sym = new DataSpace(dataSet.getSpace());
            fileSpace4x4Sym->selectHyperslab(H5S_SELECT_SET, amount, offset);

            // create memory space
            DataSpace *memorySpace = new DataSpace(2, amount, NULL);

            if (IamRoot()) {
                dataSet.write(vec4x4SymComponents.data(), *hdf5FloatT, *memorySpace, *fileSpace4x4Sym);
            }
        }

        void writeEMTU(const std::vector<floatT>& vecEMTUComponents) {
            write4x4Sym(*dataSetEMTU, vecEMTUComponents);
        }

        void writeEMTCorrData(
            const std::vector<std::vector<COMPLEX(floatT)>>& vecEMTcorrComplex
        ) {
            // create vector of ComplexData instead of COMPLEX(floatT)
            std::vector<std::vector<ComplexData<floatT>>> vecEMTCorrComplexTransformed(10, std::vector<ComplexData<floatT>>(r2max+1));
            for (int i = 0; i < 10; i++) {
                for (int r2 = 0; r2 < r2max + 1; r2++) {
                    vecEMTCorrComplexTransformed[i][r2] = {real(vecEMTcorrComplex[i][r2]), imag(vecEMTcorrComplex[i][r2])};
                }
            }

            // flatten array
            std::vector<ComplexData<floatT>> vecEMTCorrComplexDataTransformedFlat(10 * (r2max + 1));
            for (int i = 0; i < 10; i++) {
                for (int r2 = 0; r2 < r2max + 1; r2++) {
                    vecEMTCorrComplexDataTransformedFlat[i * (r2max + 1) + r2] = vecEMTCorrComplexTransformed[i][r2];
                }
            }

            // get dataspace of dataset
            DataSpace *fileSpaceCorr = new DataSpace(dataSetEMTCorr->getSpace());

            // get rank
            int rank = fileSpaceCorr->getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            fileSpaceCorr->getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[3] = {0, currentDims[1], 0};
            hsize_t amount[3] = {10, 1, currentDims[2]};
            hsize_t newsize[3] = {10, currentDims[1]+1, currentDims[2]};
            dataSetEMTCorr->extend(newsize);
            
            // select hyperslab in file
            fileSpaceCorr = new DataSpace(dataSetEMTCorr->getSpace());
            fileSpaceCorr->selectHyperslab(H5S_SELECT_SET, amount, offset);
            
            // create memory space
            DataSpace *memorySpace = new DataSpace(3, amount, NULL);
            
            if (IamRoot()) {
                // write complex data
                dataSetEMTCorr->write(vecEMTCorrComplexDataTransformedFlat.data(), *compTypeComplex, *memorySpace, *fileSpaceCorr);
            }
        }

};
