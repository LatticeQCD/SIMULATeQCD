//
// hdf5fileWriter.h
//
// File for writing gradient flow data to HDF5 files
//

#include "H5Cpp.h"
#include "../communication/communicationBase.h"
#include "../latticeParameters.h"
#include "../../modules/tensor_decomposition/tensorDecomposition.h"

using namespace H5;


template<class floatT>
struct ComplexData {
    floatT real;
    floatT imag;
};


class CommunicationBase;
class LatticeParameters;


template<class floatT>
class HDF5FileWriter {
    private:
        CommunicationBase& _commBase;
        LatticeParameters& _latParams;

        H5File* _file;
        const H5std_string _fileName;
        Group* group_gradFlowMeasurements;
        const H5std_string group_name_gradFlowMeasurements;
        Group* group_EMT_corr;
        const H5std_string group_name_EMT_corr;

        hsize_t r2max;
        
        hsize_t initDims_flowTimeQuantity[1];
        hsize_t initDims_r2Counts[1];
        hsize_t initDims_corr[3];
        hsize_t maxDims_flowTimeQuantity[1];
        hsize_t maxDims_r2Counts[1];
        hsize_t maxDims_corr[3];
        hsize_t chunkSize_flowTimeQuantity[1];
        hsize_t chunkSize_corr[3];
        
        const H5::DataType* hdf5floatT = nullptr;
        CompType *compTypeComplex;

        DataSpace *dataSpace_flowTimeQuantity;
        DataSpace *dataSpace_r2Counts;
        DataSpace *dataSpace_corr;
        DataSpace *dataSpace_LL;

        const H5std_string dataSet_name_flowTime;
        const H5std_string dataSet_name_flowTimeNecessary;
        const H5std_string dataSet_name_plaquette, dataSet_name_topCharge;
        const H5std_string dataSet_name_r2Counts;
        const H5std_string dataSet_name_corr;
        const H5std_string dataSet_name_LL;

        DataSet *dataSet_flowTime;
        DataSet *dataSet_flowTimeNecessary;
        DataSet *dataSet_plaquette, *dataSet_topCharge;
        DataSet *dataSet_r2Counts;
        DataSet *dataSet_corr;
        DataSet *dataSet_LL;

    public:
        // standard constructor
        HDF5FileWriter(CommunicationBase& commBase, LatticeParameters& latParams, const std::string& fileName) :
            _commBase(commBase), _latParams(latParams), _fileName(fileName),
            group_name_gradFlowMeasurements("/GradFlow_Measurements"),
            group_name_EMT_corr("EMTU_Correlator"),
            dataSet_name_flowTime("flow_time"), dataSet_name_flowTimeNecessary("flow_time_necessary"),
            dataSet_name_plaquette("plaquette"), dataSet_name_topCharge("topological_charge"),
            dataSet_name_r2Counts("r2_counts"), dataSet_name_corr("G"), dataSet_name_LL("G_LL") {
            
            // fix PredType based on floatT
            if constexpr (std::is_same_v<floatT, double>) {
                hdf5floatT = &PredType::NATIVE_DOUBLE;
            } else if constexpr (std::is_same_v<floatT, float>) {
                hdf5floatT = &PredType::NATIVE_FLOAT;
            }

            r2max = TensorDecomposition<floatT, 0>::get_r2max();

            // create the HDF5 file
            _file = new H5File(_fileName, H5F_ACC_TRUNC);

            // create groups
            group_gradFlowMeasurements = new Group(_file->createGroup(group_name_gradFlowMeasurements));
            group_EMT_corr = new Group(group_gradFlowMeasurements->createGroup(group_name_EMT_corr));

            // set initial dimension to zero for flow time, r2max+1 for separations
            initDims_flowTimeQuantity[0] = 0;
            initDims_r2Counts[0] = r2max + 1;
            initDims_corr[0] = 5;
            initDims_corr[1] = 0;
            initDims_corr[2] = r2max + 1;

            // set maximum dimensions, unlimited for flow time, r2max+1 for separations
            maxDims_flowTimeQuantity[0] = H5S_UNLIMITED;
            maxDims_r2Counts[0] = r2max + 1;
            maxDims_corr[0] = 5;
            maxDims_corr[1] = H5S_UNLIMITED;
            maxDims_corr[2] = r2max + 1;

            // set chunk size, one for flow time, r2max+1 for separations
            chunkSize_flowTimeQuantity[0] = 1;
            chunkSize_corr[0] = 1;
            chunkSize_corr[1] = 1;
            chunkSize_corr[2] = r2max + 1;

            // create dataSpaces
            dataSpace_flowTimeQuantity = new DataSpace(1, initDims_flowTimeQuantity, maxDims_flowTimeQuantity);
            dataSpace_r2Counts = new DataSpace(1, initDims_r2Counts, maxDims_r2Counts);
            dataSpace_corr = new DataSpace(3, initDims_corr, maxDims_corr);
            dataSpace_LL = new DataSpace(1, initDims_r2Counts, maxDims_r2Counts);
            
            // create dataset property list and set the chunking
            DSetCreatPropList propList_flowTimeQuantity;
            propList_flowTimeQuantity.setChunk(1, chunkSize_flowTimeQuantity);
            DSetCreatPropList propList_corr;
            propList_corr.setChunk(3, chunkSize_corr);

            // create compound data type for storing complex numbers
            compTypeComplex = new CompType(sizeof(ComplexData<floatT>));
            compTypeComplex->insertMember("real", HOFFSET(ComplexData<floatT>, real), *hdf5floatT);
            compTypeComplex->insertMember("imag", HOFFSET(ComplexData<floatT>, imag), *hdf5floatT);

            // create datasets
            dataSet_flowTime = new DataSet(group_gradFlowMeasurements->createDataSet(dataSet_name_flowTime, *hdf5floatT, *dataSpace_flowTimeQuantity, propList_flowTimeQuantity));
            dataSet_flowTimeNecessary = new DataSet(group_EMT_corr->createDataSet(dataSet_name_flowTimeNecessary, *hdf5floatT, *dataSpace_flowTimeQuantity, propList_flowTimeQuantity));
            dataSet_plaquette = new DataSet(group_gradFlowMeasurements->createDataSet(dataSet_name_plaquette, *hdf5floatT, *dataSpace_flowTimeQuantity, propList_flowTimeQuantity));
            dataSet_topCharge = new DataSet(group_gradFlowMeasurements->createDataSet(dataSet_name_topCharge, *hdf5floatT, *dataSpace_flowTimeQuantity, propList_flowTimeQuantity));
            dataSet_r2Counts = new DataSet(group_EMT_corr->createDataSet(dataSet_name_r2Counts, PredType::NATIVE_INT, *dataSpace_r2Counts));
            dataSet_corr = new DataSet(group_EMT_corr->createDataSet(dataSet_name_corr, *compTypeComplex, *dataSpace_corr, propList_corr));
            dataSet_LL = new DataSet(group_EMT_corr->createDataSet(dataSet_name_LL, *compTypeComplex, *dataSpace_LL));

        }

        bool IamRoot() {
            return _commBase.IamRoot();
        }

        void writeFlowTimeQuantity(DataSet &dataSet, const floatT flowTimeQuantity) {
            // get dataspace of dataset
            DataSpace *filespace_flowTimeQuantity = new DataSpace(dataSet.getSpace());

            // get rank
            int rank = filespace_flowTimeQuantity->getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            filespace_flowTimeQuantity->getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[1] = {currentDims[0]};
            hsize_t amount[1] = {1};
            hsize_t newsize[1] = {currentDims[0]+1};
            dataSet.extend(newsize);

            // select hyperslab in file
            filespace_flowTimeQuantity = new DataSpace(dataSet.getSpace());
            filespace_flowTimeQuantity->selectHyperslab(H5S_SELECT_SET, amount, offset);

            // create memory space
            DataSpace *memoryspace = new DataSpace(1, amount, NULL);

            if (IamRoot()) {
                dataSet.write(&flowTimeQuantity, *hdf5floatT, *memoryspace, *filespace_flowTimeQuantity);
            }
        }

        void writeFlowTime(const floatT flow_time) {
            writeFlowTimeQuantity(*dataSet_flowTime, flow_time);
        }

        void writeFlowTimeNecessary(const floatT flow_time_necessary) {
            writeFlowTimeQuantity(*dataSet_flowTimeNecessary, flow_time_necessary);
        }

        void writePlaquette(const floatT plaquette) {
            writeFlowTimeQuantity(*dataSet_plaquette, plaquette);
        }

        void writeTopologicalCharge(const floatT topological_charge) {
            writeFlowTimeQuantity(*dataSet_topCharge, topological_charge);
        }

        void writeR2Counts(const std::vector<int>& vec_counts) {
            if (IamRoot()) {
                dataSet_r2Counts->write(vec_counts.data(), PredType::NATIVE_INT);
            }
        }

        void writeLL(const std::vector<COMPLEX(floatT)>& vec_LL_COMPLEX) {
            if (IamRoot()) {
                dataSet_LL->write(vec_LL_COMPLEX.data(), *compTypeComplex);
            }
        }

        void writeEMTUCorrData(
            const std::vector<std::vector<COMPLEX(floatT)>>& vec_EMTU_corr_COMPLEX
        ) {
            // create vector of ComplexData instead of COMPLEX(floatT)
            std::vector<std::vector<ComplexData<floatT>>> vec_EMTU_LL_complex_data(5, std::vector<ComplexData<floatT>>(r2max+1));
            for (int i = 0; i < 5; i++) {
                for (int r2 = 0; r2 < r2max + 1; r2++) {
                    vec_EMTU_LL_complex_data[i][r2] = {real(vec_EMTU_corr_COMPLEX[i][r2]), imag(vec_EMTU_corr_COMPLEX[i][r2])};
                }
            }

            // flatten array
            std::vector<ComplexData<floatT>> vec_EMTU_LL_complex_data_flat(5 * (r2max + 1));
            for (int i = 0; i < 5; i++) {
                for (int r2 = 0; r2 < r2max + 1; r2++) {
                    vec_EMTU_LL_complex_data_flat[i * (r2max + 1) + r2] = vec_EMTU_LL_complex_data[i][r2];
                }
            }

            // get dataspace of dataset
            DataSpace *filespace_corr = new DataSpace(dataSet_corr->getSpace());

            // get rank
            int rank = filespace_corr->getSimpleExtentNdims();

            // get current dimensions
            std::vector<hsize_t> currentDims(rank);
            filespace_corr->getSimpleExtentDims(currentDims.data(), NULL);

            // set offset, amount and new size
            hsize_t offset[3] = {0, currentDims[1], 0};
            hsize_t amount[3] = {5, 1, currentDims[2]};
            hsize_t newsize[3] = {5, currentDims[1]+1, currentDims[2]};
            dataSet_corr->extend(newsize);
            
            // select hyperslab in file
            filespace_corr = new DataSpace(dataSet_corr->getSpace());
            filespace_corr->selectHyperslab(H5S_SELECT_SET, amount, offset);
            
            // create memory space
            DataSpace *memoryspace = new DataSpace(3, amount, NULL);
            
            if (IamRoot()) {
                // write complex data
                dataSet_corr->write(vec_EMTU_LL_complex_data_flat.data(), *compTypeComplex, *memoryspace, *filespace_corr);
            }
        }

};
