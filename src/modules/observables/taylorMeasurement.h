// #pragma once

#include "../../base/communication/siteComm.h"
#include "../../base/latticeParameters.h"
#include "../../base/IO/fileWriter.h"
#include "../../gauge/gaugefield.h"
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h"
#include "./dslashDerivative.h"
#include "../hisq/hisqSmearing.h"
#include <sstream>

// This sets the maximal derivative order.
// This is used for indexing of the operators using the operator ids.
// Setting this to higher values does NOT result in performance problems.
#define MAX_DERIVATIVE 4 // 8 >= MAX_DERIVATIVE >= 1

class TaylorMeasurementParameters : public LatticeParameters {
public:
    DynamicParameter<long> operator_ids;
    DynamicParameter<double> valence_masses;

    Parameter<int> num_random_vectors;
    Parameter<int> num_toread_vectors;
    Parameter<int> seed;

    // Dslash related values
    Parameter<double> mu_f;
    Parameter<bool> use_naik_epsilon;
    Parameter<double> residue;
    Parameter<int> cgMax;

    Parameter<std::string> eigen_file;
    Parameter<std::string> output_file;
    Parameter<std::string> collected_output_file;

    TaylorMeasurementParameters() {
        add(operator_ids, "operator_ids");
        add(valence_masses, "valence_masses");
        add(num_random_vectors, "num_random_vectors");
        add(num_toread_vectors, "num_toread_vectors");
        addDefault(seed, "seed", 0);
        addOptional(eigen_file, "eigen_file");
        addOptional(output_file, "output_file");
        addOptional(collected_output_file, "collected_output_file");

        addDefault(mu_f, "mu0", 0.0);
        add(use_naik_epsilon, "use_naik_epsilon"); // No default to make this very clear in usage!
        addDefault(residue, "residue", 1e-12);
        addDefault(cgMax, "cgMax", 20000);
    }
};

// std::vector version (see https://medium.com/the-programming-club-iit-indore/graphs-and-trees-using-c-stl-322e5779eef9)

struct DerivativeOperatorMeasurement {
    const long operatorId;
    const COMPLEX(double) measurement;
    const COMPLEX(double) std;

    DerivativeOperatorMeasurement(long operatorId, COMPLEX(double) measurement, COMPLEX(double) std) :
        operatorId(operatorId),
        measurement(measurement),
        std(std) { }
};

class DerivativeOperatorTreeNode
{
public:
    // the following contains pointers to DerivativeOperatorTreeNode classes
    DerivativeOperatorTreeNode *nodes[MAX_DERIVATIVE + 1];

    COMPLEX(double) measurement_accum = 0.0;
    COMPLEX(double) measurement_sqr_accum = 0.0;

    DerivativeOperatorTreeNode() {
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            nodes[i] = nullptr;
    }

    bool hasNext() const {
        bool has_next = false;
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            if (nodes[i])
                has_next |= nodes[i] != nullptr;
        return has_next;
    }

    int size() const {
        int sum = 1;
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            if (nodes[i])
                sum += nodes[i]->size();
        return sum;
    }

    int depth() const {
        int max_depth = 0;
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            if (nodes[i]) {
                int d = nodes[i]->depth() + 1;
                if (d > max_depth)
                    max_depth = d;
            }
        }
        return max_depth;
    }

    int leaf_count() const {
        int leaves = 0;
        int is_leaf = 1;
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            if (nodes[i]) {
                leaves += nodes[i]->leaf_count();
                is_leaf = 0;
            }
        }
        return leaves + is_leaf;
    }

    /**
     * @brief collect the computed measurements from the tree into a list
     * @param measurements - an empty list to be filled with the measurements
     * @param samples - the number of accumulated measurements
     * @param normalization - the normalisation factor, the result is divided by this
     * @param id - the operator id of this node
     */
    void collectMeasurements(std::vector<DerivativeOperatorMeasurement> &measurements, const int samples, const double normalization, const long id = 0) const {
        COMPLEX(double) meas = (measurement_accum / samples) / normalization;
        COMPLEX(double) meas_sqr = (measurement_sqr_accum / samples) / (normalization * normalization);
        // standard deviation of the mean is sqrt(sigma^2 / N)
        COMPLEX(double) std = COMPLEX(double)(sqrt(std::max(0.0, real(meas_sqr) - real(meas)*real(meas)) / (samples - 1)), sqrt(std::max(0.0, imag(meas_sqr) - imag(meas)*imag(meas)) / (samples - 1)));
        measurements.emplace_back(id, meas, std);
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            // The new operator ids are in base 10 (decimal) to make them easy to read.
            // This limits us to a maximal order 8 derivative (which is a 9 in the id)
            if (nodes[i]) {
                const long next_id = id * 10 + (i + 1);
                nodes[i]->collectMeasurements(measurements, samples, normalization, next_id);
            }
        }
    }

    void insertOperator(const long id) {
        if (id <= 0)
            return;
        // get the most significant decimal digit -> that represents the first operator
        long power = 1;
        long digit = id;
        // The new operator ids are in base 10 (decimal) to make them easy to read.
        // This limits us to a maximal order 8 derivative (which is a 9 in the id)
        while (digit >= 10) {
            digit /= 10;
            power *= 10;
        }

        int link = (int)(digit - 1);
        if (link < 0 || link > 9)
            throw std::runtime_error(stdLogger.fatal("Operator ID: ", id, " is invalid"));
        if (nodes[link] == nullptr) {
            nodes[link] = new DerivativeOperatorTreeNode();
        }
        nodes[link]->insertOperator(id - digit * power);
    }

    ~DerivativeOperatorTreeNode() {
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            delete nodes[i];
    }
};

template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class TaylorMeasurement
{
private:
    std::vector<SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks>> spinors;
    SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks> random_vector;

    Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik;
    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared;

    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge;

    eigenpairs<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks> &eigen;

    const TaylorMeasurementParameters param;
    const floatT mass;
    const floatT naik_epsilon;
    grnd_state<onDevice> rand;

    HisqDSlashInverse<floatT, onDevice, HaloDepthGauge, HaloDepthSpin, NStacks> dslash_inv;

    DerivativeOperatorTreeNode tree;
    int tree_operator_accum_counter = 0;

    FileWriter output_file;
    bool use_output;

public:
    TaylorMeasurement(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge,
                    eigenpairs<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks> &eigen,
                    const TaylorMeasurementParameters &param, const floatT mass, const bool use_naik_epsilon,
                    grnd_state<onDevice> &rand) :
        gauge_Naik(gauge.getComm()),
        gauge_smeared(gauge.getComm()),
        mass(mass),
        random_vector(gauge.getComm()),
        rand(rand),
        param(param),
        gauge(gauge),
        eigen(eigen),
        naik_epsilon(use_naik_epsilon ? get_naik_epsilon_from_amc(mass) : 0.0),
        dslash_inv(gauge_smeared, gauge_Naik, mass, naik_epsilon),
        output_file(gauge.getComm(), param),
        use_output(false) { }

    /**
     * @brief insert an operator of the form tr(D^-1 D_i D^-1 D_j) into the computation tree
     * @param id - the id of the operator which for above example would be (i+1,j+1)_10
     * such that the ids are well human readble readable.
     * The Operator tr(D^-1 D^-1) would be 11
     * The Operator tr(D^-1 dD/dµ) would be 2
     */
    void insertOperator(long operatorId) {
        tree.insertOperator(operatorId);
    }

    void computeOperators() {
        {
            HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik, naik_epsilon);
            smearing.SmearAll(param.mu_f());
        } // at this point the two temporary gauge fields in HisqSmearing will be freed again

        int max_depth = tree.depth();
        rootLogger.info("Depth: ", max_depth);
        int inversions = tree.size() - tree.leaf_count();
        rootLogger.info("Inversions per Vector: ", inversions);

        // array of Spinorfields
        spinors.clear();
        spinors.reserve(max_depth + 1);
        for (int i = 0; i <= max_depth; i++)
            spinors.emplace_back(gauge.getComm());

        // now compute my operators
        const int num_random_vectors = (param.num_random_vectors() + NStacks - 1) / NStacks; // = ceil(n / NStacks)
        const double gpu_count = param.nodeDim[0] * param.nodeDim[1] * param.nodeDim[2] * param.nodeDim[3]; // scaling is not optimal, but optimal scaling is assumed here
        const double base_factor = (double)GIndexer<All, HaloDepthGauge>::getLatData().globvol4 * (inversions * NStacks * num_random_vectors) / gpu_count;
        const double cg_factor = -log(param.residue() / 2.0) / (mass + 1e-6);
        rootLogger.info("Estimated Computation Time (V100): ", (int)(base_factor * cg_factor * 0.876e-9 / 60 + 1), "min");
        StopWatch<true> timer;
        timer.start();
        for (int i = 0; i < num_random_vectors; i++) {
            random_vector.gauss(rand.state);
            spinors[0] = random_vector;

            if (use_output) {
                // NOTE: the following code is soo difficult just because LineFormatter does not wait
                // with writing to the file until endLine(), but does it immediately, so I can't simply
                // have multiple LineFormatters running simultaneously.
                std::vector<std::ostringstream> streams;
                std::vector<LineFormatter> tags;
                streams.reserve(NStacks);
                tags.reserve(NStacks);
                for (int i = 0; i < NStacks; i++) {
                    streams.emplace_back();
                    tags.emplace_back(streams[i] << "", std::string(""));
                    tags[i] << std::to_string(mass) + " ";
                    tags[i] << std::to_string(tree_operator_accum_counter + i) + " ";
                }

                recursive_apply_operator_tree(&tags, tree, 0);

                // transfer all the temporary LineFormatters into the file
                for (int i = 0; i < NStacks; i++) {
                    LineFormatter line = output_file.tag("");
                    line << streams[i].str();
                }
            }
            else {
                // Run the operator tree without worrying about single measurement output
                recursive_apply_operator_tree(nullptr, tree, 0);
            }
            tree_operator_accum_counter += NStacks;
            rootLogger.info(((i + 1) * 100) / num_random_vectors, "%");
        }
        rootLogger.info("Time taken: ", timer.stop()/1000.0, "s");
    }

    void collectResults(std::vector<DerivativeOperatorMeasurement> &results) {
        double norm = GIndexer<All, HaloDepthGauge>::getLatData().globvol4 * 3;
        results.clear();
        results.reserve(tree.size());
        tree.collectMeasurements(results, tree_operator_accum_counter, norm);

        if (param.collected_output_file.isSet()) {
            // write these collected measurements into a file
            FileWriter collected_output_file(gauge.getComm(), param);
            collected_output_file.createFile(std::to_string(mass) + "_" + param.collected_output_file());
            LineFormatter header = collected_output_file.header();
            header << "ID   ";
            header << "N Random Vectors   ";
            header << "tr(Op) [real, imaginary]   ";
            header << "std(tr(Op)) [real, imaginary]";
            // NOTE that tr(Op)^2 would be estimated in an unbiased way by tr(Op)^2 - Var(tr(Op))
            // So this kind of output is very useful
            for (DerivativeOperatorMeasurement &meas : results) {
                LineFormatter tag = collected_output_file.tag("");
                tag << meas.operatorId << " ";
                tag << tree_operator_accum_counter << " ";
                tag << real(meas.measurement) << " ";
                tag << imag(meas.measurement) << " ";
                tag << real(meas.std) << " ";
                tag << imag(meas.std);
                tag.endLine();
            }
        }
    }

    void write_output_file_header() {
        use_output = param.output_file.isSet();
        if (!use_output)
            return;
        output_file.createFile(std::to_string(mass) + "_" + param.output_file());
        LineFormatter header = output_file.header();
        header << "Mass ";
        header << "Random Vector ";
        write_output_file_header(header, tree, 0L);
        header.endLine();
    }

private:
    void write_output_file_header(LineFormatter &header, DerivativeOperatorTreeNode &operatorNode, long id) {
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            if (operatorNode.nodes[i] == nullptr)
                continue;
            const long next_id = id * 10 + (i + 1);
            header << "Operator " + std::to_string(next_id) + " ";
            write_output_file_header(header, *operatorNode.nodes[i], next_id);
        }
    }

    void recursive_apply_operator_tree(std::vector<LineFormatter> *tags, DerivativeOperatorTreeNode &operatorNode, int depth) {
        // check if there are next nodes in the operator tree
        if (!operatorNode.hasNext())
            return;

        SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks> &spinorIn = spinors[depth];
        SpinorfieldAll<floatT, onDevice, HaloDepthSpin, NStacks> &spinorOut = spinors[depth + 1];

        // invert once for each tree node that gets traversed
        // cg uses dslash.applyMdaggM as the positive semidefinite matrix, so I need to solve MdaggM * x = Mdagg * b =? -M * b
        // however the inversion can be done more efficient by exploiting the odd and even structure of the matrix.

        //rootLogger.info("CG started");
        dslash_inv.apply_Dslash_inverse_deflation(spinorOut, spinorIn, eigen, param.cgMax(), param.residue());
        spinorIn = spinorOut; // spinorIn is the node with the inverse applied from here on

        // MemoryManagement::memorySummary();

        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            if (operatorNode.nodes[i] == nullptr)
                continue;
            DerivativeOperatorTreeNode &node = *operatorNode.nodes[i];

            // now do the dDdmu
            const int order = i;
            if (order > 0) {
                spinorOut.odd.template iterateOverBulk<>(dDdmuFunctor<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks>(spinorIn.even, gauge_smeared, gauge_Naik, order, naik_epsilon));
                spinorOut.even.template iterateOverBulk<>(dDdmuFunctor<floatT, onDevice, Odd, HaloDepthGauge, HaloDepthSpin, NStacks>(spinorIn.odd, gauge_smeared, gauge_Naik, order, naik_epsilon));
                spinorOut.updateAll();
            }
            else {
                // do nothing here as spinorOut is already correctly set to spinorIn
            }
            //rootLogger.info("Operator ", i, " computed successful");

            // output the current measurement data into the output accumulation
            SimpleArray<COMPLEX(double), NStacks> tr(0);
            tr = spinorOut.dotProductStacked(random_vector);
            // [Rank 2] A GPU error occured: performFunctor: Failed to launch kernel: too many resources requested for launch ( cudaErrorLaunchOutOfResources )
            //rootLogger.info("Dot Product ", i, " computed successful");

            if (tags) {
                double norm = GIndexer<All, HaloDepthGauge>::getLatData().globvol4 * 3;
                for (int i = 0; i < NStacks; i++){
                    // output to the current line in the output file
                    (*tags)[i] << (tr[i] / norm) << " ";
                }
            }

            node.measurement_accum += tr.sum();
            for (size_t i = 0; i < NStacks; i++) {
                tr[i] = COMPLEX(double)(real(tr[i]) * real(tr[i]), imag(tr[i]) * imag(tr[i]));
            }
            node.measurement_sqr_accum += tr.sum();

            recursive_apply_operator_tree(tags, node, depth + 1);
        }
    }
};
