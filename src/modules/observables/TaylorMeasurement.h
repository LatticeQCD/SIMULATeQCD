#pragma once

// TODO remove unused imports
#include "../../base/communication/communicationBase.h"
#include "../../base/communication/siteComm.h"
#include "../../base/IO/parameterManagement.h"
#include "../../base/LatticeDimension.h"
#include "../../base/latticeParameters.h"
#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h" // has the C_1000
#include "../inverter/inverter.h"
#include "../HISQ/hisqSmearing.h"
#include <string>

// This sets the maximal derivative order.
// This is used for indexing of the operators using the operator ids.
// Setting this to higher values does NOT result in performance problems.
#define MAX_DERIVATIVE 4 // 8 >= MAX_DERIVATIVE >= 1

#define C_3000 (1/48.)

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif

class TaylorMeasurementParameters : public LatticeParameters {
public:
    DynamicParameter<long> operator_ids;
    DynamicParameter<double> valence_masses;

    Parameter<int> num_random_vectors;
    Parameter <int> seed;

    // Dslash related values
    Parameter <double> mu_f;
    Parameter<double> step_size;
    Parameter<double> residue_meas;
    Parameter<int> cgMax_meas;

    TaylorMeasurementParameters() {
        add(operator_ids, "operator_ids");
        add(valence_masses, "valence_masses");
        add(num_random_vectors, "num_random_vectors");
        addOptional(seed, "seed");

        addDefault(mu_f, "mu0", 0.0);
        //addDefault(residue, "residue", 1e-12);
        //addDefault(cgMax, "cgMax", 10000);
        addDefault(residue_meas, "residue_meas", 1e-12);
        addDefault(cgMax_meas, "cgMax_meas", 20000);

    }
};

// std::vector version (see https://medium.com/the-programming-club-iit-indore/graphs-and-trees-using-c-stl-322e5779eef9)

/*
class DerivativeOperatorTree
{
private:
    std::vector<std::vector<int>> adj;
public:
    DerivativeOperatorTree(int capacity = 10) {
        adj.reserve(capacity);
        adj.push_back(std::vector<int>(MAX_DERIVATIVE));
    }

    void insertOperator(int id) {
        while (id != 0) {
            int link = id % MAX_DERIVATIVE;
            id /= MAX_DERIVATIVE;
        }
    }
};*/

// Memory Management version

/*
class DerivativeOperatorTreeNode
{
public:
    // the following contains pointers to DerivativeOperatorTreeNode classes
    gMemoryPtr<false> nodes[MAX_DERIVATIVE];
    DerivativeOperatorTreeNode() { }

//#define GET_NODE(i) ()
    DerivativeOperatorTreeNode &get(int i) {
        return (DerivativeOperatorTreeNode)*nodes[i];
    }

    bool hasNext() {
        bool has_next = false;
        #pragma unroll
        for (int i = 0; i < MAX_DERIVATIVE; i++)
            has_next |= nodes[i] != nullptr; // TODO how to check if something is a nullptr
        return has_next
    }

    void insertOperator(int id) {
        if (id == 0)
            return;
        int link = id % MAX_DERIVATIVE;
        if (get(link) == nullptr) {
            DerivativeOperatorTreeNode node_stack();
            gMemoryPtr<false> g_mem_ptr = MemoryManagement::getMemAt<false>("DerivativeOperatorTreeNode", sizeof(DerivativeOperatorTreeNode));
            // TODO does this move the data of the stack-allocated class to the heap? Does it get destructed at the end of this?
            *node = node_stack;
            nodes[link] = node; // move assignment operator, this moves the pointer to the heap
        }
        get(link)->insertOperator(id / MAX_DERIVATIVE);
    }
};*/

// normal version

struct DerivativeOperatorMeasurement {
public:
    int operatorId;
    double measurement;

    DerivativeOperatorMeasurement(int operatorId, double measurement) :
        operatorId(operatorId),
        measurement(measurement) { }
};

class DerivativeOperatorTreeNode
{
public:
    // the following contains pointers to DerivativeOperatorTreeNode classes
    DerivativeOperatorTreeNode *nodes[MAX_DERIVATIVE + 1];

    double measurement_akkum = 0.0;

    DerivativeOperatorTreeNode() {
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            nodes[i] = nullptr;
    }

    bool hasNext() {
        bool has_next = false;
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            has_next |= nodes[i] != nullptr;
        return has_next;
    }

    int size() {
        int sum = 0;
        for (int i = 0; i <= MAX_DERIVATIVE; i++)
            sum += nodes[i]->size();
        return sum;
    }

    int depth() {
        int max_depth = 0;
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            int d = nodes[i]->depth();
            if (d > max_depth)
                max_depth = d;
        }
        return max_depth;
    }

    void collectMeasurements(std::vector<DerivativeOperatorMeasurement> &measurements, int divide_by_N, long id = 0) {
        measurements.emplace_back(id, measurement_akkum / divide_by_N);
        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            // The new operator ids are in base 10 (decimal) to make them easy to read.
            // This limits us to a maximal order 8 derivative (which is a 9 in the id)
            long next_id = id * 10 + (i + 1);
            nodes[i]->collectMeasurements(measurements, divide_by_N, next_id);
        }
    }

    void insertOperator(long id) {
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

        // TODO figure out the exact indexing
        int link = (int)(digit - 1);
        assert(link >= 0 && link <= MAX_DERIVATIVE); // TODO maybe not use asserts...
        if (nodes[link] == nullptr) {
            nodes[link] = new DerivativeOperatorTreeNode();
        }
        nodes[link]->insertOperator(id - digit * power);
    }

    ~DerivativeOperatorTreeNode() {
        for (int i = 0; i < MAX_DERIVATIVE; i++)
            delete nodes[i];
    }
};
/*
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct dDdmuFunctor {

    gVect3arrayAcc<floatT> _spinorIn;
    gaugeAccessor<floatT, R18> _gAcc_smeared;
    gaugeAccessor<floatT, U3R14> _gAcc_Naik;
    floatT _c_3000;
    floatT _sign;
    floatT _pow_3;
    floatT _mass;
    int _order;

    template<bool onDevice, size_t NStacks>
    dDdmuFunctor(
            Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &spinorIn,
            Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge_smeared,
            Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gauge_Naik, floatT mass, int order, floatT c_3000) :
        _spinorIn(spinorIn.getAccessor()),
        _gAcc_smeared(gauge_smeared.getAccessor()),
        _gAcc_Naik(gauge_Naik.getAccessor()), _c_3000(c_3000), _mass(_mass)
    {
        _order = order;
        _sign = floatT(pow(-1.0, order));
        _pow_3 = floatT(pow(3.0, order));
    }

    auto getAccessor() const {
        return *this;
    }

    __device__ __host__ gVect3<floatT> operator()(gSiteStack site) const;
};
*/
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class TaylorMeasurement
{
private:
    ConjugateGradient<floatT, NStacks> cg;
    HisqDSlash<floatT, onDevice, LatLayout, HaloDepthGauge, HaloDepthSpin, NStacks> dslash;
    std::vector<Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>> spinors;
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> invert_spinor_field;
    std::vector<double> results;

    CommunicationBase* commBase;

    Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> gauge_Naik;
    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> gauge_smeared;

    Gaugefield<floatT, onDevice, HaloDepthGauge, R18>* gauge;
    const TaylorMeasurementParameters param;
    floatT mass;
    grnd_state<onDevice> d_rand;

    DerivativeOperatorTreeNode tree;

public:
    // TODO use some pointer construct to have the pointers to these big data thingies in this class for all methods
    // TODO use weak pointers! I want to be able to access, but not hold.
    // but for now... use normal pointers, as that is much simpler
    TaylorMeasurement(CommunicationBase &commBase, const TaylorMeasurementParameters &param, floatT mass,
                      Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge,
                      grnd_state<onDevice> &d_rand) :
        commBase(&commBase),
        gauge(&gauge),
        gauge_Naik(commBase, "SHARED_GAUGENAIK"),
        gauge_smeared(commBase, "SHARED_GAUGELVL2"),
        dslash(gauge_smeared, gauge_Naik, mass),
        invert_spinor_field(commBase),
        d_rand(d_rand),
        param(param) {

    }

    void insertOperator(long operatorId) {
        tree.insertOperator(operatorId);
    }

    void computeOperators() {
        HisqSmearing<floatT, onDevice, HaloDepthGauge, R18, R18, R18, U3R14> smearing(*gauge, gauge_smeared, gauge_Naik);
        smearing.SmearAll(param.mu_f());

        int max_depth = tree.depth();

        // array of Spinorfields
        // TODO why is there an NStacks in the spinor field??? is that already an array of spinorfields?
        spinors.clear();
        spinors.reserve(max_depth + 1);
        for (int i = 0; i <= max_depth; i++)
            spinors.emplace_back(*commBase);

        // now compute my operators
        const int num_random_vectors = param.num_random_vectors();
        for (int i = 0; i < num_random_vectors; i++) {
            spinors[0].gauss(d_rand.state);
            // TODO spinors[0] /= sqrt(spinors[0].dotProduct(spinors[0]));
            recursive_apply_operator_tree(tree, 0);
        }
    }

    void collectResults(std::vector<DerivativeOperatorMeasurement> &results) {
        results.clear();
        results.reserve(tree.size());
        tree.collectMeasurements(results, param.num_random_vectors() * 4); // 4 dimensions
    }

private:
    void recursive_apply_operator_tree(DerivativeOperatorTreeNode &operatorNode, int depth) {
        // check if there are next nodes in the operator tree
        if (!operatorNode.hasNext())
            return;

        Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &spinorIn = spinors[depth];
        Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> &spinorOut = spinors[depth + 1];

        // invert for each tree node that gets traversed
        // TODO I need an invert in place in spinorIn, so I don't need this additional spinorfield in memory
        cg.invert_new(dslash, invert_spinor_field, spinorIn, param.cgMax_meas(), param.residue_meas());
        // TODO figure out if I can disable this sometimes to gain something??
        invert_spinor_field.updateAll();

        for (int i = 0; i <= MAX_DERIVATIVE; i++) {
            if (!operatorNode.nodes[i])
                continue;

            // now do the dDdmu
            const int order = i;
            if (order > 0) {
                //spinorOut.template iterateOverBulk<BLOCKSIZE>(dDdmuFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(invert_spinor_field, gauge_smeared, gauge_Naik, mass, order, C_3000));
                //spinorOut.updateAll();
            }

            // output the current measurement data into the output akkumulation
            operatorNode.nodes[i]->measurement_akkum += 0.5 * spinorIn.realdotProduct(spinorIn);

            recursive_apply_operator_tree(*(operatorNode.nodes[i]), depth + 1);
        }
    }
};
