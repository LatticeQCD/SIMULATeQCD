#include "../SIMULATeQCD.h"

struct CheckParams : LatticeParameters {
    Parameter<std::string> Randfile;

    CheckParams() {
        add(Randfile, "Randfile");
    }
};

void CheckRand(CommunicationBase &commBase, const std::string& rand_file){
    grnd_state<false> h_rand;
    h_rand.read_from_file(rand_file, commBase);
}


int main(int argc, char *argv[]) {

    try {
        stdLogger.setVerbosity(INFO);
        const size_t HaloDepth = 0;

        CheckParams param;
        param.nodeDim.set({1,1,1,1});
        CommunicationBase commBase(&argc, &argv);

        param.readfile(commBase, "../parameter/applications/CheckRand.param", argc, argv);
        commBase.init(param.nodeDim());
        initIndexer(HaloDepth, param, commBase);

        CheckRand(commBase, param.Randfile());
    }
    catch (const std::runtime_error &error) {
        return 1;
    }
    rootLogger.info("Random state seems to be fine.");
    return 0;
}
