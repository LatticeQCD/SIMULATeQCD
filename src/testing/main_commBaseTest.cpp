#include "../base/communication/communicationBase.h"
#include "../base/memoryManagement.h"


int main(int argc, char* argv[]) {

    rootLogger.setVerbosity(TRACE);
    const LatticeDimenions Dim(8,8,8,8);
    
    rootLogger.info("Call CommunicationBase constructor");
    CommunicationBase commbase(&argc,&argv);
    rootLogger.info("Initialize commBase");
    commbase.init(Dim);
    

    //Test all public member functions:
    bool gpuAware = commbase.gpuAwareMPIAvail();
    rootLogger.info("gpuAwareMPIAvail(): ", gpuAware);

    bool useP2P = commbase.useGpuP2P();
    rootLogger.info("useGpuP2P(): ", useP2P);

    rootLogger.info("getNeighborInfo() ");
    NeighborInfo ninfo = commbase.getNeighborInfo();

    const LatticeDimensions thesecoords = commbase.mycoords();
    const LatticeDimensions thesenodes = commbase.nodes();
    rootLogger.info("mycoords(): ", thesecoords);
    rootLogger.info("nodes(): ", thesenodes);

    bool amroot = commbase.IamRoot();
    rootLogger.info("IamRoot(): ", amroot);

    int myrank = commbase.MyRank();
    stdLogger.info("MyRank(): ", myrank); //does std logger print from every rank? 

    bool fhalos = commbase.forceHalos();
    rootLogger.info("forceHalos(): ", fhalos);

    int grank = commbase.getRank(Dim); //get some rank?
    rootLogger.info("getRank(): ", grank);

    int nproc = commbase.getNumberProcesses();
    rootLogger.info("getNumberProcesses(): ", nproc);

    MPI_Comm ccomm = commbase.getCart_comm();
    rootLogger.info("getCart_comm(): ", ccomm);



    //TODO:
    //test root2all for all types

    //test reduce for all types

    //test reduceMax for all types

    //test globalAverage for all types

    //test globalMinimum for all types

    //test globalAverage for all types

    //test globalBarrier()

    //test nodeBarrier()

    //test updateAll()

    //test initIOBinary

    //test writeBinary

    //test readBinary

    //test closeIOBinary



    return 0;
}

