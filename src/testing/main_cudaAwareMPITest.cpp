/* 
 * main_cudaAwareMPITest.cpp                                                               
 * 
 * L. Mazur
 * 
 */

#include "../SIMULATeQCD.h"

void allocDevice(char **_memoryChunk, size_t size) {
    gpuError_t gpuErr = gpuMalloc((void **) _memoryChunk, size);
    if (gpuErr) GpuError("_memoryChunk: Failed to allocate memory on device", gpuErr);
}

void freeDevice(char *_memoryChunk) {
    gpuError_t gpuErr = gpuFree(_memoryChunk);
    if (gpuErr) {
        GpuError("_memoryChunk: Failed to free memory on host", gpuErr);
    }
}

void allocHost(char **_memoryChunk, size_t size) {
    gpuError_t gpuErr = gpuMallocHost((void **) _memoryChunk, size);
    if (gpuErr) GpuError("_memoryChunk: Failed to allocate memory on device", gpuErr);
}

void freeHost(char *_memoryChunk) {
    gpuError_t gpuErr = gpuFreeHost(_memoryChunk);
    if (gpuErr) {
        GpuError("_memoryChunk: Failed to free memory on host", gpuErr);
    }
}

void copyHostToDevice(char *_memoryChunkHost, char *_memoryChunkDevice, size_t size) {
    gpuError_t gpuErr = gpuMemcpy(_memoryChunkDevice, _memoryChunkHost, size, gpuMemcpyHostToDevice);
    if (gpuErr) GpuError("Failed to copy data H2D", gpuErr);
}

void copyDeviceToHost(char *_memoryChunkDevice, char *_memoryChunkHost, size_t size) {
    gpuError_t gpuErr = gpuMemcpy(_memoryChunkHost, _memoryChunkDevice, size, gpuMemcpyDeviceToHost);
    if (gpuErr)
        GpuError("Failed to copy data D2H", gpuErr);
}

// This part simulates the communication of corner parts of a halo update.
// This should reproduce a bug in the current (08.06.2020) cuda-aware openmpi installation on the Bielefeld cluster.
// There are also unnecessary code parts ...
void run1(CommunicationBase &commBase) {
    char *sendbuf, *recvbuf, *gaugefield;
    char *hostbufsend, *hostbufrecv;
    MPI_Request req[80];

    rootLogger.info("");
    rootLogger.info("=====================");
    rootLogger.info("======= START =======");
    rootLogger.info("=====================\n");

    rootLogger.info("Allocate Memory...");

    allocHost(&hostbufsend, 65396737 + 1);
    allocHost(&hostbufrecv, 65396737 + 1);
    allocDevice(&sendbuf, 65396737 + 1);
    allocDevice(&recvbuf, 65396737 + 1);
    allocDevice(&gaugefield, 111476736);

    MPI_Comm cart_comm = commBase.getCart_comm();

    rootLogger.info("Memory allocated. Start communication...");

    rootLogger.info("comm 0");
    MPI_Isend(sendbuf + 65249280, 9216, MPI_CHAR, !commBase.MyRank(), 79, cart_comm, &req[64]);
    MPI_Irecv(recvbuf + 65249280, 9216, MPI_CHAR, !commBase.MyRank(), 64, cart_comm, &req[64]);

    rootLogger.info("comm 1");
    MPI_Isend(sendbuf + 65258496, 9216, MPI_CHAR, !commBase.MyRank(), 78, cart_comm, &req[66]);
    MPI_Irecv(recvbuf + 65258496, 9216, MPI_CHAR, !commBase.MyRank(), 65, cart_comm, &req[66]);

    rootLogger.info("comm 2");
    MPI_Isend(sendbuf + 65267712, 9216, MPI_CHAR, !commBase.MyRank(), 77, cart_comm, &req[68]);
    MPI_Irecv(recvbuf + 65267712, 9216, MPI_CHAR, !commBase.MyRank(), 66, cart_comm, &req[68]);

    rootLogger.info("comm 3");
    MPI_Isend(sendbuf + 65276928, 9216, MPI_CHAR, !commBase.MyRank(), 76, cart_comm, &req[70]);
    MPI_Irecv(recvbuf + 65276928, 9216, MPI_CHAR, !commBase.MyRank(), 67, cart_comm, &req[70]);

    rootLogger.info("comm 4");
    MPI_Isend(sendbuf + 65286144, 9216, MPI_CHAR, !commBase.MyRank(), 75, cart_comm, &req[72]);
    MPI_Irecv(recvbuf + 65286144, 9216, MPI_CHAR, !commBase.MyRank(), 68, cart_comm, &req[72]);

    rootLogger.info("comm 5");
    MPI_Isend(sendbuf + 65295360, 9216, MPI_CHAR, !commBase.MyRank(), 74, cart_comm, &req[74]);
    MPI_Irecv(recvbuf + 65295360, 9216, MPI_CHAR, !commBase.MyRank(), 69, cart_comm, &req[74]);

    rootLogger.info("comm 6");
    MPI_Isend(sendbuf + 65304576, 9216, MPI_CHAR, !commBase.MyRank(), 73, cart_comm, &req[76]);
    MPI_Irecv(recvbuf + 65304576, 9216, MPI_CHAR, !commBase.MyRank(), 70, cart_comm, &req[76]);

    rootLogger.info("comm 7");
    MPI_Isend(sendbuf + 65313792, 9216, MPI_CHAR, !commBase.MyRank(), 72, cart_comm, &req[78]);
    MPI_Irecv(recvbuf + 65313792, 9216, MPI_CHAR, !commBase.MyRank(), 71, cart_comm, &req[78]);

    rootLogger.info("comm 8");
    MPI_Isend(sendbuf + 65323008, 9216, MPI_CHAR, !commBase.MyRank(), 71, cart_comm, &req[79]);
    MPI_Irecv(recvbuf + 65323008, 9216, MPI_CHAR, !commBase.MyRank(), 72, cart_comm, &req[79]);

    rootLogger.info("comm 9");
    MPI_Isend(sendbuf + 65332224, 9216, MPI_CHAR, !commBase.MyRank(), 70, cart_comm, &req[77]);
    MPI_Irecv(recvbuf + 65332224, 9216, MPI_CHAR, !commBase.MyRank(), 73, cart_comm, &req[77]);

    rootLogger.info("comm 10");
    MPI_Isend(sendbuf + 65341440, 9216, MPI_CHAR, !commBase.MyRank(), 69, cart_comm, &req[75]);
    MPI_Irecv(recvbuf + 65341440, 9216, MPI_CHAR, !commBase.MyRank(), 74, cart_comm, &req[75]);

    rootLogger.info("comm 11");
    MPI_Isend(sendbuf + 65350656, 9216, MPI_CHAR, !commBase.MyRank(), 68, cart_comm, &req[73]);
    MPI_Irecv(recvbuf + 65350656, 9216, MPI_CHAR, !commBase.MyRank(), 75, cart_comm, &req[73]);

    rootLogger.info("comm 12");
    MPI_Isend(sendbuf + 65359872, 9216, MPI_CHAR, !commBase.MyRank(), 67, cart_comm, &req[71]);
    MPI_Irecv(recvbuf + 65359872, 9216, MPI_CHAR, !commBase.MyRank(), 76, cart_comm, &req[71]);

    rootLogger.info("comm 13");
    MPI_Isend(sendbuf + 65369088, 9216, MPI_CHAR, !commBase.MyRank(), 66, cart_comm, &req[69]);
    MPI_Irecv(recvbuf + 65369088, 9216, MPI_CHAR, !commBase.MyRank(), 77, cart_comm, &req[69]);

    rootLogger.info("comm 14");
    MPI_Isend(sendbuf + 65378304, 9216, MPI_CHAR, !commBase.MyRank(), 65, cart_comm, &req[67]);
    MPI_Irecv(recvbuf + 65378304, 9216, MPI_CHAR, !commBase.MyRank(), 78, cart_comm, &req[67]);

    rootLogger.info("comm 15");
    MPI_Isend(sendbuf + 65387520, 9216, MPI_CHAR, !commBase.MyRank(), 64, cart_comm, &req[65]);
    MPI_Irecv(recvbuf + 65387520, 9216, MPI_CHAR, !commBase.MyRank(), 79, cart_comm, &req[65]);

    rootLogger.info("Wait for requests...");
    MPI_Waitall(16, &req[64], MPI_STATUS_IGNORE);
    rootLogger.info("Communication done.");


    rootLogger.info("Free requests...");
    for (int i = 64; i < 80; i++){
        if ((req[i] != MPI_REQUEST_NULL) && (req[i] != 0)) {
            MPI_Request_free(&req[i]);
        }
    }
    rootLogger.info("Requests freed.");

    rootLogger.info("Free memory...");
    freeDevice(gaugefield);
    freeDevice(recvbuf);
    freeDevice(sendbuf);
    freeHost(hostbufrecv);
    freeHost(hostbufsend);

    rootLogger.info("Memory freed.");
    rootLogger.info("");
    rootLogger.info("=====================");
    rootLogger.info("======= STOP ========");
    rootLogger.info("=====================\n");
}

void run0(CommunicationBase &commBase) {
    size_t size = 65396736;

    char *sendBufferDevice;
    char *recvBufferDevice;

    allocDevice(&sendBufferDevice, size);
    allocDevice(&recvBufferDevice, size);

    char *sendBufferHost;
    char *recvBufferHost;

    allocHost(&sendBufferHost, size);
    allocHost(&recvBufferHost, size);

    StopWatch<true> timer;

    for (size_t i = 0; i < size; i++) {
        sendBufferHost[i] = (char) (i % 100 + commBase.MyRank());
    }

    copyHostToDevice(sendBufferHost, sendBufferDevice, size);

    MPI_Request requestSend[2];

    int tag = 0;
    for (int j = 0; j < 10; ++j) {

        timer.reset();
        timer.start();
#ifdef USE_GPU_AWARE_MPI
        rootLogger.info("Without memcpies!");
        MPI_Isend(sendBufferDevice, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[0]);
        MPI_Irecv(recvBufferDevice, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[1]);

       MPI_Waitall(2, requestSend, MPI_STATUS_IGNORE);

#else
        rootLogger.info("Do memcpies.");
        copyDeviceToHost(sendBufferDevice, sendBufferHost, size);
        MPI_Isend(sendBufferHost, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[0]);
        MPI_Irecv(recvBufferHost, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[1]);

        MPI_Waitall(2, requestSend, MPI_STATUS_IGNORE);

        copyHostToDevice(recvBufferHost, recvBufferDevice, size);
#endif
        timer.stop();

        copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

        rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
        stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

        rootLogger.info("Time: " ,  timer ,  " (That should be worse than without gpu-awareness) ");
    }

    timer.reset();
    timer.start();
#ifdef USE_GPU_AWARE_MPI
    rootLogger.info("Without memcpies!");
    MPI_Isend(sendBufferDevice, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[0]);
    MPI_Irecv(recvBufferDevice, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[1]);

    MPI_Waitall(2, requestSend, MPI_STATUS_IGNORE);

#else
    rootLogger.info("Do memcpies.");
    copyDeviceToHost(sendBufferDevice, sendBufferHost, size);
    MPI_Isend(sendBufferHost, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[0]);
    MPI_Irecv(recvBufferHost, size, MPI_CHAR, !commBase.MyRank(), tag, commBase.getCart_comm(), &requestSend[1]);


    MPI_Waitall(2, requestSend, MPI_STATUS_IGNORE);

    copyHostToDevice(recvBufferHost, recvBufferDevice, size);
#endif
    timer.stop();

    copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

    rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
    stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

    rootLogger.info("Last time: " ,  timer ,  " (That should be better than without gpu-awareness)");

    freeDevice(sendBufferDevice);
    freeDevice(recvBufferDevice);
    freeHost(sendBufferHost);
    freeHost(recvBufferHost);
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;
    const int LatDim[] = {20, 20, 20, 20};
    const int NodeDim[] = {2, 1, 1, 1};
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    run0(commBase);

#ifdef USE_GPU_AWARE_MPI
    run1(commBase);
    run1(commBase);
    run1(commBase);
#endif
    rootLogger.info(CoutColors::green,"Test passed!",CoutColors::reset);
    return 0;
}

