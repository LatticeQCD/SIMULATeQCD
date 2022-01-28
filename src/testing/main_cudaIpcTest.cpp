/* 
 * main_cudaIpcTest.cpp                                                               
 * 
 * Lukas Mazur
 * 
 */
#include "../SIMULATeQCD.h"

// This test checks whether Gpu inter-process communication works.
// Since our code is based on MPI processes, we have to communicate our gpu memory pointer first.
// That is why we need Gpu inter-process communication.

void allocDevice(uint8_t **_memoryChunk, size_t size) {
    gpuError_t gpuErr = gpuMalloc((void **) _memoryChunk, size);
    if (gpuErr) GpuError("_memoryChunk: Failed to allocate memory on device", gpuErr);
}

void freeDevice(uint8_t *_memoryChunk) {
    gpuError_t gpuErr = gpuFree(_memoryChunk);
    if (gpuErr) {
        GpuError("_memoryChunk: Failed to free memory on host", gpuErr);
    }
}

void allocHost(uint8_t **_memoryChunk, size_t size) {
    gpuError_t gpuErr = gpuMallocHost((void **) _memoryChunk, size);
    if (gpuErr) GpuError("_memoryChunk: Failed to allocate memory on device", gpuErr);
}

void freeHost(uint8_t *_memoryChunk) {
    gpuError_t gpuErr = gpuFreeHost(_memoryChunk);
    if (gpuErr) {
        GpuError("_memoryChunk: Failed to free memory on host", gpuErr);
    }
}

void copyHostToDevice(uint8_t *_memoryChunkHost, uint8_t *_memoryChunkDevice, size_t size) {
    gpuError_t gpuErr = gpuMemcpy(_memoryChunkDevice, _memoryChunkHost, size, gpuMemcpyHostToDevice);
    if (gpuErr) GpuError("Failed to copy data H2D", gpuErr);
}

void copyDeviceToHost(uint8_t *_memoryChunkDevice, uint8_t *_memoryChunkHost, size_t size) {
    gpuError_t gpuErr = gpuMemcpy(_memoryChunkHost, _memoryChunkDevice, size, gpuMemcpyDeviceToHost);
    if (gpuErr)
        GpuError("Failed to copy data D2H", gpuErr);
}

uint8_t *sendRecvHandles(uint8_t *recvBuffer, int rank, MPI_Comm comm) {

    int size = sizeof(gpuIpcMemHandle_t) + 1;
    uint8_t handleBuffer[size];

    gpuIpcMemHandle_t recvHandle;
    gpuError_t gpuErr = gpuIpcGetMemHandle(&recvHandle, recvBuffer);
    if (gpuErr) GpuError("gpuIpcGetMemHandle", gpuErr);

    MPI_Sendrecv((uint8_t *) (&recvHandle), size, MPI_CHAR, rank, 0,
                 handleBuffer, size, MPI_CHAR, rank, 0,
                 comm, MPI_STATUS_IGNORE);

    gpuIpcMemHandle_t neighborHandle;
    memcpy((uint8_t *) (&neighborHandle), handleBuffer, sizeof(neighborHandle));

    uint8_t *neighborBuffer;
    gpuErr = gpuIpcOpenMemHandle((void **) &neighborBuffer, neighborHandle, gpuIpcMemLazyEnablePeerAccess);
    if (gpuErr) GpuError("gpuIpcOpenMemHandle", gpuErr);

    return neighborBuffer;
}

void sendRecvBufferMPI(int rank, uint8_t *sendBufferHost, uint8_t *recvBufferHost,
                       uint8_t *sendBufferDevice, uint8_t *recvBufferDevice, int size,
                       MPI_Request requestSend[2], MPI_Comm comm) {
    int tag = 0;
    rootLogger.info("Do memcpies.");
    copyDeviceToHost(sendBufferDevice, sendBufferHost, size);
    MPI_Isend(sendBufferHost, size, MPI_UINT8_T, rank, tag, comm, &requestSend[0]);
    MPI_Irecv(recvBufferHost, size, MPI_UINT8_T, rank, tag, comm, &requestSend[1]);

    MPI_Waitall(2, requestSend, MPI_STATUS_IGNORE);

    copyHostToDevice(recvBufferHost, recvBufferDevice, size);
}

void sendRecvBufferP2P(CommunicationBase& commBase, uint8_t *sendBufferDevice, uint8_t *recvBufferDeviceP2P, int size, gpuStream_t &deviceStream){

    gpuError_t gpuErr;
    gpuErr = gpuMemcpyAsync(recvBufferDeviceP2P, sendBufferDevice, size, gpuMemcpyDefault, deviceStream);
    if (gpuErr) GpuError("gpuMemcpyAsync", gpuErr);
    
    gpuErr = gpuStreamSynchronize(deviceStream);
    if (gpuErr) GpuError("gpuStreamSynchronize", gpuErr);
    
    commBase.nodeBarrier();
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/cudaIpcTest.param", argc, argv);
    commBase.init(param.nodeDim());
    StopWatch<true> timer;
    
    if (!commBase.useGpuP2P()) {
        throw std::runtime_error(stdLogger.fatal("P2P is not activated. Exit."));
    }

    size_t size = 200000000;

    uint8_t *sendBufferDevice;
    uint8_t *recvBufferDevice;

    allocDevice(&sendBufferDevice, size);
    allocDevice(&recvBufferDevice, size);

    int offset = 0;
    uint8_t *recvBufferDeviceP2P;
    recvBufferDeviceP2P = sendRecvHandles(recvBufferDevice, !commBase.MyRank(), commBase.getCart_comm());

    uint8_t *sendBufferHost;
    uint8_t *recvBufferHost;

    allocHost(&sendBufferHost, size);
    allocHost(&recvBufferHost, size);

    for (size_t i = 0; i < size; i++) {
        sendBufferHost[i] = (uint8_t) (i % 100 + commBase.MyRank());
    }

    copyHostToDevice(sendBufferHost, sendBufferDevice, size);

    MPI_Request requestSend[2];
    gpuStream_t deviceStream;
    
    gpuError_t gpuErr;
    gpuErr = gpuStreamCreate(&deviceStream);
    if (gpuErr) GpuError("gpuStreamCreate", gpuErr);


    ////// CHECK TIMINGS WITH CUDA DIRECT P2P (IPC) COMMUNICATION //////

    for (int j = 0; j < 4; ++j) {

        timer.reset();
        timer.start();

        sendRecvBufferP2P(commBase,sendBufferDevice, &recvBufferDeviceP2P[offset], size-offset, deviceStream);
        timer.stop();

        copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

        rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
        stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

        rootLogger.info("P2P Time: " ,  timer);
    }

    timer.reset();
    timer.start();

    sendRecvBufferP2P(commBase,sendBufferDevice, recvBufferDeviceP2P+offset, size-offset, deviceStream);
    timer.stop();

    copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

    rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
    stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

    rootLogger.info("P2P Last time: " ,  timer);

    ////// CHECK TIMINGS WITH STANDARD MPI COMMUNICATION //////

    for (int j = 0; j < 4; ++j) {

        timer.reset();
        timer.start();

        sendRecvBufferMPI(!commBase.MyRank(), sendBufferHost, recvBufferHost, sendBufferDevice, recvBufferDevice, size,
                          requestSend, commBase.getCart_comm());
        timer.stop();

        copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

        rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
        stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

        rootLogger.info("MPI Time: " ,  timer);
    }

    timer.reset();
    timer.start();

    sendRecvBufferMPI(!commBase.MyRank(), sendBufferHost, recvBufferHost, sendBufferDevice, recvBufferDevice, size,
                      requestSend, commBase.getCart_comm());
    timer.stop();

    copyDeviceToHost(recvBufferDevice, recvBufferHost, size);

    rootLogger.info("Note that in what follows, the output will look strange. It's okay.");
    stdLogger.info(recvBufferHost[0] ,  recvBufferHost[(int) (size / 2.)] ,  recvBufferHost[size - 1]);

    rootLogger.info("MPI Last time: " ,  timer);

    rootLogger.info(CoutColors::green,"Test passed!",CoutColors::reset);
    gpuErr = gpuIpcCloseMemHandle(recvBufferDeviceP2P);
    if (gpuErr) GpuError("gpuIpcCloseMemHandle", gpuErr);
    freeDevice(sendBufferDevice);
    freeDevice(recvBufferDevice);
    freeHost(sendBufferHost);
    freeHost(recvBufferHost);

    return 0;
}

