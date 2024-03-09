# Created by: Jinchen
# Created on: 2024.01.08

FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# Time zone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install GCC 9.2
RUN apt-get update && apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Install OpenMPI with CUDA support
RUN apt-get install -y openmpi-bin libopenmpi-dev

# Install CMake
RUN apt install -y build-essential libssl-dev autoconf automake tar wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2.tar.gz
RUN tar -zxvf cmake-3.19.2.tar.gz
RUN cd cmake-3.19.2 && ./bootstrap && make && make install && cd ..

# Install Python
RUN apt-get update && apt-get install -y python3.8 python3-pip

# Install vim, git and wget
RUN apt-get update && apt-get install -y vim git wget

# Install SIMULATeQCD
#TODO: change the DARCHITECTURE flag according to the GPU you are using
RUN git clone https://github.com/LatticeQCD/SIMULATeQCD.git -b v1.2.0
RUN cd SIMULATeQCD && pip install -r requirements.txt && cd ..
RUN mkdir buildsimqcd
RUN cd buildsimqcd && cmake ../SIMULATeQCD/ \
-DARCHITECTURE="80" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=ON \