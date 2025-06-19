FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

RUN apt-get update && apt-get install -y \ 
    cmake curl git vim tmux wget libssl-dev \
    ninja-build build-essential \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    python3.10 python3-pip ffmpeg && apt-get clean

# CMake 3.31.1
RUN mkdir -p /root/repos && cd /root/repos && \ 
    git clone --depth=1 --branch v3.31.1 https://github.com/Kitware/CMake.git && \
    cd CMake && mkdir -p build && cd build && \
    cmake .. && make -j4 && make install

WORKDIR /root/repos
RUN mkdir cudss && cd cudss && \ 
    wget -c -t 0 https://developer.download.nvidia.com/compute/cudss/0.5.0/local_installers/cudss-local-repo-ubuntu2004-0.5.0_0.5.0-1_amd64.deb

RUN git clone --branch 3.4.0 --recursive https://gitlab.com/libeigen/eigen.git
RUN git clone --recursive https://github.com/ceres-solver/ceres-solver.git
RUN git clone --recursive https://github.com/colmap/colmap.git
RUN git clone --recursive https://github.com/colmap/glomap.git

# cudss
RUN cd /root/repos/cudss && \ 
    dpkg -i cudss-local-repo-ubuntu2004-0.5.0_0.5.0-1_amd64.deb && \
    cp /var/cudss-local-repo-ubuntu2004-0.5.0/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get -y install cudss

# Eigen
RUN cd /root/repos/eigen && mkdir -p build && cd build && \ 
    cmake .. && make -j$(nproc) && make install

# Ceres Solver
RUN cd /root/repos/ceres-solver && mv third_party _third_party
RUN cd /root/repos/ceres-solver/_third_party/googletest && mkdir build && cd build && \ 
    cmake .. && make -j$(nproc) && make install
RUN cd /root/repos/ceres-solver/_third_party/abseil-cpp && mkdir build && cd build && \ 
    cmake .. && make -j$(nproc) && make install
RUN cd /root/repos/ceres-solver && mkdir -p build && cd build && \ 
    sed -i 's/find_dependency(cudss 0.5.0)/find_dependency(cudss)/' /root/repos/ceres-solver/CMakeLists.txt && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=$(which nvcc) .. && \
    make -j4 && make install
    
# COLMAP
RUN cd /root/repos/colmap && mkdir -p build && cd build && \ 
    cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=$(which nvcc) .. -GNinja && \
    ninja -j2 && ninja install

RUN cd /root/repos/glomap && mkdir -p build && cd build && \ 
    cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=$(which nvcc) .. -GNinja && \
    ninja -j2 && ninja install

# GLOMAP

ENV PATH="/usr/local/bin:/colmap/bin:${PATH}"
ENV PATH="/glomap/bin:${PATH}"

RUN mkdir /workspace
COPY scripts/ /workspace/scripts
WORKDIR /workspace