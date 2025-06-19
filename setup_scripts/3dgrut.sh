#!/bin/bash

sudo apt-get install gcc-11 g++-11

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh

conda init
exec bash

git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

chmod +x install_env.sh
CUDA_VERSION=12.8.1 ./install_env.sh 3dgrut WITH_GCC11

conda activate 3dgrut
