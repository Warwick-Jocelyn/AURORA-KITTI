#!/bin/bash
source /path/to/aurora-root/anaconda3/etc/profile.d/conda.sh
conda activate OMNIDC
export CUDA_HOME=/path/to/aurora-root/CUDA/cuda-11.3
export PATH=/path/to/aurora-root/Code/NLSPN/gcc8_shim:$CUDA_HOME/bin:/usr/bin:$PATH
export CC=gcc-8
export CXX=g++-8
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
cd /path/to/aurora-root/Code/NLSPN/src/model/deformconv
rm -rf build dist DCN.egg-info
echo "gcc -> $(gcc --version | head -1)"
python setup.py build install 2>&1
echo "BUILD_EXIT=$?"
