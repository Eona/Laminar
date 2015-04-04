#!/bin/bash
# arg1: give a build directory name. Defaults to 'project'

BUILD_DIR=ubuntu

if [ $# -eq 1 ]; then
    BUILD_DIR=$1
fi

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
else
    echo $BUILD_DIR already exists.
fi

if [[ "$(whoami)" == "Eona" && "$OSTYPE" == "darwin"* ]]; then
    echo Mac OS X detected.
    EXTRA_FLAG="-DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/cuda-7.0 -DGTEST_ROOT=/opt/gtest/clang"
else
    EXTRA_FLAG=
fi

cd $BUILD_DIR
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_CXX_COMPILER_ARG1=-std=c++11 -DCMAKE_BUILD_TYPE=Release ../src/ $EXTRA_FLAG 
cd ..

echo
echo Patch eclipse project configurations:
python patch_eclipse_nvcc.py $BUILD_DIR

echo
echo DONE
