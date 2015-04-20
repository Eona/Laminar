GCC = gcc
CXX = g++
LIB = -I/usr/local/cuda/include/
LD_LIB = -L/usr/local/cuda/lib -framework OpenCL
# make and compile
all:
	$(CXX) -std=c++11 ocl_engine_test.cc $(LIB) $(LD_LIB) 

