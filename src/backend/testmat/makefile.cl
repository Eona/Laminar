GCC = gcc
CXX = g++
LIB = -I/usr/local/cuda/include/
LD_LIB = -L/usr/local/cuda/lib -l OpenCL
# make and compile
all:
	$(CXX) opencl_engine.c $(LIB) $(LD_LIB) 

