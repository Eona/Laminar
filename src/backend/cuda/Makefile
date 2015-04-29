NVCC=nvcc
CXX=g++
ARCH=sm_35
OPT=-std=c++11
LIB=-lcublas -lcublas_device -lcudadevrt

# here are all the objects
# # make and compile
#
test:
	$(NVCC) -arch=$(ARCH) $(OPT) -o run test_engine.cu $(LIB)
