message("Eona build demo/mnist")

if (true)
set(MNIST_CUBLAS_EXE 
    cublas_mnist_train)
cublas_add_executable(${MNIST_CUBLAS_EXE} ${MNIST_CUBLAS_EXE}.cu)
endif()

if (false)
set(MNIST_OPENCL_EXE 
    opencl_mnist_train)
opencl_add_executable(${MNIST_OPENCL_EXE} ${MNIST_OPENCL_EXE}.cpp)
endif()

if (false)
set(MNIST_VECMAT_EXE
    vecmat_mnist_train)
add_executable(${MNIST_VECMAT_EXE} ${MNIST_VECMAT_EXE}.cpp)
endif()
    