message("Eona build demo/MNIST")

if (true)
set(MNIST_CUBLAS_EXE 
    cublas_mnist_train)
cublas_add_executable(${MNIST_CUBLAS_EXE} ${MNIST_CUBLAS_EXE}.cu)
link_opencl(${MNIST_CUBLAS_EXE})
endif()

if (true)
set(MNIST_VECMAT_EXE
    vecmat_mnist_train)
add_executable(${MNIST_VECMAT_EXE} ${MNIST_VECMAT_EXE}.cpp)
endif()
    
if (true)
set(MNIST_EIGEN_EXE
    eigen_mnist_train)
add_executable(${MNIST_EIGEN_EXE} ${MNIST_EIGEN_EXE}.cpp)
endif()
    