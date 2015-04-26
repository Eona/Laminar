message("MNIST training demo")

if (false)
set(MNIST_CUBLAS_EXE 
    cublas_train)
cublas_add_executable(${MNIST_CUBLAS_EXE} ${MNIST_CUBLAS_EXE}.cu)
endif()

if (true)
set(MNIST_VECMAT_EXE
    vecmat_train)
add_executable(${MNIST_VECMAT_EXE} ${MNIST_VECMAT_EXE}.cpp)
endif()
    