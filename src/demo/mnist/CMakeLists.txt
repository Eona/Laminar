message("MNIST training demo")

set(MNIST_CUBLAS_EXE 
    cublas_train)

cublas_add_executable(${MNIST_CUBLAS_EXE} ${MNIST_CUBLAS_EXE}.cu)