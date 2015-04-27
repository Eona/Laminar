message("Eona build demo/mnist")

if (false)
set(NLP_CUBLAS_EXE 
    cublas_train_nlp)
cublas_add_executable(${NLP_CUBLAS_EXE} ${NLP_CUBLAS_EXE}.cu)
endif()

if (false)
set(NLP_OPENCL_EXE 
    opencl_train_nlp)
opencl_add_executable(${NLP_OPENCL_EXE} ${NLP_OPENCL_EXE}.cpp)
endif()

if (true)
set(NLP_VECMAT_EXE
    vecmat_train_nlp)
add_executable(${NLP_VECMAT_EXE} ${NLP_VECMAT_EXE}.cpp)
endif()
    