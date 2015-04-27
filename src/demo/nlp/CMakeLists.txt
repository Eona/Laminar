message("Eona build demo/NLP")

if (false)
set(NLP_CUBLAS_EXE 
    cublas_nlp_train)
cublas_add_executable(${NLP_CUBLAS_EXE} ${NLP_CUBLAS_EXE}.cu)
endif()

if (false)
set(NLP_OPENCL_EXE 
    opencl_nlp_train)
opencl_add_executable(${NLP_OPENCL_EXE} ${NLP_OPENCL_EXE}.cpp)
endif()

if (true)
set(NLP_VECMAT_EXE
    vecmat_nlp_train)
add_executable(${NLP_VECMAT_EXE} ${NLP_VECMAT_EXE}.cpp)
endif()
    