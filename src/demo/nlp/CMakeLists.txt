message("Eona build demo/NLP")

if (true)
set(NLP_CUBLAS_EXE 
    cublas_nlp_train)
cublas_add_executable(${NLP_CUBLAS_EXE} ${NLP_CUBLAS_EXE}.cu)
link_opencl(${NLP_CUBLAS_EXE})
endif()

if (true)
set(NLP_VECMAT_EXE
    vecmat_nlp_train)
add_executable(${NLP_VECMAT_EXE} ${NLP_VECMAT_EXE}.cpp)
endif()
    
if (true)
set(NLP_EIGEN_EXE
    eigen_nlp_train)
add_executable(${NLP_EIGEN_EXE} ${NLP_EIGEN_EXE}.cpp)
endif()
    