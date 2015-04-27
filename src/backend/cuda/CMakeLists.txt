message("Eona build backend/cublas")

set(CUBLAS_TEST_EXE 
    gradcheck)

#cublas_add_executable(${CUBLAS_TEST_EXE} ${CUBLAS_TEST_EXE}.cu)