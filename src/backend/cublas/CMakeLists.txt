message("Folder backend/cublas")

set(CUBLAS_TEST_EXE 
    test_engine)

set(CUBLAS_TEST_SRC 
    test_engine.cu)

cuda_add_executable(${CUBLAS_TEST_EXE} ${CUBLAS_TEST_SRC})
cuda_add_cublas_to_target(${CUBLAS_TEST_EXE})