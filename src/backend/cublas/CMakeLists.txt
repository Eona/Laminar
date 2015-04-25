message("Folder backend/cublas")

set(CUBLAS_TEST_EXE 
    test_engine)

set(CUBLAS_TEST_SRC 
    test_engine.cu)

cublas_add_executable(${CUBLAS_TEST_EXE} ${CUBLAS_TEST_SRC})