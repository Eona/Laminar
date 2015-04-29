message("Eona build backend/types")

set(PROFILE_TEST_EXE 
    profile_test)

cublas_add_executable(${PROFILE_TEST_EXE} ${PROFILE_TEST_EXE}.cpp)
link_opencl(${PROFILE_TEST_EXE})