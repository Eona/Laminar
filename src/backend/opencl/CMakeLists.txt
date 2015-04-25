message("Folder backend/opencl")

set(OPENCL_TEST_EXE 
    ocl_engine_test)

set(OPENCL_TEST_SRC 
    ${OPENCL_TEST_EXE}.cc)

opencl_add_executable(${OPENCL_TEST_EXE} ${OPENCL_TEST_SRC})