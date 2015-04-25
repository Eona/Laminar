find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

function(opencl_add_executable target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} ${OPENCL_LIBRARIES})
endfunction()
