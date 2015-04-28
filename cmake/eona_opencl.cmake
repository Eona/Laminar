find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

function(opencl_add_executable target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} ${OPENCL_LIBRARIES})
endfunction()

function(link_opencl targets)
    list(APPEND targets ${ARGN})
    foreach(target IN ITEMS ${targets})
        target_link_libraries(${target} ${OPENCL_LIBRARIES})
    endforeach()
endfunction()
