message("Eona build test/")
include(eona_gtest)

# toggle this to avoid recompiling gtests
set(RUN_CPU_GTEST true)
set(RUN_CUDA_GTEST true)
set(RUN_OPENCL_GTEST true)

# global option from eona_gtest.cmake
set(RUN_TEST_DURING_BUILD false)
set(USE_CUBLAS true)

set(CPU_TEST_SUITE 
    dummy_forward_test dummy_rnn_test 
    dummy_lstm_test
    vecmat_forward_test vecmat_rnn_test 
    vecmat_lstm_test
    eigen_forward_test
#    misc_test
    )

set(CUDA_TEST_SUITE 
    cublas_forward_test
    cublas_lstm_test
    )
    
set(OPENCL_TEST_SUITE
    opencl_forward_test
    opencl_lstm_test
    )
    
# Initialize to empty list
set(TEST_SUITE ${EMPTY_STRING})
    
if (RUN_CPU_GTEST)
    #> add_gtest(${TEST_SUITE} ${TEST_SUITE}.cpp)
    add_multiple_gtests(${CPU_TEST_SUITE})
    
    list(APPEND TEST_SUITE ${CPU_TEST_SUITE})
endif()

if (RUN_CUDA_GTEST)
    #> add_gtest_cuda(${TEST_SUITE} ${TEST_SUITE}.cu)
    add_multiple_gtests_cuda(${CUDA_TEST_SUITE})

    list(APPEND TEST_SUITE ${CUDA_TEST_SUITE})
endif()

if (RUN_OPENCL_GTEST)
    add_multiple_gtests(${OPENCL_TEST_SUITE})
#    foreach(target IN ITEMS ${OPENCL_TEST_SUITE})
    link_opencl(${OPENCL_TEST_SUITE})
#    link_opencl(${target})
#    endforeach()
    
    list(APPEND TEST_SUITE ${OPENCL_TEST_SUITE})
endif()

# $TEST_SUITE contains all tests desired to run
# so that you can run 'make gtest'
#> add_custom_target(gtest COMMAND ${TEST_SUITE})
# from eona_utils.cmake:
if (TEST_SUITE)
    add_multiple_targets(gtest ${TEST_SUITE})
endif()