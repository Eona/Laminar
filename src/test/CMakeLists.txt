message("Folder test/ for gtest template")
include(eona_gtest)

# toggle this to avoid recompiling gtests
set(RUN_CPU_GTEST true)
set(RUN_GPU_GTEST false)

set(CPU_TEST_SUITE 
    dummy_forward_test dummy_rnn_test 
    dummy_lstm_test
    vecmat_forward_test vecmat_rnn_test 
    vecmat_lstm_test
#    misc_test
    )

set(GPU_TEST_SUITE 
    cuda_test
    )
    
if (RUN_CPU_GTEST)
    #> add_gtest(${TEST_SUITE} ${TEST_SUITE}.cpp)
    add_multiple_gtests(${CPU_TEST_SUITE})
    
    set(TEST_SUITE ${CPU_TEST_SUITE})
endif()

if (RUN_GPU_GTEST)
    #> add_gtest_cuda(${TEST_SUITE} ${TEST_SUITE}.cu)
    add_multiple_gtests_cuda(${GPU_TEST_SUITE})
    
    list(APPEND TEST_SUITE ${GPU_TEST_SUITE})
endif()

# $TEST_SUITE contains all tests desired to run
# so that you can run 'make gtest'
#> add_custom_target(gtest COMMAND ${TEST_SUITE})
# from eona_utils.cmake:
add_multiple_targets(gtest ${TEST_SUITE})