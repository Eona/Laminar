message("Folder test/ for gtest template")
include(eona_gtest)

set(TEST_SUITE 
#    forward_dummy_test rnn_dummy_test
#    forward_vecmat_test 
    rnn_vecmat_test
#   cuda_test
#    misc_test
    )
    
#> add_gtest(${TEST_SUITE} ${TEST_SUITE}.cpp)
add_multiple_gtests_cuda(${TEST_SUITE})

# so that you can run 'make gtest'
#> add_custom_target(gtest COMMAND ${TEST_SUITE})
# from eona_utils.cmake:
add_multiple_targets(gtest ${TEST_SUITE})
