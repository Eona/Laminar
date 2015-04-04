message("Folder test/ for gtest template")
include(eona_gtest)

set(TEST_SUITE 
    cuda_test
    forward_net_test
    recurrent_net_test
    )
    
#add_gtest(${TEST_SUITE} ${TEST_SUITE}.cpp)
add_multiple_gtests_cuda(${TEST_SUITE})

# so that you can run 'make gtest'
#add_custom_target(gtest COMMAND ${TEST_SUITE})
# from eona_utils.cmake:
add_multiple_targets(gtest ${TEST_SUITE})
