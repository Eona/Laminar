cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)
project(Laminar CXX)

# you can isolate CMake code out by adding a new dir to your src, and add a
# new file MyXXX.cmake:
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
include(eona_utils)
include(eona_cuda)
include(eona_opencl)
include(eona_eigen)

set(RUN_GTEST false)
set(RUN_DEMO true)

# CMAKE_CURRENT_BINARY_DIR: where generated cmake files go 
# CMAKE_BINARY_DIR: top level of your build tree
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/bin)

# =================== GTests ==================
if (RUN_GTEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test 
        ${CMAKE_CURRENT_BINARY_DIR}/test)
endif()

# =================== Other tests ==================
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/types 
#    ${CMAKE_CURRENT_BINARY_DIR}/backend/types)

# =================== DEMO ==================
if (RUN_DEMO)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/demo/mnist 
                    ${CMAKE_CURRENT_BINARY_DIR}/mnist)
    
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/demo/nlp 
                    ${CMAKE_CURRENT_BINARY_DIR}/nlp)
endif()

# ==========================================
message("Eona build src/ toplevel")

set(MAIN
    laminar)

set(SOURCES 
    ${MAIN}.cu
    )

if (true)
    cublas_add_executable(${MAIN} ${SOURCES})
    target_link_libraries(${MAIN} ${OPENCL_LIBRARIES})
endif()

# ==== boost ====
#   include(eona_boost)
#   link_boost(${MAIN} ${Boost_LIBRARIES})
#   add_boost(boosttest boosttest.cpp)
