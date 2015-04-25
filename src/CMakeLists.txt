cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)
project(Laminar CXX)

# you can isolate CMake code out by adding a new dir to your src, and add a
# new file MyXXX.cmake:
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
include(eona_utils)
include(eona_cuda)

# CMAKE_CURRENT_BINARY_DIR: where generated cmake files go 
# CMAKE_BINARY_DIR: top level of your build tree
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test 
    ${CMAKE_CURRENT_BINARY_DIR}/test)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cublas 
    ${CMAKE_CURRENT_BINARY_DIR}/backend/cublas)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/opencl 
#    ${CMAKE_CURRENT_BINARY_DIR}/backend/opencl)

# ==========================================
message("Folder src/")

set(MAIN
    laminar)

set(SOURCES 
    ${MAIN}.cu
    )

cuda_add_executable(${MAIN} ${SOURCES})

# ==== boost ====
#   include(eona_boost)
#   link_boost(${MAIN} ${Boost_LIBRARIES})
#   add_boost(boosttest boosttest.cpp)
