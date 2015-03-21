find_package(CUDA REQUIRED)

# Ugly workaround for CMake-NVCC bug
if (APPLE)
    set(CUDA_HOST_COMPILER /usr/bin/clang++)
elseif(UNIX)
    set(CUDA_HOST_COMPILER /usr/bin/g++)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -O3 -arch=sm_30)

#cuda_add_executable(${MAIN} ${SOURCES})

#CUDA_ADD_CUFFT_TO_TARGET( cuda_target )
#-- Adds the cufft library to the target (can be any target).  Handles whether
#you are in emulation mode or not.

#CUDA_ADD_CUBLAS_TO_TARGET( cuda_target )
#-- Adds the cublas library to the target (can be any target).  Handles
#whether you are in emulation mode or not.

#CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ...
#    [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#-- Creates an executable "cuda_target" which is made up of the files
#specified.  All of the non CUDA C files are compiled using the standard
#build rules specified by CMAKE and the cuda files are compiled to object
#files using nvcc and the host compiler.  In addition CUDA_INCLUDE_DIRS is
#added automatically to include_directories().  Some standard CMake target
#calls can be used on the target after calling this macro
#(e.g. set_target_properties and target_link_libraries), but setting
#properties that adjust compilation flags will not affect code compiled by
#nvcc.  Such flags should be modified before calling CUDA_ADD_EXECUTABLE,
#CUDA_ADD_LIBRARY or CUDA_WRAP_SRCS.

#CUDA_ADD_LIBRARY( cuda_target file0 file1 ...
#    [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#-- Same as CUDA_ADD_EXECUTABLE except that a library is created.

#CUDA_BUILD_CLEAN_TARGET()
#-- Creates a convience target that deletes all the dependency files
#generated.  You should make clean after running this target to ensure the
#dependency files get regenerated.

#CUDA_COMPILE( generated_files file0 file1 ... [STATIC | SHARED | MODULE]
#    [OPTIONS ...] )
#-- Returns a list of generated files from the input source files to be used
#with ADD_LIBRARY or ADD_EXECUTABLE.
