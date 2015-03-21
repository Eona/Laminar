enable_testing()
add_definitions(-std=c++11 -pthread)

if (NOT DEFINED GTEST_ROOT)
    if (APPLE OR UNIX)
        set(GTEST_ROOT "/opt/gtest")
    endif()
endif()

# Workaround OS-specific problems
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # force this option to ON so that Google Test will use /MD instead of /MT
    # /MD is now the default for Visual Studio, so it should be our default, too
    option(gtest_force_shared_crt
        "Use shared (DLL) run-time lib even when Google Test is built as static lib."
        ON)
elseif (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
endif()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# add_gtest(<target> <sources>...)
#
#  Adds a GTest test executable, <target>, built from <sources> and
#  adds the test so that CTest will run it. Both the executable and the test
#  will be named <target>.
#
# additional args can be accessed via ARG0, ARG1 ... ARGN
function(add_gtest target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(${target} ${Boost_LIBRARYDIR})
    target_link_libraries(${target} -pthread)

    add_test(${target} ${target})

    # run test after each time it's built
    # Here we simply run the test and if it fails the build will stop. However if you were to build again immediately the failed test would not be run again and the build will continue. 
    add_custom_command(TARGET ${target}
        POST_BUILD
        COMMAND ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "My GTest running ${target}" VERBATIM)
endfunction()

# add_multiple_gtests(<target1> <target2> ...)
# default source name: <target>.cpp
# one target may only link to one source. 
function(add_multiple_gtests target0)
    set(targets ${target0} ${ARGN})
    foreach(target IN ITEMS ${targets})
        add_gtest(${target} ${target}.cpp)
    endforeach()
endfunction()
