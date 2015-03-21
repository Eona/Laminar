add_definitions(-std=c++11 -pthread)

set(CONFIGURED_ONCE TRUE CACHE INTERNAL
    "A flag showing that CMake has configured at least once.")

# link -pthread
function (link_thread target)
    # same effect:
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    target_link_libraries(${target} -pthread)
endfunction()

# Create an executable and link it to pthread
function (add_thread target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} -pthread)
endfunction()

# Add multiple custom targets
# e.g. gtest run together
# add_multiple_targets(target_name cmd1 cmd2 cmd3 ...)
# will be expanded to system call:
# add_custom_target(target_name COMMAND cmd1 COMMAND cmd2 ....)
function (add_multiple_targets target_name cmd0)
    set(cmds ${cmd0} ${ARGN})
    set(targets ${target_name})
    foreach(cmd ${cmds})
        list(APPEND targets COMMAND ${cmd})
    endforeach()
    add_custom_target(${targets})
endfunction()
