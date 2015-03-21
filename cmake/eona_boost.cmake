if (NOT DEFINED     BOOST_ROOT        AND
    NOT DEFINED ENV{BOOST_ROOT}       AND
    NOT DEFINED     BOOST_INCLUDEDIR  AND
    NOT DEFINED ENV{BOOST_INCLUDEDIR} AND
    NOT DEFINED     BOOST_LIBRARYDIR  AND
    NOT DEFINED ENV{BOOST_LIBRARYDIR})
    if (APPLE OR UNIX)
        # default: /include and /lib
        set(BOOST_ROOT "/opt/boost/")
    elseif (WIN32)
        set(BOOST_INCLUDEDIR "C:/local/boost_1_55_0")
        set(BOOST_LIBRARYDIR "C:/local/boost_1_55_0/lib32-msvc-10.0")
    elseif (UNIX)
       #set if not default by BOOST_ROOT:
       #set(BOOST_INCLUDEDIR "/opt/boost/include")
       #set(BOOST_LIBRARYDIR "/opt/boost/lib")
    endif()
endif()
# otherwise Mac dynamic lib doesn't work
if (APPLE OR WIN32)
    set(Boost_USE_STATIC_LIBS TRUE)
endif()

find_package(Boost COMPONENTS 
    program_options
    thread
    system
    chrono
    date_time
    REQUIRED)

# SYSTEM arg tells compiler that the path contains system files. Compiler will thus ignore warning. 
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

function(add_boost target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} ${Boost_LIBRARIES})
endfunction()

function(link_boost target)
    target_link_libraries(${target} ${Boost_LIBRARIES})
endfunction()
