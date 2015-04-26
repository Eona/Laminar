if (NOT DEFINED EIGEN3_INCLUDE_DIR)
    if (APPLE OR UNIX)
        set(EIGEN3_INCLUDE_DIR "/opt/Eigen")
    endif()
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
