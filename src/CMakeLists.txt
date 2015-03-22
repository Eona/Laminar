message("Folder src/")

set(MAIN
    cadenza)

set(SOURCES 
    ${MAIN}.cu)

cuda_add_executable(${MAIN} ${SOURCES})

# ==== boost ====
#   include(eona_boost)
#   link_boost(${MAIN} ${Boost_LIBRARIES})
#   add_boost(boosttest boosttest.cpp)