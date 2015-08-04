include_directories(${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR})
set(NPP_LIB_SRCS
    fft.cpp
   )
add_library(npp_lib ${NPP_LIB_SRCS})
install(TARGETS npp_lib DESTINATION lib)
