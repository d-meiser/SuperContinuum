include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    )
set(SC_LIB_SRCS
    fft.c
   )
add_library(sc_lib ${SC_LIB_SRCS})
install(TARGETS sc_lib DESTINATION lib)

