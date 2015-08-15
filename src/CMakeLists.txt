include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    )
set(SC_LIB_SRCS
    fft.c
    fd.c
    jacobian.c
   )
add_library(sc_lib ${SC_LIB_SRCS})
set_target_properties(sc_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
            VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
            )
install(TARGETS sc_lib DESTINATION lib)

