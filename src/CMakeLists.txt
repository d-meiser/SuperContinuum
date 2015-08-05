include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    )
set(NPP_LIB_SRCS
    fft.cpp
   )
add_library(npp_lib ${NPP_LIB_SRCS})
install(TARGETS npp_lib DESTINATION lib)
generate_export_header(npp_lib
    EXPORT_FILE_NAME NppExport.h
    BASE_NAME npp
    EXPORT_MACRO_NAME NPP_API
    )

