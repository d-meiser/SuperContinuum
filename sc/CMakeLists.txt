include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    )

add_executable(sc sc.c)
target_link_libraries(sc ${PETSC_LIB} ${PETSC_PACKAGE_LIBS})

