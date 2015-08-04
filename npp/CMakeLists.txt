include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PETSC_PACKAGE_INCLUDES}
    )

add_executable(npp npp.c)
target_link_libraries(npp ${PETSC_PACKAGE_LIBS})

