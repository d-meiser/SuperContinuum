set(TESTS
    FFT
    FD
    jacobian
   )

include_directories(
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    ${CGREEN_INCLUDE_DIR}
    )

foreach(t ${TESTS})
  add_executable(test_${t} test_${t}.c)
  target_link_libraries(test_${t}
      sc_lib ${CGREEN_LIB} ${PETSC_LIB} ${PETSC_PACKAGE_LIBS})
  add_test(${t} test_${t})
endforeach()

