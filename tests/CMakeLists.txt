set(TESTS
    FFT
   )

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PETSC_INCLUDE_DIR}
    ${PETSC_PACKAGE_INCLUDES}
    ${CGREEN_INCLUDE_DIR}
    )

foreach(t ${TESTS})
  add_executable(test_${t} test_${t}.cpp)
  target_link_libraries(test_${t} ${CGREEN_LIB})
  add_test(${t} test_${t})
endforeach()

