set(TESTS
    FFT
   )
foreach(t ${TESTS})
  add_executable(test_${t} test_${t}.cpp)
  target_link_libraries(test_${t} gtest_main gtest)
  add_test(${t} test_${t})
endforeach()

