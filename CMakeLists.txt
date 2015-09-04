cmake_minimum_required(VERSION 2.6.2)
project(SuperContinuum)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_TWEAK 0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
enable_testing()
include(CheckCCompilerFlag)
include(GenerateExportHeader)

option(SC_BUILD_TESTS "Whether to build tests" OFF)
option(SC_ENABLE_COVERAGE "Instrument tests for coverage" OFF)
option(SC_WITH_ADDRESS_SANITIZER "Whether to build with address sanitizer enabled" OFF)

find_package(PETSc REQUIRED)

if(SC_WITH_ADDRESS_SANITIZER)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
  CHECK_C_COMPILER_FLAG(-fsanitize=address HAVE_ADDRESS_SANITIZER)
  if(HAVE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
  endif()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif()

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
  if(SC_ENABLE_COVERAGE)
    add_definitions("-ftest-coverage -fprofile-arcs")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftest-coverage -fprofile-arcs")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ftest-coverage -fprofile-arcs")
    set(COV_LIBRARIES gcov)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif()


add_subdirectory(src)
generate_export_header(sc_lib
    EXPORT_FILE_NAME ScExport.h
    BASE_NAME sc
    EXPORT_MACRO_NAME SC_API
    )
add_subdirectory(include)
add_subdirectory(sc)
if(SC_BUILD_TESTS)
  find_package(Cgreen REQUIRED)
  add_subdirectory(tests)
endif()

