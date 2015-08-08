# Use CGREEN_ROOT_DIR to specify location of cgreen
# When successful this defines
# CGREEN_FOUND
# CGREEN_LIB
# CGREEN_INCLUDE_DIR

find_path(CGREEN_INCLUDE_DIR cgreen/cgreen.h
    HINTS ${CGREEN_ROOT_DIR}/include;/usr/include;/usr/local/include)
find_library(CGREEN_LIB cgreen
    HINTS ${CGREEN_ROOT_DIR}/lib64;/usr/local/lib;/usr/lib;/usr/lib64)
if (CGREEN_INCLUDE_DIR AND CGREEN_LIB)
  set(CGREEN_FOUND TRUE)
endif ()
