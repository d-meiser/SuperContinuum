# Use PETSC_DIR and PETSC_ARCH to specify location of PETSc
# When successful this defines
# PETSC_FOUND
# PETSC_LIB
# PETSC_INCLUDE_DIR
include(${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/PETScConfig.cmake)
find_path(PETSC_INCLUDE_DIR petsc.h
    PATHS ${PETSC_PACKAGE_INCLUDES} ${PETSC_DIR}/include)
find_library(PETSC_LIB petsc
    PATHS ${PETSC_DIR}/${PETSC_ARCH}
    PATH_SUFFIXES lib)
if (PETSC_LIB AND PETSC_INCLUDE_DIR)
  set(PETSC_FOUND TRUE)
endif ()

