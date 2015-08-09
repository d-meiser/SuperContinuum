#!/bin/sh
PETSC_TAR_BALL=petsc-lite-3.6.1.tar.gz
PETSC_RELEASE_URL=http://ftp.mcs.anl.gov/pub/petsc/release-snapshots
PETSC_ARCH=${PETSC_ARCH:-double-dbg}
PETSC_DIR=`pwd`/petsc-3.6.1

WITH_X=${WITH_X:-0}
WITH_DEBUGGING=${WITH_DEBUGGING:-0}

if [ ! -e $PETSC_DIR/configure ]; then
  echo "$PETSC_DIR/configure not found - downloading PETSc tarball it."
  wget $PETSC_RELEASE_URL/$PETSC_TAR_BALL
  tar xfz $PETSC_TAR_BALL
else
  echo "$PETSC_DIR/configure found - assuming we already have PETSc source."
fi

PETSC_LIBRARY=$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so
if [ ! -e $PETSC_LIBRARY ]; then
  echo "$PETSC_LIBRARY not found - reconfiguring and building."
  cd petsc-3.6.1
  PETSC_ARCH=$PETSC_ARCH PETSC_DIR=$PETSC_DIR ./configure `pwd`\
    --with-x=$WITH_X \
    --with-debugging=$WITH_DEBUGGING \
    --with-ssl=0 \
    --with-fortran-kernels=0 \
    --with-pthread=0 \
    --download-fftw=1 \
    --with-mpi=1 \
    --download-mpich=1 \
    --with-mpiuni-fortran-binding=0 \
    --with-fortran-interfaces
  make PETSC_DIR=`pwd` PETSC_ARCH=$PETSC_ARCH all
  cd -
else
  echo "$PETSC_LIBRARY found - skipping configuration and build."
fi

