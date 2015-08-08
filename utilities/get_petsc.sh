#!/bin/sh
PETSC_TAR_BALL=petsc-lite-3.6.1.tar.gz
PETSC_RELEASE_URL=http://ftp.mcs.anl.gov/pub/petsc/release-snapshots
PETSC_ARCH=double-dbg
PETSC_DIR=`pwd`/petsc-3.6.1

if [ ! -f $PETSC_TAR_BALL ]; then
  echo "PETSc tarball not found - downloading it."
  wget $PETSC_RELEASE_URL/$PETSC_TAR_BALL
else
  echo "PETSc tarball found."
fi

if [ ! -d petsc-3.6.1 ]; then
  echo "PETSc directory no present - extracting tarball."
  tar xfz $PETSC_TAR_BALL
fi

PETSC_LIBRARY=$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so
if [ ! -e $PETSC_LIBRARY ]; then
  echo "$PETSC_LIBRARY not found - reconfiguring and building."
  cd petsc-3.6.1
  PETSC_ARCH=$PETSC_ARCH PETSC_DIR=$PETSC_DIR ./configure `pwd`\
    --with-x=0 \
    --with-ssl=0 \
    --with-fortran-kernels=0 \
    --with-pthread=0 \
    --download-fftw=1 \
    --with-mpi=1 \
    --download-mpich=1 \
    --with-mpiuni-fortran-binding=0 \
    --with-fortran-interfaces
  make PETSC_DIR=`pwd` PETSC_ARCH=double-dbg all
  cd -
else
  echo "$PETSC_LIBRARY found - skipping configuration and build."
fi

