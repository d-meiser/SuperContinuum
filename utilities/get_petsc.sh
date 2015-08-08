#!/bin/sh
PETSC_TAR_BALL=petsc-lite-3.6.1.tar.gz
PETSC_RELEASE_URL=http://ftp.mcs.anl.gov/pub/petsc/release-snapshots
if [ ! -f $PETSC_TAR_BALL ]; then
  wget $PETSC_RELEASE_URL/$PETSC_TAR_BALL
fi
if [ ! -d petsc-3.6.1 ]; then
  tar xfz $PETSC_TAR_BALL
fi
cd petsc-3.6.1
PETSC_ARCH=double-dbg PETSC_DIR=`pwd` ./configure \
  --with-mpi=0 \
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

