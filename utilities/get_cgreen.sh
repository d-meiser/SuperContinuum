#!/bin/sh

if [ ! -d cgreen-code/cgreen ]; then
  svn checkout svn://svn.code.sf.net/p/cgreen/code/trunk cgreen-code
else
  cd cgreen-code
  svn up
  cd -
fi
cd cgreen-code
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:BOOL=../../cgreen ../cgreen
make -j2
make install


