#ifndef FFT_H
#define FFT_H

#include <petscdm.h>
#include <petscmat.h>

namespace npp {
  Mat createFFTFromDMDA(DM da);
};

#endif
