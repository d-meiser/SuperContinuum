#ifndef FFT_H
#define FFT_H

#include <petscdm.h>
#include <petscmat.h>
#include <NppExport.h>


struct NppFft_;
typedef struct NppFft_* NppFft;

NPP_API NppFft nppCreateFftFromDMDA(DM da);

#endif
