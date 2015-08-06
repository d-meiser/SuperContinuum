#ifndef FFT_H
#define FFT_H

#include <petscdm.h>
#include <petscmat.h>
#include <NppExport.h>


struct NppFft_;
typedef struct NppFft_* NppFft;

NPP_API PetscErrorCode nppFftCreate(DM da, NppFft *fft);
NPP_API PetscErrorCode nppFftDestroy(NppFft *fft);
NPP_API PetscErrorCode nppFftGetDM(NppFft fft, DM *da);
NPP_API PetscErrorCode nppFftTransform(Vec v, PetscInt i, Vec y);
NPP_API PetscErrorCode nppFftCreateVecsFFTW(NppFft fft, Vec *x, Vec *y, Vec *z);

#endif
