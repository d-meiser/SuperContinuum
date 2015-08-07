#ifndef FFT_H
#define FFT_H

#include <petscdm.h>
#include <petscmat.h>
#include <ScExport.h>


struct ScFft_;
typedef struct ScFft_* ScFft;

SC_API PetscErrorCode scFftCreate(DM da, ScFft *fft);
SC_API PetscErrorCode scFftDestroy(ScFft *fft);
SC_API PetscErrorCode scFftGetDM(ScFft fft, DM *da);
SC_API PetscErrorCode scFftTransform(Vec v, PetscInt i, Vec y);
SC_API PetscErrorCode scFftCreateVecsFFTW(ScFft fft, Vec *x, Vec *y, Vec *z);

#endif
