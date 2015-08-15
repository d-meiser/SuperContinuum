/*
Copyright 2015 Dominic Meiser

This file is part of SuperContinuum.

SuperContinuum is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your
option) any later version.

SuperContinuum is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along
with SuperContinuum.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef FFT_H
#define FFT_H

#include <petscdm.h>
#include <petscmat.h>
#include <ScExport.h>
#include <basic_types.h>


struct ScFft_;
typedef struct ScFft_* ScFft;

SC_API PetscErrorCode scFftCreate(DM da, ScFft *fft);
SC_API PetscErrorCode scFftDestroy(ScFft *fft);
SC_API PetscErrorCode scFftGetDM(ScFft fft, DM *da);
SC_API PetscErrorCode scFftTransform(ScFft fft, Vec v, PetscInt i, Vec y);
SC_API PetscErrorCode scFftITransform(ScFft fft, Vec v, PetscInt i, Vec y);
SC_API PetscErrorCode scFftCreateVecsFFTW(ScFft fft, Vec *x, Vec *y, Vec *z);
SC_API PetscErrorCode scFftCreateVecPSD(ScFft fft, Vec *psd);
SC_API PetscErrorCode scFftComputePSD(ScFft fft, Vec v, PetscInt component, Vec work, Vec psd, PetscBool logScale);

#endif
