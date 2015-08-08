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
#include <fft.h>
#include <petscdmda.h>

struct ScFft_ {
  DM da;
  Mat matFft;
  Vec work;
  Vec x;
  Vec y;
  Vec z;
};

static PetscErrorCode fftGetNumDofs(ScFft fft, PetscInt *dof);

#undef __FUNCT__
#define __FUNCT__ "scFftCreate"
PetscErrorCode scFftCreate(DM da, ScFft *fft)
{
  PetscErrorCode ierr;
  PetscInt       dim, dims[3], dof, nLocal;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!da) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Invalid da.");
  }
  ierr = DMDAGetInfo(da, &dim, &dims[0], &dims[1], &dims[2], 0, 0, 0, &dof, 0, 0, 0, 0, 0);CHKERRQ(ierr);
  *fft = malloc(sizeof(**fft));
  (*fft)->da = da;

  ierr = MatCreateFFT(PETSC_COMM_WORLD, dim, dims, MATFFTW, &(*fft)->matFft);CHKERRQ(ierr);
  ierr = MatSetType((*fft)->matFft, MATFFTW);CHKERRQ(ierr);
  ierr = MatSetUp((*fft)->matFft);CHKERRQ(ierr);

  ierr = MatGetLocalSize((*fft)->matFft, 0, &nLocal);
  ierr = PetscObjectGetComm((PetscObject)da, &comm);CHKERRQ(ierr);
  ierr = VecCreate(comm, &(*fft)->work);CHKERRQ(ierr);
  ierr = VecSetFromOptions((*fft)->work);CHKERRQ(ierr);
  ierr = VecSetSizes((*fft)->work, nLocal / dof, PETSC_DECIDE);CHKERRQ(ierr);

  ierr = MatCreateVecsFFTW((*fft)->matFft, &(*fft)->x, &(*fft)->y, &(*fft)->z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftDestroy"
PetscErrorCode scFftDestroy(ScFft *fft)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&(*fft)->matFft);CHKERRQ(ierr);
  ierr = VecDestroy(&(*fft)->work);CHKERRQ(ierr);
  ierr = VecDestroy(&(*fft)->x);CHKERRQ(ierr);
  ierr = VecDestroy(&(*fft)->y);CHKERRQ(ierr);
  ierr = VecDestroy(&(*fft)->z);CHKERRQ(ierr);
  free(*fft);
  *fft = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftGetDM"
PetscErrorCode scFftGetDM(ScFft fft, DM *da)
{
  PetscFunctionBegin;
  *da = fft->da;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftTransform"
PetscErrorCode scFftTransform(ScFft fft, Vec v, PetscInt i, Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecStrideGather(v, i, fft->work, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterPetscToFFTW(fft->matFft, fft->work, fft->x);CHKERRQ(ierr);
  ierr = MatMult(fft->matFft, fft->x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftITransform"
PetscErrorCode scFftITransform(ScFft fft, Vec v, PetscInt i, Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultTranspose(fft->matFft, y, fft->z);CHKERRQ(ierr);
  ierr = VecScatterFFTWToPetsc(fft->matFft, fft->z, fft->work);CHKERRQ(ierr);
  ierr = VecStrideScatter(fft->work, i, v, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftCreateVecsFFTW"
PetscErrorCode scFftCreateVecsFFTW(ScFft fft, Vec *x, Vec *y, Vec *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecsFFTW(fft->matFft, x, y, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "fftGetNumDofs"
PetscErrorCode fftGetNumDofs(ScFft fft, PetscInt *dof) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(fft->da, 0, 0, 0, 0, 0, 0, 0, dof, 0, 0, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
