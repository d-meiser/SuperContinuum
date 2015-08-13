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
#include <fd.h>
#include <petscdmda.h>

#ifndef SQR
#define SQR(a) ((a) * (a))
#endif

PetscErrorCode scFdAddFirstDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[2] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[2] = {-0.5 * alpha / hx, 0.5 * alpha / hx};

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 2; ++ii) {
    col[ii].c = ccomp;
  }
  if (info.xs == 0) {
    i = 0;
    row.i = i;
    col[0].i = Mx - 1;
    col[1].i = i + 1;
    ierr = MatSetValuesStencil(m, 1, &row, 2, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    col[0].i = i - 1;
    col[1].i = 0;
    ierr = MatSetValuesStencil(m, 1, &row, 2, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    --info.xm;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    col[0].i = i - 1;
    col[1].i = i + 1;
    ierr = MatSetValuesStencil(m, 1, &row, 2, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode scFdAddFirstDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[4] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[4] = {
    1.0 * alpha / (12.0 * hx),
    -8.0 * alpha / (12.0 * hx),
    8.0 * alpha / (12.0 * hx),
    -1.0 * alpha / (12.0 * hx)
  };

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 4; ++ii) {
    col[ii].c = ccomp;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    col[0].i = clamp(i - 2, 0, Mx);
    col[1].i = clamp(i - 1, 0, Mx);
    col[2].i = clamp(i + 1, 0, Mx);
    col[3].i = clamp(i + 2, 0, Mx);
    ierr = MatSetValuesStencil(m, 1, &row, 4, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode scFdAddSecondDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[3] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[3] = {alpha / SQR(hx), -2.0 * alpha / SQR(hx), alpha / SQR(hx)};

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 3; ++ii) {
    col[ii].c = ccomp;
  }
  if (info.xs == 0) {
    i = 0;
    row.i = i;
    col[0].i = Mx - 1;
    for (ii = i + 1; ii < 3; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 3, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    for (ii = 0; ii < 2; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    col[2].i = 0;
    ierr = MatSetValuesStencil(m, 1, &row, 3, col, v, ADD_VALUES);CHKERRQ(ierr);
    --info.xm;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    for (ii = 0; ii < 3; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 3, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscInt clamp(PetscInt i, PetscInt imin, PetscInt imax)
{
  PetscInt L = imax - imin;
  if (i < imin) {
    i += L;
  }
  if (i >= imax) {
    i -= L;
  }
  return i;
}

PetscErrorCode scFdAddSecondDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[3] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[5] = {
    -alpha / (12.0 * SQR(hx)),
    16.0 * alpha /(12.0 * SQR(hx)),
    -30.0 * alpha /(12.0 * SQR(hx)),
    16.0 * alpha / (12.0 * SQR(hx)),
    -alpha / (12.0 * SQR(hx))
  };

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 5; ++ii) {
    col[ii].c = ccomp;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    for (ii = 0; ii < 5; ++ii) {
      col[ii].i = clamp(i - 2 + ii, 0, Mx);
    }
    ierr = MatSetValuesStencil(m, 1, &row, 5, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
