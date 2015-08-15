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
#include <jacobian.h>
#include <petscdmda.h>
#include <fd.h>

#ifndef SQR
#define SQR(a) ((a) * (a))
#endif

struct Field {
  PetscScalar u;
  PetscScalar v;
};

static PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J);
static PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J);

PetscErrorCode scJacobianBuildConstantPart(DM da, Mat J, PetscBool fourthOrder)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fourthOrder) {
    ierr = buildConstantPartOfJacobianFourthOrder(da, J);CHKERRQ(ierr);
  } else {
    ierr = buildConstantPartOfJacobian(da, J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J)
{
  PetscErrorCode ierr;
  PetscInt       i, Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v, hx;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);
  ierr = scFdAddSecondDerivative(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivative(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivative(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

  /* Auxiliary variable relation */
  v = -1.0;
  row.c = 0;
  col.c = 1;
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    ierr=MatSetValuesStencil(J,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr=MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J)
{
  PetscErrorCode ierr;
  PetscInt       i, Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v, hx;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);
  ierr = scFdAddSecondDerivativeFourthOrder(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivativeFourthOrder(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivativeFourthOrder(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

  /* Auxiliary variable relation */
  v = -1.0;
  row.c = 0;
  col.c = 1;
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    ierr=MatSetValuesStencil(J,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr=MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianBuild(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, struct JacobianCtx *ctx)
{
  PetscFunctionBegin;
  ctx->alpha = a;
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianBuildPre(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat Jpre, struct JacobianCtx *ctx)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;
  struct Field   *x;
  PetscInt       i, c;
  PetscScalar    v;
  MatStencil     col = {0}, row = {0};

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  ierr = MatRetrieveValues(Jpre);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Non-linear term */
  row.c = 0;
  col.c = 0;
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    v = -3.0 * ctx->problem->gamma * SQR(x[i].u);
    ierr=MatSetValuesStencil(Jpre,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);

  /* Time derivative terms */
  v = a;
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    col.i = i;
    for (c = 0; c < 2; ++c) {
      col.c = c;
      ierr=MatSetValuesStencil(Jpre,1,&col,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr=MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
