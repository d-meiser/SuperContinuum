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
