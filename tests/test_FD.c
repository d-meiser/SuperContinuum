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
#include <cgreen/cgreen.h>
#include <fd.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>

static const char help[] = "Unit tests for FD methods.";

struct FDFixture {
  PetscInt dim;
  PetscInt numComponents;
  DM da;
  Mat m;
  Vec x;
  Vec y;
  PetscScalar *xarr;
  PetscScalar *yarr;
};

static PetscErrorCode setup(struct FDFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fixture->dim = 5;
  fixture->numComponents = 2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, fixture->dim, fixture->numComponents, 1, NULL, &fixture->da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(fixture->da, &fixture->m);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fixture->da, &fixture->x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fixture->da, &fixture->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode teardown(struct FDFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDestroy(&fixture->da);CHKERRQ(ierr);
  ierr = MatDestroy(&fixture->m);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->x);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

Ensure(first_derivative)
{
  struct FDFixture fixture;
  PetscErrorCode ierr;
  ierr = setup(&fixture);CHKERRV(ierr);
  ierr = scFdAddFirstDerivative(fixture.da, fixture.m, 1.0, 1.0, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(fixture.m, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(ierr);
  ierr = teardown(&fixture);CHKERRV(ierr);
}

Ensure(second_derivative)
{
  struct FDFixture fixture;
  PetscErrorCode ierr;
  ierr = setup(&fixture);CHKERRV(ierr);
  ierr = scFdAddSecondDerivative(fixture.da, fixture.m, 1.0, 1.0, 1, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(fixture.m, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(ierr);
  ierr = teardown(&fixture);CHKERRV(ierr);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, first_derivative);
  add_test(suite, second_derivative);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
