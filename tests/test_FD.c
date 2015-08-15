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
#include <test_utils.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>

static const char help[] = "Unit tests for FD methods.";

Ensure(first_derivative_of_constant_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddFirstDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecSet(fixture.x, 23.0);CHKERRV(ierr);
  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  PetscScalar nrm;
  ierr = VecStrideNorm(fixture.y, 0, NORM_INFINITY, &nrm);CHKERRV(ierr);
  assert_that(nrm < 1.0e-6, is_true);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(first_derivative_fourth_order_of_constant_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddFirstDerivativeFourthOrder(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecSet(fixture.x, 23.0);CHKERRV(ierr);
  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  PetscScalar nrm;
  ierr = VecStrideNorm(fixture.y, 0, NORM_INFINITY, &nrm);CHKERRV(ierr);
  assert_that(nrm < 1.0e-6, is_true);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(first_derivative_of_linear_function_is_computed_exactly)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  PetscInt i;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddFirstDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);

  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecGetArray(fixture.x, &fixture.xarr);CHKERRV(ierr);
  for (i = fixture.info.xs; i < fixture.info.xs + fixture.info.xm; ++i) {
    fixture.xarr[i * fixture.numComponents] = i * 0.49;
  }
  ierr = VecRestoreArray(fixture.x, &fixture.xarr);CHKERRV(ierr);

  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  ierr = VecGetArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);
  for (i = fixture.info.xs; i < fixture.info.xs + fixture.info.xm; ++i) {
    PetscScalar err;
    if (i == 0) {
      err = fabs(3.3 * (0.49 - 0.49 * (fixture.dim - 1)) / 2.0 / 0.1 - fixture.yarr[i * fixture.numComponents]);
    } else if (i == fixture.dim - 1) {
      err = fabs(3.3 * (0.0 - (i - 1) * 0.49) / 2.0 / 0.1 - fixture.yarr[i * fixture.numComponents]);
    } else {
      err = fabs(3.3 * 0.49 / 0.1 - fixture.yarr[i * fixture.numComponents]);
    }
    assert_that(err < 1.0e-6, is_true);
  }
  ierr = VecRestoreArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(first_derivative_fourth_order_of_linear_function_is_computed_exactly)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  PetscInt i;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddFirstDerivativeFourthOrder(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);

  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecGetArray(fixture.x, &fixture.xarr);CHKERRV(ierr);
  for (i = fixture.info.xs; i < fixture.info.xs + fixture.info.xm; ++i) {
    fixture.xarr[i * fixture.numComponents] = i * 0.49;
  }
  ierr = VecRestoreArray(fixture.x, &fixture.xarr);CHKERRV(ierr);

  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  ierr = VecGetArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);
  i = 2;
  PetscScalar err = 3.3 * 0.49 / 0.1 - fixture.yarr[i * fixture.numComponents];
  assert_that(err < 1.0e-6, is_true);
  ierr = VecRestoreArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(second_derivative_of_constant_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddSecondDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecSet(fixture.x, 23.0);CHKERRV(ierr);
  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  PetscScalar nrm;
  ierr = VecStrideNorm(fixture.y, 0, NORM_INFINITY, &nrm);CHKERRV(ierr);
  assert_that(nrm < 1.0e-6, is_true);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(second_derivative_fourth_order_of_constant_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddSecondDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecSet(fixture.x, 23.0);CHKERRV(ierr);
  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  PetscScalar nrm;
  ierr = VecStrideNorm(fixture.y, 0, NORM_INFINITY, &nrm);CHKERRV(ierr);
  assert_that(nrm < 1.0e-6, is_true);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(second_derivative_of_linear_function_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  PetscInt i;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddSecondDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecGetArray(fixture.x, &fixture.xarr);CHKERRV(ierr);
  for (i = fixture.info.xs; i < fixture.info.xs + fixture.info.xm; ++i) {
    fixture.xarr[i * fixture.numComponents] = i * 0.49;
  }
  ierr = VecRestoreArray(fixture.x, &fixture.xarr);CHKERRV(ierr);

  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  ierr = VecGetArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);
  i = 2;
  PetscScalar err = fabs(0.0 - fixture.yarr[i * fixture.numComponents]);
  assert_that(err < 1.0e-6, is_true);
  ierr = VecRestoreArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(second_derivative_fourth_order_of_linear_function_is_zero)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  PetscInt i;

  ierr = scMatSetup(&fixture);CHKERRV(ierr);

  ierr = scFdAddSecondDerivative(fixture.da, fixture.m, 3.3, 0.1, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(fixture.m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(fixture.m, MAT_FINAL_ASSEMBLY);

  ierr = VecGetArray(fixture.x, &fixture.xarr);CHKERRV(ierr);
  for (i = fixture.info.xs; i < fixture.info.xs + fixture.info.xm; ++i) {
    fixture.xarr[i * fixture.numComponents] = i * 0.49;
  }
  ierr = VecRestoreArray(fixture.x, &fixture.xarr);CHKERRV(ierr);

  ierr = MatMult(fixture.m, fixture.x, fixture.y);CHKERRV(ierr);

  ierr = VecGetArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);
  i = 2;
  PetscScalar err = fabs(0.0 - fixture.yarr[i * fixture.numComponents]);
  assert_that(err < 1.0e-6, is_true);
  ierr = VecRestoreArrayRead(fixture.y, (const PetscScalar**)&fixture.yarr);CHKERRV(ierr);

  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}


int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, first_derivative_of_constant_is_zero);
  add_test(suite, first_derivative_fourth_order_of_constant_is_zero);
  add_test(suite, first_derivative_of_linear_function_is_computed_exactly);
  add_test(suite, first_derivative_fourth_order_of_linear_function_is_computed_exactly);
  add_test(suite, second_derivative_of_constant_is_zero);
  add_test(suite, second_derivative_fourth_order_of_constant_is_zero);
  add_test(suite, second_derivative_of_linear_function_is_zero);
  add_test(suite, second_derivative_fourth_order_of_linear_function_is_zero);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
