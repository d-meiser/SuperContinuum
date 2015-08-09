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
#include <fft.h>
#include <petscdmda.h>

static const char help[] = "Unit tests for FFT methods.";

Ensure(fft_can_be_constructed_from_DMDA)
{
  PetscErrorCode ierr;
  ScFft fft;
  DM da;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);
  assert_that(ierr, is_equal_to(0));
  ierr = scFftCreate(da, &fft);
  assert_that(ierr, is_equal_to(0));
  scFftDestroy(&fft);
  ierr = DMDestroy(&da);
  assert_that(ierr, is_equal_to(0));
}

Ensure(the_right_dm_gets_registered_with_fft)
{
  PetscErrorCode ierr;
  ScFft fft;
  DM da, da1;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);
  assert_that(ierr, is_equal_to(0));
  ierr = scFftCreate(da, &fft);
  assert_that(ierr, is_equal_to(0));
  ierr = scFftGetDM(fft, &da1);
  assert_that(ierr, is_equal_to(0));
  assert_that(da1, is_equal_to(da));
  scFftDestroy(&fft);
  ierr = DMDestroy(&da);
  assert_that(ierr, is_equal_to(0));
}

Ensure(forward_transform_yields_constant_vector_from_delta_function)
{
  ScFft fft;
  DM da;
  Vec v;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,2,1,NULL,&da);

  DMCreateGlobalVector(da, &v);
  VecZeroEntries(v); 
  VecSetValue(v, 0, 0.7, INSERT_VALUES);
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  scFftCreate(da, &fft);

  Vec x, y, z;
  scFftCreateVecsFFTW(fft, &x, &y, &z);

  scFftTransform(fft, v, 0, y);
  const PetscScalar *arr;
  VecGetArrayRead(y, &arr);
  assert_that(fabs(arr[0] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[1] - 0.0) < 1.0e-6, is_true);
  assert_that(fabs(arr[1] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[2] - 0.0) < 1.0e-6, is_true);
  assert_that(fabs(arr[3] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[4] - 0.0) < 1.0e-6, is_true);
  VecRestoreArrayRead(y, &arr);

  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  VecDestroy(&v);
  scFftDestroy(&fft);
  DMDestroy(&da);
}

Ensure(second_component_transforms_ok)
{
  ScFft fft;
  DM da;
  Vec v;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,2,1,NULL,&da);

  DMCreateGlobalVector(da, &v);
  VecZeroEntries(v); 
  VecSetValue(v, 1, 0.7, INSERT_VALUES);
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  scFftCreate(da, &fft);

  Vec x, y, z;
  scFftCreateVecsFFTW(fft, &x, &y, &z);

  scFftTransform(fft, v, 1, y);
  const PetscScalar *arr;
  VecGetArrayRead(y, &arr);
  assert_that(fabs(arr[0] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[1] - 0.0) < 1.0e-6, is_true);
  assert_that(fabs(arr[1] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[2] - 0.0) < 1.0e-6, is_true);
  assert_that(fabs(arr[3] - 0.7) < 1.0e-6, is_true);
  assert_that(fabs(arr[4] - 0.0) < 1.0e-6, is_true);
  VecRestoreArrayRead(y, &arr);

  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  VecDestroy(&v);
  scFftDestroy(&fft);
  DMDestroy(&da);
}

Ensure(i_transform_is_inverse_of_transform)
{
  ScFft fft;
  DM da;
  Vec v;
  PetscInt dim = 5;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,dim,2,1,NULL,&da);

  DMCreateGlobalVector(da, &v);
  PetscRandom rdm;
  PetscRandomCreate(PETSC_COMM_WORLD, &rdm);
  PetscRandomSetFromOptions(rdm);
  VecSetRandom(v, rdm);
  PetscRandomDestroy(&rdm);
  Vec vIn;
  VecDuplicate(v, &vIn);
  VecCopy(v, vIn);

  scFftCreate(da, &fft);

  Vec x, y, z;
  scFftCreateVecsFFTW(fft, &x, &y, &z);

  int component = 0;
  scFftTransform(fft, v, component, y);
  scFftITransform(fft, v, component, y);
  VecAXPY(v, -dim, vIn);
  PetscReal error;
  VecStrideNorm(v, component, NORM_INFINITY, &error);
  assert_that(error < 1.0e-6, is_true);

  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  VecDestroy(&v);
  VecDestroy(&vIn);
  scFftDestroy(&fft);
  DMDestroy(&da);
}

Ensure(output_vector_has_correct_size)
{
  ScFft fft;
  DM da;
  Vec x, y, z;
  PetscInt inputDim = 10;
  PetscInt inputBlockSize = 3;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,inputDim,inputBlockSize,1,NULL,&da);
  scFftCreate(da, &fft);
  scFftCreateVecsFFTW(fft, &x, &y, &z);
  PetscInt dim = 10;
  VecGetSize(x, &dim);
  assert_that(dim, is_greater_than(inputDim));
  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  scFftDestroy(&fft);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, fft_can_be_constructed_from_DMDA);
  add_test(suite, the_right_dm_gets_registered_with_fft);
  add_test(suite, output_vector_has_correct_size);
  add_test(suite, forward_transform_yields_constant_vector_from_delta_function);
  add_test(suite, second_component_transforms_ok);
  add_test(suite, i_transform_is_inverse_of_transform);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
