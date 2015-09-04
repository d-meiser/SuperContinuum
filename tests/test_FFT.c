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


Describe(FFT);

static PetscErrorCode ierr;
static ScFft fft;
static DM da;

BeforeEach(FFT)
{
  significant_figures_for_assert_double_are(6);
}

AfterEach(FFT) {}

Ensure(FFT, can_be_constructed_from_DMDA)
{
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);CHKERRV(ierr);
  ierr = scFftCreate(da, &fft);
  assert_that(ierr, is_equal_to(0));
  scFftDestroy(&fft);CHKERRV(ierr);
  ierr = DMDestroy(&da);CHKERRV(ierr);
}

Ensure(FFT, registers_the_right_DM)
{
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);CHKERRV(ierr);
  ierr = scFftCreate(da, &fft);CHKERRV(ierr);
  DM da1;
  ierr = scFftGetDM(fft, &da1);CHKERRV(ierr);
  assert_that(da1, is_equal_to(da));
  ierr = scFftDestroy(&fft);CHKERRV(ierr);
  ierr = DMDestroy(&da);CHKERRV(ierr);
}

Ensure(FFT, transforms_constant_into_delta_function)
{
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
  //assert_that_double(fabs(arr[0] - 0.7), is_less_than_double(1.0e-6));
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

Ensure(FFT, can_transform_second_component)
{
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

Ensure(FFT, i_transform_is_inverse_of_transform)
{
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

Ensure(FFT, PSD_of_delta_function_is_flat)
{
  Vec v;

  PetscInt dim = 10;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,dim,2,1,NULL,&da);

  DMCreateGlobalVector(da, &v);
  VecZeroEntries(v); 
  PetscScalar val = 2.8;
  VecSetValue(v, 0, val, INSERT_VALUES);
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  scFftCreate(da, &fft);

  Vec x, y, z;
  scFftCreateVecsFFTW(fft, &x, &y, &z);

  Vec psd;
  scFftCreateVecPSD(fft, &psd);

  scFftComputePSD(fft, v, 0, y, psd, PETSC_FALSE);
  const PetscScalar *arr;
  VecGetArrayRead(psd, &arr);
  PetscInt i;
  for (i = 0; i < dim / 2; ++i) {
    assert_that(fabs(arr[i] - val * val) < 1.0e-6, is_true);
  }
  VecRestoreArrayRead(psd, &arr);

  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  VecDestroy(&v);
  VecDestroy(&psd);
  scFftDestroy(&fft);
  DMDestroy(&da);
}

Ensure(FFT, creates_output_vector_of_correct_size)
{
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

Ensure(FFT, yields_PSD_of_the_correct_size)
{
  PetscInt inputDim = 10;
  PetscInt inputBlockSize = 3;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,inputDim,inputBlockSize,1,NULL,&da);
  scFftCreate(da, &fft);
  Vec x;
  scFftCreateVecPSD(fft, &x);
  PetscInt dim;
  VecGetSize(x, &dim);
  assert_that(dim, is_equal_to(inputDim / 2));
  VecDestroy(&x);
  scFftDestroy(&fft);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test_with_context(suite, FFT, can_be_constructed_from_DMDA);
  add_test_with_context(suite, FFT, registers_the_right_DM);
  add_test_with_context(suite, FFT, creates_output_vector_of_correct_size);
  add_test_with_context(suite, FFT, transforms_constant_into_delta_function);
  add_test_with_context(suite, FFT, can_transform_second_component);
  add_test_with_context(suite, FFT, i_transform_is_inverse_of_transform);
  add_test_with_context(suite, FFT, PSD_of_delta_function_is_flat);
  add_test_with_context(suite, FFT, yields_PSD_of_the_correct_size);
  int result;
  if (argc > 2) {
    result = run_single_test(suite, argv[1], create_text_reporter());
  } else {
    result = run_test_suite(suite, create_text_reporter());
  }
  destroy_test_suite(suite);
  PetscFinalize();
  return result;
}
