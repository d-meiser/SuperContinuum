#include <cgreen/cgreen.h>
#include <fft.h>
#include <petscdmda.h>

static const char help[] = "Unit tests for FFT methods.";

Ensure(fft_can_be_constructed_from_DMDA)
{
  PetscErrorCode ierr;
  NppFft fft;
  DM da;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);
  assert_that(ierr, is_equal_to(0));
  ierr = nppFftCreate(da, &fft);
  assert_that(ierr, is_equal_to(0));
  nppFftDestroy(&fft);
  ierr = DMDestroy(&da);
  assert_that(ierr, is_equal_to(0));
}

Ensure(the_right_dm_gets_registered_with_fft)
{
  PetscErrorCode ierr;
  NppFft fft;
  DM da, da1;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,20,2,1,NULL,&da);
  assert_that(ierr, is_equal_to(0));
  ierr = nppFftCreate(da, &fft);
  assert_that(ierr, is_equal_to(0));
  ierr = nppFftGetDM(fft, &da1);
  assert_that(ierr, is_equal_to(0));
  assert_that(da1, is_equal_to(da));
  nppFftDestroy(&fft);
  ierr = DMDestroy(&da);
  assert_that(ierr, is_equal_to(0));
}

Ensure(forward_transform_yields_constant_vector_from_delta_function)
{
  NppFft fft;
  DM da;
  Vec v;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,2,1,NULL,&da);

  DMGetGlobalVector(da, &v);
  VecZeroEntries(v); 
  VecSetValue(v, 0, 1.0, INSERT_VALUES);
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);
  VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  DMRestoreGlobalVector(da, &v);
  nppFftCreate(da, &fft);
  nppFftDestroy(&fft);
}

Ensure(output_vector_has_correct_size)
{
  NppFft fft;
  DM da;
  Vec x, y, z;
  PetscInt dim;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,10,2,1,NULL,&da);
  nppFftCreate(da, &fft);
  nppFftCreateVecsFFTW(fft, &x, &y, &z);
  VecGetSize(x, &dim);
  assert_that(dim, is_greater_than(3));
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  nppFftDestroy(&fft);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, fft_can_be_constructed_from_DMDA);
  add_test(suite, the_right_dm_gets_registered_with_fft);
  add_test(suite, forward_transform_yields_constant_vector_from_delta_function);
  add_test(suite, output_vector_has_correct_size);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
