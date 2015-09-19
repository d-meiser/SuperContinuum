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
#include <jacobian.h>
#include <test_utils.h>
#include <jacobian_test_utils.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

static const char help[] = "Unit tests for Jacobian construction methods.";


Describe(SecondOrderJacobian);

struct JacobianFixture fixture;
PetscErrorCode ierr;

BeforeEach(SecondOrderJacobian) {
  jacobianSetup(&fixture, 0, 0);
}
AfterEach(SecondOrderJacobian) {
  jacobianTeardown(&fixture);
}

Ensure(SecondOrderJacobian, can_be_built) {
  ierr = scJacobianBuildLinearPart(fixture.matFixture.da, fixture.matFixture.m, PETSC_FALSE);CHKERRV(ierr);
  assert_that(ierr, is_equal_to(0));
}

Ensure(SecondOrderJacobian, is_consistent_for_constant_fields) {
  ierr = checkJacobianPreConsistency(&fixture, waveAtRest, constant_func, 0.0, 0.0, PETSC_FALSE, 1.0e-6, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_sine_waves) {
  ierr = checkJacobianPreConsistency(&fixture, waveAtRest, sine_wave, 0.0, 0.0, PETSC_FALSE, 1.0e-2, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_gaussian) {
  ierr = checkJacobianPreConsistency(&fixture, waveAtRest, gaussian, 0.0, 0.0, PETSC_FALSE, 1.0e-1, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_constant_right_moving) {
  ierr = checkJacobianPreConsistency(&fixture, rightMovingWave, constant_func, 0.0, 0.0, PETSC_FALSE, 1.0e-6, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_sine_wave_right_moving) {
  ierr = checkJacobianPreConsistency(&fixture, rightMovingWave, sine_wave, 0.0, 0.0, PETSC_FALSE, 1.0e0, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_gaussian_right_moving) {
  ierr = checkJacobianPreConsistency(&fixture, rightMovingWave, gaussian, 0.0, 0.0, PETSC_FALSE, 2.0e0, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_sine_wave_non_zero_alpha) {
  ierr = checkJacobianPreConsistency(&fixture, waveAtRest, sine_wave, 1.0, 0.0, PETSC_FALSE, 1.0e-2, PETSC_FALSE);CHKERRV(ierr);
}

Ensure(SecondOrderJacobian, is_consistent_for_sine_wave_non_zero_gamma) {
  ierr = checkJacobianPreConsistency(&fixture, waveAtRest, sine_wave, 0.0, 1.0, PETSC_FALSE, 1.0e-2, PETSC_TRUE);CHKERRV(ierr);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test_with_context(suite, SecondOrderJacobian, can_be_built);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_constant_fields);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_sine_waves);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_gaussian);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_constant_right_moving);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_sine_wave_right_moving);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_gaussian_right_moving);
  add_test_with_context(suite, SecondOrderJacobian, is_consistent_for_sine_wave_non_zero_alpha);
  /* Disable this test for now, need the right Chi3 physics
  add_test(suite, prec_consistent_sine_wave_non_zero_gamma);
  */
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

