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
#include <petscdmda.h>

static const char help[] = "Unit tests for Jacobian construction methods.";

Ensure(can_build_jacobian)
{
  struct FDFixture fixture;
  PetscErrorCode ierr;
  ierr = scFdSetup(&fixture);CHKERRV(ierr);
  ierr = scJacobianBuildConstantPart(fixture.da, fixture.m, PETSC_FALSE);CHKERRV(ierr);
  assert_that(ierr, is_equal_to(0));
  ierr = scFdTeardown(&fixture);CHKERRV(ierr);
}

Ensure(can_build_jacobian_fourth_order)
{
  struct FDFixture fixture;
  PetscErrorCode ierr;
  ierr = scFdSetup(&fixture);CHKERRV(ierr);
  ierr = scJacobianBuildConstantPart(fixture.da, fixture.m, PETSC_TRUE);CHKERRV(ierr);
  assert_that(ierr, is_equal_to(0));
  ierr = scFdTeardown(&fixture);CHKERRV(ierr);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, can_build_jacobian);
  add_test(suite, can_build_jacobian_fourth_order);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
