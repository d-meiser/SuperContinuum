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
#include <petsc.h>
#include <jacobian_test_utils.h>

static const char help[] = "Unit tests for Jacobian construction methods.";

Describe(Jacobian);

struct JacobianFixture fixture;
PetscErrorCode ierr;

BeforeEach(Jacobian) {
  jacobianSetup(&fixture, 0, 0);
}
AfterEach(Jacobian) {
  jacobianTeardown(&fixture);
}

static PetscErrorCode fillVector(FillerMethod filler, WaveForm function, Vec x);

Ensure(Jacobian, is_correct_in_linear_case) {
  fixture.jCtx.alpha = 0;
  fixture.problem.gamma = 0;
  Vec X = fixture.matFixture.x;
  ierr = fillVector(rightMovingWave, gaussian, X);CHKERRV(ierr);
  ierr = scJacobianBuildLinearPart(fixture.matFixture.da, fixture.Jpre, PETSC_TRUE);CHKERRV(ierr);
  ierr = MatStoreValues(fixture.Jpre);CHKERRV(ierr);
  ierr = scJacobianBuild(fixture.ts, 0.0, X, fixture.Xdot, fixture.jCtx.alpha, fixture.J, &fixture.jCtx);CHKERRV(ierr);
}


int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test_with_context(suite, Jacobian, is_correct_in_linear_case);
  int result = run_test_suite(suite, create_text_reporter());
  destroy_test_suite(suite);
  PetscFinalize();
  return result;
}

PetscErrorCode fillVector(FillerMethod filler, WaveForm function, Vec X) {
  PetscErrorCode i = filler(X, function);CHKERRQ(i);
  return 0;
}
