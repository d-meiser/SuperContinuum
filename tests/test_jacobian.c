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
#include <petscts.h>
#include <petscvec.h>

static const char help[] = "Unit tests for Jacobian construction methods.";

Ensure(can_build_jacobian)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  ierr = scMatSetup(&fixture);CHKERRV(ierr);
  ierr = scJacobianBuildConstantPart(fixture.da, fixture.m, PETSC_FALSE);CHKERRV(ierr);
  assert_that(ierr, is_equal_to(0));
  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

Ensure(can_build_jacobian_fourth_order)
{
  struct MatFixture fixture;
  PetscErrorCode ierr;
  ierr = scMatSetup(&fixture);CHKERRV(ierr);
  ierr = scJacobianBuildConstantPart(fixture.da, fixture.m, PETSC_TRUE);CHKERRV(ierr);
  assert_that(ierr, is_equal_to(0));
  ierr = scMatTeardown(&fixture);CHKERRV(ierr);
}

struct JacobianFixture {
  struct MatFixture  matFixture;
  struct FftData     fftData;
  struct ProblemSpec problem;
  struct JacobianCtx jCtx;
  Mat                Jpre;
  Mat                J;
  TS                 ts;
  Vec                Xdot;
};

static PetscErrorCode jacobianSetup(struct JacobianFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = scMatSetup(&fixture->matFixture);CHKERRQ(ierr);
  ierr = scFftCreateFftData(fixture->matFixture.da, &fixture->fftData);CHKERRQ(ierr);
  fixture->problem.gamma = 0.0;
  fixture->jCtx.alpha = 0.0;
  fixture->jCtx.fftData = &fixture->fftData;
  fixture->jCtx.problem = &fixture->problem;

  fixture->Jpre = fixture->matFixture.m;

  PetscInt m, n, M, N;
  ierr = MatGetLocalSize(fixture->Jpre, &m, &n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(fixture->Jpre, &M, &N);CHKERRQ(ierr);
  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)fixture->Jpre,&comm);CHKERRQ(ierr);
  ierr = MatCreateShell(comm, m, n, M, N, &fixture->jCtx, &fixture->J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(fixture->J, MATOP_MULT, (void (*)(void))scJacobianMatMult);CHKERRQ(ierr);

  ierr = TSCreate(comm, &fixture->ts);CHKERRQ(ierr);
  ierr = TSSetDM(fixture->ts, fixture->matFixture.da);CHKERRQ(ierr);
  ierr = TSSetType(fixture->ts, TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetProblemType(fixture->ts, TS_NONLINEAR);CHKERRQ(ierr);

  ierr = VecDuplicate(fixture->matFixture.x, &fixture->Xdot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode jacobianTeardown(struct JacobianFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&fixture->Xdot);CHKERRQ(ierr);
  ierr = TSDestroy(&fixture->ts);CHKERRQ(ierr);
  ierr = MatDestroy(&fixture->J);CHKERRQ(ierr);
  ierr = scFftDestroyFftData(&fixture->fftData);CHKERRQ(ierr);
  ierr = scMatTeardown(&fixture->matFixture);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

Ensure(jacobian_and_preconditioner_are_consistent)
{
  PetscErrorCode ierr;
  struct JacobianFixture fixture;
  ierr = jacobianSetup(&fixture);CHKERRV(ierr);
  ierr = jacobianTeardown(&fixture);CHKERRV(ierr);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, can_build_jacobian);
  add_test(suite, can_build_jacobian_fourth_order);
  add_test(suite, jacobian_and_preconditioner_are_consistent);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
