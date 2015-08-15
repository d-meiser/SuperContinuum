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

struct JacobianFixture {
  struct MatFixture  matFixture;
  struct FftData     fftData;
  struct ProblemSpec problem;
  struct JacobianCtx jCtx;
  Mat                Jpre;
  Mat                J;
  TS                 ts;
  Vec                Xdot;
  Vec                yJ;
};

static PetscErrorCode jacobianSetup(struct JacobianFixture* fixture);
static PetscErrorCode jacobianTeardown(struct JacobianFixture* fixture);
static PetscErrorCode rightMovingWave(Vec X, PetscScalar (*f)(PetscScalar));
static PetscScalar constant_func(PetscScalar x);
static PetscErrorCode checkJacobianPreConsistency(PetscScalar (*state)(PetscScalar), PetscScalar tol, PetscBool view);

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


static PetscScalar constant_func(PetscScalar x) {
  return 2.3;
}

Ensure(jacobian_and_preconditioner_are_consistent)
{
  PetscErrorCode ierr;
  ierr = checkJacobianPreConsistency(constant_func, 1.0e-6, PETSC_FALSE);CHKERRV(ierr);
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

static PetscErrorCode checkJacobianPreConsistency(PetscScalar (*state)(PetscScalar), PetscScalar tol, PetscBool view) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  struct JacobianFixture fixture;
  ierr = jacobianSetup(&fixture);CHKERRQ(ierr);
  Vec X = fixture.matFixture.x;
  ierr = rightMovingWave(X, state);CHKERRQ(ierr);
  if (view) {
    ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = scJacobianBuildConstantPart(fixture.matFixture.da, fixture.Jpre, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatStoreValues(fixture.Jpre);CHKERRQ(ierr);
  ierr = scJacobianBuildPre(fixture.ts, 0.0, X, fixture.Xdot, 0.0, fixture.Jpre, &fixture.jCtx);CHKERRQ(ierr);
  ierr = scJacobianBuild(fixture.ts, 0.0, X, fixture.Xdot, 0.0, fixture.J, &fixture.jCtx);CHKERRQ(ierr);
  ierr = MatMult(fixture.Jpre, X, fixture.matFixture.y);CHKERRQ(ierr);
  if (view) {
    ierr = VecView(fixture.matFixture.y, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatMult(fixture.J, X, fixture.yJ);CHKERRQ(ierr);
  if (view) {
    ierr = VecView(fixture.yJ, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecAXPY(fixture.yJ, -1.0, fixture.matFixture.y);CHKERRQ(ierr);
  PetscScalar err;
  ierr = VecNorm(fixture.yJ, NORM_2, &err);CHKERRQ(ierr);
  assert_that(err < tol, is_true);
  ierr = jacobianTeardown(&fixture);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = VecDuplicate(fixture->matFixture.x, &fixture->yJ);CHKERRQ(ierr);
  ierr = VecZeroEntries(fixture->Xdot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode jacobianTeardown(struct JacobianFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&fixture->yJ);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->Xdot);CHKERRQ(ierr);
  ierr = TSDestroy(&fixture->ts);CHKERRQ(ierr);
  ierr = MatDestroy(&fixture->J);CHKERRQ(ierr);
  ierr = scFftDestroyFftData(&fixture->fftData);CHKERRQ(ierr);
  ierr = scMatTeardown(&fixture->matFixture);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode rightMovingWave(Vec X, PetscScalar (*f)(PetscScalar))
{
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, Mx;
  PetscScalar    **x;
  PetscReal      hx, z;
  DM             da;

  PetscFunctionBegin;
  ierr = VecGetDM(X, &da);CHKERRQ(ierr);
  DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  hx = 1.0/(PetscReal)(Mx-1);

  ierr = DMDAVecGetArrayDOF(da,X,&x);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    z = (i - 0.5 * Mx) * hx;
    x[i][0] = f(z);
    x[i][1] = -(f(z + 0.5 * hx) - f(z - 0.5 * hx)) / hx;
  }

  ierr = DMDAVecRestoreArrayDOF(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
