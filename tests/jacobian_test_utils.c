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
#include <jacobian_test_utils.h>
#include <cgreen/cgreen.h>

PetscScalar constant_func(PetscScalar x) { return 2.3; }
PetscScalar sine_wave(PetscScalar x) {
  PetscScalar k = 2.0 * M_PI * 1.0;
  return sin(k * x);
}

PetscScalar gaussian(PetscScalar x) {
  PetscScalar sigma = 0.1;
  return exp(-(x * x) / (sigma * sigma));
}

PetscErrorCode checkJacobianPreConsistency(struct JacobianFixture *fixture, FillerMethod filler, WaveForm f, PetscScalar alpha, PetscScalar gamma, PetscBool fourthOrder, PetscScalar tol, PetscBool view) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fixture->jCtx.alpha = alpha;
  fixture->problem.gamma = gamma;
  Vec X = fixture->matFixture.x;
  ierr = filler(X, f);CHKERRQ(ierr);
  ierr = scJacobianBuildLinearPart(fixture->matFixture.da, fixture->Jpre, fourthOrder);CHKERRQ(ierr);
  ierr = MatStoreValues(fixture->Jpre);CHKERRQ(ierr);
  ierr = scJacobianBuildPre(fixture->ts, 0.0, X, fixture->Xdot, alpha, fixture->Jpre, &fixture->jCtx);CHKERRQ(ierr);
  ierr = scJacobianBuild(fixture->ts, 0.0, X, fixture->Xdot, alpha, fixture->J, &fixture->jCtx);CHKERRQ(ierr);

  ierr = MatMult(fixture->Jpre, X, fixture->matFixture.y);CHKERRQ(ierr);
  ierr = MatMult(fixture->J, X, fixture->yJ);CHKERRQ(ierr);

  if (view) {
    DM da;
    ierr = VecGetDM(X, &da);CHKERRQ(ierr);
    PetscInt Mx;
    DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const PetscScalar *xarr, *yprearr, *yjarr;
    ierr = VecGetArrayRead(X, &xarr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(fixture->matFixture.y, &yprearr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(fixture->yJ, &yjarr);CHKERRQ(ierr);
    PetscInt i = 0;
    PetscScalar k = 2.0 * M_PI / 1.0;
    PetscScalar hx = 1.0 / Mx;
    for (i = 0; i < 30; ++i) {
      if (i % 2 == 0) {
        printf("%lf %lf %lf %lf\n", xarr[i], yprearr[i], yjarr[i], -k * cos(k * (i / 2 - 0.5 * Mx) * hx));
      }
    }
    ierr = VecRestoreArrayRead(fixture->yJ, &yjarr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(fixture->matFixture.y, &yprearr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X, &xarr);CHKERRQ(ierr);
  }

  ierr = VecAXPY(fixture->yJ, -1.0, fixture->matFixture.y);CHKERRQ(ierr);
  PetscScalar err;
  PetscScalar nrm;
  ierr = VecStrideNorm(fixture->yJ, 0, NORM_2, &err);CHKERRQ(ierr);
  ierr = VecStrideNorm(fixture->matFixture.y, 0, NORM_2, &nrm);CHKERRQ(ierr);
  nrm += 1.0e-6;
  assert_that((err / nrm < tol), is_true);
  ierr = VecStrideNorm(fixture->yJ, 1, NORM_2, &err);CHKERRQ(ierr);
  ierr = VecStrideNorm(fixture->matFixture.y, 1, NORM_2, &nrm);CHKERRQ(ierr);
  nrm += 1.0e-6;
  assert_that(err / nrm < tol, is_true);
  PetscFunctionReturn(0);
}

PetscErrorCode jacobianSetup(struct JacobianFixture* fixture, PetscScalar alpha, PetscScalar gamma) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = scMatSetup(&fixture->matFixture);CHKERRQ(ierr);
  ierr = scFftCreateFftData(fixture->matFixture.da, &fixture->fftData);CHKERRQ(ierr);
  fixture->problem.gamma = gamma;
  ierr = scJacobianCreate(&fixture->fftData, &fixture->problem, &fixture->jCtx);CHKERRQ(ierr);
  fixture->jCtx.alpha = alpha;

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

PetscErrorCode jacobianTeardown(struct JacobianFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&fixture->yJ);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->Xdot);CHKERRQ(ierr);
  ierr = TSDestroy(&fixture->ts);CHKERRQ(ierr);
  ierr = MatDestroy(&fixture->J);CHKERRQ(ierr);
  ierr = scFftDestroyFftData(&fixture->fftData);CHKERRQ(ierr);
  ierr = scMatTeardown(&fixture->matFixture);CHKERRQ(ierr);
  ierr = scJacobianDestroy(&fixture->jCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode rightMovingWave(Vec X, PetscScalar (*f)(PetscScalar))
{
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, Mx;
  PetscScalar    **x;
  PetscReal      hx, z;
  DM             da;

  PetscFunctionBegin;
  ierr = VecGetDM(X, &da);CHKERRQ(ierr);
  DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  hx = 1.0/(PetscReal)Mx;

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

PetscErrorCode waveAtRest(Vec X, PetscScalar (*f)(PetscScalar))
{
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, Mx;
  PetscScalar    **x;
  PetscReal      hx, z;
  DM             da;

  PetscFunctionBegin;
  ierr = VecGetDM(X, &da);CHKERRQ(ierr);
  DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  hx = 1.0 / (PetscReal)Mx;

  ierr = DMDAVecGetArrayDOF(da,X,&x);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    z = (i - 0.5 * Mx) * hx;
    x[i][0] = f(z);
    x[i][1] = 0;
  }

  ierr = DMDAVecRestoreArrayDOF(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
