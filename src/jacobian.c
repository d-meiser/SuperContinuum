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
#include <jacobian.h>
#include <petscdmda.h>
#include <fd.h>
#include <basic_types.h>

#ifndef SQR
#define SQR(a) ((a) * (a))
#endif

/*add some stuff that breaks the code;*/

static PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J);
static PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J);

PetscErrorCode scJacobianCreate(struct FftData *fftData, struct ProblemSpec *problem, struct JacobianCtx *ctx)
{
  PetscFunctionBegin;
  ctx->alpha = 0;
  ctx->fftData = fftData;
  ctx->problem = problem;
  ctx->X0 = 0;
  ctx->Xdot0 = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianDestroy(struct JacobianCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->X0) {
    ierr = VecDestroy(&ctx->X0);CHKERRQ(ierr);
  }
  if (ctx->Xdot0) {
    ierr = VecDestroy(&ctx->Xdot0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianBuildLinearPart(DM da, Mat J, PetscBool fourthOrder)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fourthOrder) {
    ierr = buildConstantPartOfJacobianFourthOrder(da, J);CHKERRQ(ierr);
  } else {
    ierr = buildConstantPartOfJacobian(da, J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J)
{
  PetscErrorCode ierr;
  PetscInt       i, Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v, hx;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);
  ierr = scFdAddSecondDerivative(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivative(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivative(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

  /* Auxiliary variable relation */
  v = -1.0;
  row.c = 0;
  col.c = 1;
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    ierr=MatSetValuesStencil(J,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr=MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J)
{
  PetscErrorCode ierr;
  PetscInt       i, Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v, hx;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);
  ierr = scFdAddSecondDerivativeFourthOrder(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivativeFourthOrder(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = scFdAddFirstDerivativeFourthOrder(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

  /* Auxiliary variable relation */
  v = -1.0;
  row.c = 0;
  col.c = 1;
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    ierr=MatSetValuesStencil(J,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr=MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianBuild(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, struct JacobianCtx *ctx)
{
  PetscReal ierr;

  PetscFunctionBegin;
  ctx->alpha = a;
  if (!ctx->X0) {
    ierr = VecDuplicate(X, &ctx->X0);CHKERRQ(ierr);
  }
  ierr = VecCopy(X, ctx->X0);CHKERRQ(ierr);
  if (!ctx->Xdot0) {
    ierr = VecDuplicate(Xdot, &ctx->Xdot0);CHKERRQ(ierr);
  }
  ierr = VecCopy(Xdot, ctx->Xdot0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianBuildPre(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat Jpre, struct JacobianCtx *ctx)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;
  PetscInt       i, c;
  PetscScalar    v;
  MatStencil     col = {0};

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  ierr = MatRetrieveValues(Jpre);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Time derivative terms */
  v = a;
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    col.i = i;
    for (c = 0; c < 2; ++c) {
      col.c = c;
      ierr=MatSetValuesStencil(Jpre,1,&col,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr=MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianMatMult(Mat J, Vec x, Vec y)
{
  PetscErrorCode     ierr;
  struct JacobianCtx *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J, &ctx);CHKERRQ(ierr);
  ierr = scJacobianApply(ctx, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scJacobianApply(struct JacobianCtx *ctx, Vec x, Vec y)
{
  PetscErrorCode        ierr;
  PetscInt              i,imin,imax,Mx;
  PetscReal             k;
  struct Cmplx          tmpu, tmpv;
  PetscScalar           *utilde, *vtilde;

  PetscFunctionBegin;
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecGetSize(y, &Mx);CHKERRQ(ierr);
  Mx /= 2;
  /* sampling frequency */
  PetscScalar hx = 1.0 / (Mx - 1);
  PetscScalar kNyquist = M_PI / hx;
  
  /*
  Equations:

  gu = u_t - c u_x - v - gamma * u^3
  gv = v_t - c v_x - u_xx

  The time derivative terms are added in real space, all other terms are
  dealt with in the frequency domain.
  */

  ierr = scFftTransform(ctx->fftData->fft, x, 0, ctx->fftData->yu);CHKERRQ(ierr);
  ierr = scFftTransform(ctx->fftData->fft, x, 1, ctx->fftData->yv);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->fftData->yu, &utilde);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->fftData->yv, &vtilde);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(ctx->fftData->yu, &imin, &imax);CHKERRQ(ierr);
  imin /= 2;
  imax /= 2;
  for (i = imin; i < imax; ++i) {
    tmpu.re = utilde[2 * i];
    tmpu.im = utilde[2 * i + 1];
    tmpv.re = vtilde[2 * i];
    tmpv.im = vtilde[2 * i + 1];
    k = 2.0 * M_PI / hx * (PetscScalar)i / (PetscScalar)Mx;
    if (k > kNyquist) {
      k -= 2.0 * kNyquist;
    }
    utilde[2 * i]     = k * tmpu.im - tmpv.re;
    utilde[2 * i + 1] = -k * tmpu.re - tmpv.im;
    vtilde[2* i]      = k * tmpv.im + SQR(k) * tmpu.re;
    vtilde[2 * i + 1] = -k * tmpv.re + SQR(k) * tmpu.im;
  }
  ierr = VecRestoreArray(ctx->fftData->yv, &vtilde);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->fftData->yu, &utilde);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData->fft, y, 0, ctx->fftData->yu);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData->fft, y, 1, ctx->fftData->yv);CHKERRQ(ierr);
  ierr = VecScale(y, 1.0 / (PetscScalar)Mx);CHKERRQ(ierr);

  /* time dependent term */
  ierr = VecAXPY(y, ctx->alpha, x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
