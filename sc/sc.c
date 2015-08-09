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
static char help[] = "Nonlinear optical pulse propagation.\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petscmat.h>

#include <fft.h>

#define SQR(a) ((a) * (a))


struct Cmplx {
  PetscScalar re;
  PetscScalar im;
};

struct FftData {
  ScFft fft;
  Vec xu;
  Vec yu;
  Vec zu;
  Vec xv;
  Vec yv;
  Vec zv;
};

struct JacobianMatMul {
  PetscScalar    alpha;
  struct FftData *fftData;
};

struct AppCtx {
  PetscScalar           l;
  PetscBool             visualize;
  PetscBool             useFourthOrder;
  PetscViewer           viewer;
  PetscReal             gamma;
  struct FftData        fftData;
  struct JacobianMatMul jctx;
};

struct Field {
  PetscScalar u;
  PetscScalar v;
};


static PetscErrorCode FormInitialSolution(DM,Vec);
static PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode SCIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec G,void *ptr);
static PetscErrorCode SCIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx);
PETSC_STATIC_INLINE PetscScalar initialState(PetscReal x);
static PetscErrorCode addFirstDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode addFirstDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode addSecondDerivative(DM da, Mat m, PetscReal alph, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode addSecondDerivativeFourthOrder(DM da, Mat m, PetscReal alph, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode testFirstDerivative();
static PetscErrorCode testSecondDerivative();
static PetscErrorCode checkIFunctionAndJacobianConsistent(TS ts, void *ptr);
static PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J, void *ctx);
static PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J, void *ctx);
PetscErrorCode matrixFreeJacobian(Mat, Vec, Vec);
static PetscErrorCode matrixFreeJacobianImpl(struct JacobianMatMul *ctx, Vec x, Vec y);
static PetscInt clamp(PetscInt i, PetscInt imin, PetscInt imax);

int main(int argc,char **argv) {
  TS             ts;
  Vec            x,r;
  PetscInt       steps, maxsteps = 100, m, n, M, N;
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime;
  SNES           ts_snes;
  struct AppCtx  user;
  Mat            J,Jprec;
  PetscDraw      draw;
  PetscBool      useColoring, flg;

  PetscFunctionBegin;
  PetscInitialize(&argc,&argv,(char*)0,help);

  user.gamma = 0.20;
  ierr = PetscOptionsGetReal("", "-gamma", &user.gamma, &flg);CHKERRQ(ierr);
  user.visualize = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-visualize", &user.visualize, &flg);CHKERRQ(ierr);
  if (user.visualize) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"", 80, 380, 400, 160, &user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(user.viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  }
  useColoring = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-use_coloring", &useColoring, &flg);CHKERRQ(ierr);
  user.useFourthOrder = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-use_fourth_order", &user.useFourthOrder, &flg);CHKERRQ(ierr);

  if (user.useFourthOrder) {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,-1024,2,2,NULL,&da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,-1024,2,1,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  ierr = scFftCreate(da, &user.fftData.fft);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xu, &user.fftData.yu, &user.fftData.zu);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xv, &user.fftData.yv, &user.fftData.zv);CHKERRQ(ierr);
  user.jctx.fftData = &user.fftData;

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);

  ierr = TSSetIFunction(ts, NULL, SCIFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&Jprec);CHKERRQ(ierr);
  if (user.useFourthOrder) {
  ierr = buildConstantPartOfJacobianFourthOrder(da, Jprec, &user);CHKERRQ(ierr);
  } else {
    ierr = buildConstantPartOfJacobian(da, Jprec, &user);CHKERRQ(ierr);
  }
  ierr = MatSetOption(Jprec, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatStoreValues(Jprec);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Jprec, &m, &n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Jprec, &M, &N);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, &user.jctx, &J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J, MATOP_MULT, (void (*)(void))matrixFreeJacobian);CHKERRQ(ierr);


  if (!useColoring) {
    ierr = TSSetIJacobian(ts,J,Jprec,SCIJacobian,&user);CHKERRQ(ierr);
  } else {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,0);CHKERRQ(ierr);
  }

  ierr = TSSetDuration(ts,maxsteps,1.0);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MyTSMonitor,&user,0);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&ts_snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(ts_snes);CHKERRQ(ierr);

  ierr = FormInitialSolution(da,x);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.0001);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = scFftDestroy(&user.fftData.fft);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xv);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fftData.xv);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&Jprec);CHKERRQ(ierr);
  if (user.visualize) {
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

PetscErrorCode testFirstDerivative() {
  Mat m;
  DM  dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,5,2,1,NULL,&dm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &m);CHKERRQ(ierr);
  ierr = addFirstDerivative(dm, m, 1.0, 1.0, 0, 0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = MatDestroy(&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode testSecondDerivative() {
  Mat m;
  DM  dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,5,2,1,NULL,&dm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &m);CHKERRQ(ierr);
  ierr = addSecondDerivative(dm, m, 1.0, 1.0, 1, 0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = MatDestroy(&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SCIFunction(TS ts,PetscReal t,Vec X, Vec Xdot, Vec F, void *ptr)
{
  PetscErrorCode        ierr;
  DM                    da;
  DMDALocalInfo         info;
  struct AppCtx         *ctx = ptr;
  struct JacobianMatMul jctx;
  PetscInt              i;
  struct Field          *x, *f;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  /* linear term */
  jctx.fftData = &ctx->fftData;
  jctx.alpha = 0.0;
  ierr = matrixFreeJacobianImpl(&jctx, X, F);

  /* Nonlinear term */
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    f[i].u += -ctx->gamma * SQR(x[i].u) * x[i].u;
  }
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);

  /* Time derivative term */
  ierr = VecAXPY(F, 1.0, Xdot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode matrixFreeJacobian(Mat J, Vec x, Vec y)
{
  PetscErrorCode        ierr;
  struct JacobianMatMul *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J, &ctx);CHKERRQ(ierr);
  ierr = matrixFreeJacobianImpl(ctx, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode matrixFreeJacobianImpl(struct JacobianMatMul *ctx, Vec x, Vec y)
{
  PetscErrorCode        ierr;
  PetscInt              i,imin,imax,Mx;
  PetscReal             k;
  struct Cmplx          tmpu, tmpv;
  PetscScalar           *utilde, *vtilde;

  PetscFunctionBegin;
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  
  /*
  Equations:

  gu = u_t - c u_x - v
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
    k = 2.0 * M_PI * (PetscScalar)i / 1.0;
    utilde[2 * i]     = k * tmpu.im - tmpv.re;
    utilde[2 * i + 1] = -k * tmpu.re - tmpv.im;
    vtilde[2* i]      = k * tmpv.im + SQR(k) * tmpu.re;
    vtilde[2 * i + 1] = -k * tmpv.re + SQR(k) * tmpu.im;
  }
  ierr = VecRestoreArray(ctx->fftData->yv, &vtilde);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->fftData->yu, &utilde);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData->fft, y, 0, ctx->fftData->yu);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData->fft, y, 1, ctx->fftData->yv);CHKERRQ(ierr);
  ierr = VecGetSize(y, &Mx);CHKERRQ(ierr);
  ierr = VecScale(y, 1.0 / Mx);CHKERRQ(ierr);

  ierr = VecAXPY(y, ctx->alpha, x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode addFirstDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[2] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[2] = {-0.5 * alpha / hx, 0.5 * alpha / hx};

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 2; ++ii) {
    col[ii].c = ccomp;
  }
  if (info.xs == 0) {
    i = 0;
    row.i = i;
    col[0].i = Mx - 1;
    col[1].i = i + 1;
    ierr = MatSetValuesStencil(m, 1, &row, 2, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    col[0].i = i - 1;
    col[1].i = 0;
    ierr = MatSetValuesStencil(m, 1, &row, 2, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    --info.xm;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    col[0].i = i - 1;
    col[1].i = i + 1;
    ierr = MatSetValuesStencil(m, 1, &row, 2, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode addFirstDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[4] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[4] = {
    1.0 * alpha / (12.0 * hx),
    -8.0 * alpha / (12.0 * hx),
    8.0 * alpha / (12.0 * hx),
    -1.0 * alpha / (12.0 * hx)
  };

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 4; ++ii) {
    col[ii].c = ccomp;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    col[0].i = clamp(i - 2, 0, Mx);
    col[1].i = clamp(i - 1, 0, Mx);
    col[2].i = clamp(i + 1, 0, Mx);
    col[3].i = clamp(i + 2, 0, Mx);
    ierr = MatSetValuesStencil(m, 1, &row, 4, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode addSecondDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[3] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[3] = {alpha / SQR(hx), -2.0 * alpha / SQR(hx), alpha / SQR(hx)};

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 3; ++ii) {
    col[ii].c = ccomp;
  }
  if (info.xs == 0) {
    i = 0;
    row.i = i;
    col[0].i = Mx - 1;
    for (ii = i + 1; ii < 3; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 3, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    for (ii = 0; ii < 2; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    col[2].i = 0;
    ierr = MatSetValuesStencil(m, 1, &row, 3, col, v, ADD_VALUES);CHKERRQ(ierr);
    --info.xm;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    for (ii = 0; ii < 3; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 3, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscInt clamp(PetscInt i, PetscInt imin, PetscInt imax)
{
  PetscInt L = imax - imin;
  if (i < imin) {
    i += L;
  }
  if (i >= imax) {
    i -= L;
  }
  return i;
}

PetscErrorCode addSecondDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp)
{
  DMDALocalInfo  info;
  PetscInt       i, ii, Mx;
  MatStencil     col[3] = {{0}},row = {0};
  PetscErrorCode ierr;
  PetscScalar    v[5] = {
    -alpha / (12.0 * SQR(hx)),
    16.0 * alpha /(12.0 * SQR(hx)),
    -30.0 * alpha /(12.0 * SQR(hx)),
    16.0 * alpha / (12.0 * SQR(hx)),
    -alpha / (12.0 * SQR(hx))
  };

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  row.c = rcomp;
  for (ii = 0; ii < 5; ++ii) {
    col[ii].c = ccomp;
  }
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    row.i = i;
    for (ii = 0; ii < 5; ++ii) {
      col[ii].i = clamp(i - 2 + ii, 0, Mx);
    }
    ierr = MatSetValuesStencil(m, 1, &row, 5, col, v, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode buildConstantPartOfJacobian(DM da, Mat J, void *ctx)
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
  ierr = addSecondDerivative(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = addFirstDerivative(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = addFirstDerivative(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

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

PetscErrorCode buildConstantPartOfJacobianFourthOrder(DM da, Mat J, void *ctx)
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
  ierr = addSecondDerivativeFourthOrder(da, J, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = addFirstDerivativeFourthOrder(da, J, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = addFirstDerivativeFourthOrder(da, J, -1.0, hx, 1, 1);CHKERRQ(ierr);

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

PetscErrorCode SCIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscInt       i,c,Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;
  struct Field   *x;
  struct AppCtx  *user = ctx;

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  ierr = MatRetrieveValues(Jpre);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  /* Non-linear term */
  row.c = 0;
  col.c = 0;
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    v = -3.0 * user->gamma * SQR(x[i].u);
    ierr=MatSetValuesStencil(Jpre,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);

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

  ((struct AppCtx*)ctx)->jctx.alpha = a;

  PetscFunctionReturn(0);
}


PetscErrorCode FormInitialSolution(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  PetscScalar    **u;
  PetscReal      hx,x;

  PetscFunctionBeginUser;
  DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  hx = 1.0/(PetscReal)(Mx-1);

  ierr = DMDAVecGetArrayDOF(da,U,&u);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    x = (i - 0.5 * Mx) * hx;
    u[i][0] = initialState(x);
    u[i][1] = -(initialState(x + 0.5 * hx) - initialState(x - 0.5 * hx)) / hx;
  }

  ierr = DMDAVecRestoreArrayDOF(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscScalar initialState(PetscReal x) {
  return exp(-SQR(SQR(SQR(SQR(SQR(x / 0.1))))));
}

PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      norm, dt;
  MPI_Comm       comm;
  struct AppCtx* appCtx = ctx;

  PetscFunctionBeginUser;
  ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"timestep %D time %g (dt = %g) norm %g\n",step,(double)ptime,(double)dt,(double)norm);CHKERRQ(ierr);
  if (appCtx->visualize) {
    ierr = VecView(v, appCtx->viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode checkIFunctionAndJacobianConsistent(TS ts, void *ptr)
{
  DM             da;
  PetscErrorCode ierr;
  Vec            X, Xdot, GIfunc, GJfunc;
  PetscRandom    rdm;
  Mat            J;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);

  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(da, &X);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(X, rdm);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecSet(X, 1.0);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(da, &Xdot);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xdot);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &GIfunc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &GJfunc);CHKERRQ(ierr);

  ierr = SCIFunction(ts, 0, X, Xdot, GIfunc, ptr);
  ierr = VecView(GIfunc, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = SCIJacobian(ts, 0, X, Xdot, 0, J, J, ptr);CHKERRQ(ierr);
  ierr = MatMult(J, X, GJfunc);CHKERRQ(ierr);
  ierr = VecView(GJfunc, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(da, &GJfunc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &GIfunc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &Xdot);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
