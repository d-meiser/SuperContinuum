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
#include <fd.h>
#include <jacobian.h>

#ifndef SQR
#define SQR(a) ((a) * (a))
#endif


struct FftData {
  ScFft fft;
  Vec xu;
  Vec yu;
  Vec zu;
  Vec xv;
  Vec yv;
  Vec zv;
};

struct AppCtx {
  PetscScalar           l;
  PetscBool             monitorRealSpace;
  PetscBool             monitorSpectrum;
  PetscBool             useFourthOrder;
  PetscViewer           viewerRealSpace;
  PetscViewer           viewerSpectrum;
  struct ProblemSpec    problem;
  struct FftData        fftData;
  struct JacobianCtx    jctx;
  Vec                   psd;
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
static PetscErrorCode checkIFunctionAndJacobianConsistent(TS ts, void *ptr);
PetscErrorCode matrixFreeJacobian(Mat, Vec, Vec);
static PetscErrorCode matrixFreeJacobianImpl(struct JacobianCtx *ctx, Vec x, Vec y);
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

  user.problem.gamma = 0.20;
  ierr = PetscOptionsGetReal("", "-gamma", &user.problem.gamma, &flg);CHKERRQ(ierr);

  user.monitorRealSpace = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-monitor_real_space", &user.monitorRealSpace, &flg);CHKERRQ(ierr);
  if (user.monitorRealSpace) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"", 80, 380, 400, 160, &user.viewerRealSpace);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(user.viewerRealSpace,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  }

  user.monitorSpectrum = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-monitor_spectrum", &user.monitorSpectrum, &flg);CHKERRQ(ierr);
  if (user.monitorSpectrum) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"", 80, 380, 400, 160, &user.viewerSpectrum);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(user.viewerSpectrum,0,&draw);CHKERRQ(ierr);
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
  if (user.monitorSpectrum) {
    ierr = VecCreate(PETSC_COMM_WORLD, &user.psd);CHKERRQ(ierr);
    ierr = VecSetFromOptions(user.psd);CHKERRQ(ierr);
    PetscInt dim;
    ierr = VecGetSize(x, &dim);CHKERRQ(ierr);
    ierr = VecSetSizes(user.psd, PETSC_DECIDE, dim / 4);CHKERRQ(ierr);
  }

  ierr = scFftCreate(da, &user.fftData.fft);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xu, &user.fftData.yu, &user.fftData.zu);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xv, &user.fftData.yv, &user.fftData.zv);CHKERRQ(ierr);
  user.jctx.fftData = &user.fftData;
  user.jctx.problem = &user.problem;

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);

  ierr = TSSetIFunction(ts, NULL, SCIFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&Jprec);CHKERRQ(ierr);
  ierr = scJacobianBuildConstantPart(da, Jprec, user.useFourthOrder);CHKERRQ(ierr);
  ierr = MatSetOption(Jprec, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatStoreValues(Jprec);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Jprec, &m, &n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Jprec, &M, &N);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, &user.jctx, &J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J, MATOP_MULT, (void (*)(void))matrixFreeJacobian);CHKERRQ(ierr);


  if (!useColoring) {
    ierr = TSSetIJacobian(ts,J,Jprec,SCIJacobian,&user.jctx);CHKERRQ(ierr);
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
  if (user.monitorRealSpace) {
    ierr = PetscViewerDestroy(&user.viewerRealSpace);CHKERRQ(ierr);
  }
  if (user.monitorSpectrum) {
    ierr = PetscViewerDestroy(&user.viewerSpectrum);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

PetscErrorCode SCIFunction(TS ts,PetscReal t,Vec X, Vec Xdot, Vec F, void *ptr)
{
  PetscErrorCode        ierr;
  DM                    da;
  DMDALocalInfo         info;
  struct AppCtx         *ctx = ptr;
  struct JacobianCtx jctx;
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
    f[i].u += -ctx->problem.gamma * SQR(x[i].u) * x[i].u;
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
  struct JacobianCtx *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J, &ctx);CHKERRQ(ierr);
  ierr = matrixFreeJacobianImpl(ctx, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode matrixFreeJacobianImpl(struct JacobianCtx *ctx, Vec x, Vec y)
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


PetscErrorCode SCIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode     ierr;
  struct JacobianCtx *jac = ctx;

  PetscFunctionBegin;
  ierr = scJacobianBuild(ts, t, X, Xdot, a, J, jac);CHKERRQ(ierr);
  ierr = scJacobianBuildPre(ts, t, X, Xdot, a, Jpre, jac);CHKERRQ(ierr);
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
  ierr = VecStrideNorm(v,0,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"timestep %D time %g (dt = %g) norm %g\n",step,(double)ptime,(double)dt,(double)norm);CHKERRQ(ierr);
  if (appCtx->monitorRealSpace) {
    ierr = VecView(v, appCtx->viewerRealSpace);CHKERRQ(ierr);
  }
  if (appCtx->monitorSpectrum) {
    scFftComputePSD(appCtx->fftData.fft, v, 0, appCtx->fftData.yu, appCtx->psd, PETSC_TRUE);
    VecView(appCtx->psd, appCtx->viewerSpectrum);
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

