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

struct AppCtx {
  PetscScalar    l;
  PetscBool      visualize;
  PetscViewer    viewer;
  PetscReal      gamma;
  struct FftData fftData;
};

struct Field {
  PetscScalar u;
  PetscScalar v;
};


static PetscErrorCode FormInitialSolution(DM,Vec);
static PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode SCRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr);
static PetscErrorCode SCRHSJacobian(TS ts,PetscReal t,Vec X,Mat J,Mat Jpre,void *ptr);
static PetscErrorCode SCIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec G,void *ptr);
static PetscErrorCode SCIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx);
PETSC_STATIC_INLINE PetscScalar initialState(PetscReal x);
static PetscErrorCode addFirstDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode addSecondDerivative(DM da, Mat m, PetscReal alph, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
static PetscErrorCode testFirstDerivative();
static PetscErrorCode testSecondDerivative();

int main(int argc,char **argv) {
  TS             ts;
  Vec            x,r;
  PetscInt       steps,maxsteps = 100,nlocal,nglobal;
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime;
  SNES           ts_snes;
  struct AppCtx  user;
  Mat            J,Jrhs;
  PetscDraw      draw;
  PetscBool      useColoring, flg;

  PetscFunctionBegin;
  PetscInitialize(&argc,&argv,(char*)0,help);

  user.gamma = 0.010;
  user.visualize = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-visualize", &user.visualize, &flg);CHKERRQ(ierr);
  if (user.visualize) {
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"", 80, 380, 400, 160, &user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(user.viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  }
  useColoring = PETSC_FALSE;
  ierr = PetscOptionsGetBool("", "-use_coloring", &useColoring, &flg);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-1024,2,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  ierr = scFftCreate(da, &user.fftData.fft);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xu, &user.fftData.yu, &user.fftData.zu);CHKERRQ(ierr);
  ierr = scFftCreateVecsFFTW(user.fftData.fft, &user.fftData.xv, &user.fftData.yv, &user.fftData.zv);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, SCRHSFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&Jrhs);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,Jrhs,Jrhs,SCRHSJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts, NULL, SCIFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  if (!useColoring) {
    ierr = TSSetIJacobian(ts,J,J,SCIJacobian,&user);CHKERRQ(ierr);
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
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,5,2,1,NULL,&dm);CHKERRQ(ierr);
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
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,5,2,1,NULL,&dm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &m);CHKERRQ(ierr);
  ierr = addSecondDerivative(dm, m, 1.0, 1.0, 1, 0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = MatDestroy(&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SCRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  DM             da;
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  struct Field   *x, *f;
  struct AppCtx  *ctx = ptr;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs + info.xm; ++i) {
    f[i].u += ctx->gamma * SQR(x[i].u) * x[i].u;
  }
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SCRHSJacobian(TS ts,PetscReal t,Vec X,Mat J,Mat Jpre,void *ptr)
{
  PetscInt       i;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;
  struct Field   *x;
  struct AppCtx  *ctx = ptr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  row.c = 0;
  col.c = 0;
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    v = 3.0 * ctx->gamma * SQR(x[i].u);
    ierr=MatSetValuesStencil(Jpre,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr=MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SCIFunction(TS ts,PetscReal t,Vec X, Vec Xdot, Vec G,void *ptr)
{
  DM             da;
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  PetscInt       i,Mx;
  PetscReal      hx,sx,k;
  struct Cmplx   tmpu, tmpv;
  PetscScalar    *utilde, *vtilde;
  PetscScalar    u,uxx,v;
  struct Field   *x, *xdot, *g;
  Vec            Xloc;
  struct AppCtx  *ctx = ptr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  hx = 1.0/(PetscReal)(Mx-1);
  sx = 1.0/(hx*hx);

  ierr = VecZeroEntries(G);CHKERRQ(ierr);
  
  /*
  Equations:

  gu = u_t - c u_x - v
  gv = v_t - c v_x - u_xx

  The time derivative terms are added in real space, all other terms are
  dealt with in the frequency domain.
  */

  ierr = scFftTransform(ctx->fftData.fft, X, 0, ctx->fftData.yu);CHKERRQ(ierr);
  ierr = scFftTransform(ctx->fftData.fft, X, 1, ctx->fftData.yv);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->fftData.yu, &utilde);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->fftData.yv, &vtilde);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs + info.xm; ++i) {
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
  ierr = VecRestoreArray(ctx->fftData.yv, &vtilde);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->fftData.yu, &utilde);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData.fft, G, 0, ctx->fftData.yu);CHKERRQ(ierr);
  ierr = scFftITransform(ctx->fftData.fft, G, 1, ctx->fftData.yv);CHKERRQ(ierr);
  ierr = VecScale(G, 1.0 / Mx);CHKERRQ(ierr);

  ierr = VecAXPY(G, 1.0, Xdot);CHKERRQ(ierr);

  /*
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,G,&g);CHKERRQ(ierr);

  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0) {
      g[i].u += xdot[i].u - (x[i + 1].u) / (2.0 * hx) - x[i].v;
      g[i].v += xdot[i].v - (x[i + 1].v) / (2.0 * hx) - sx * (-2.0 * x[i].u + x[i + 1].u);
      continue;
    }
    if (i == Mx - 1) {
      g[i].u += xdot[i].u - (- x[i - 1].u) / (2.0 * hx) - x[i].v;
      g[i].v += xdot[i].v - (- x[i - 1].v) / (2.0 * hx) - sx * (-2.0 * x[i].u + x[i - 1].u);
      continue;
    }
    u      = x[i].u;
    v      = x[i].v;
    uxx    = (-2.0*u + x[i-1].u + x[i+1].u)*sx;
    g[i].u += xdot[i].u - (x[i + 1].u - x[i - 1].u) / (2.0 * hx) - v;
    g[i].v += xdot[i].v - (x[i + 1].v - x[i - 1].v) / (2.0 * hx) - uxx;
  }
  ierr = PetscLogFlops(4.0*info.xm);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,G,&g);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  */
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
    col[1].i = i + 1;
    ierr = MatSetValuesStencil(m, 1, &row, 1, &col[1], &v[1], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    col[0].i = i - 1;
    ierr = MatSetValuesStencil(m, 1, &row, 1, &col[0], &v[0], ADD_VALUES);CHKERRQ(ierr);
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
    for (ii = i + 1; ii < 3; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 2, &col[1], &v[1], ADD_VALUES);CHKERRQ(ierr);
    ++info.xs;
    --info.xm;
  }
  if (info.xs + info.xm == Mx) {
    i = Mx - 1;
    row.i = i;
    for (ii = 0; ii < 2; ++ii) {
      col[ii].i = i - 1 + ii;
    }
    ierr = MatSetValuesStencil(m, 1, &row, 2, col, v, ADD_VALUES);CHKERRQ(ierr);
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

PetscErrorCode SCIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscInt       i,c,Mx;
  MatStencil     col = {0}, row = {0};
  PetscScalar    v,hx;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = 1.0/(PetscReal)(Mx-1);
  ierr = addSecondDerivative(da, Jpre, -1.0, hx, 1, 0);CHKERRQ(ierr);
  ierr = addFirstDerivative(da, Jpre, -1.0, hx, 0, 0);CHKERRQ(ierr);
  ierr = addFirstDerivative(da, Jpre, -1.0, hx, 1, 1);CHKERRQ(ierr);

  v = -1.0;
  row.c = 0;
  col.c = 1;
  for (i=info.xs; i<info.xs+info.xm; i++) {
    row.i = i;
    col.i = i;
    ierr=MatSetValuesStencil(Jpre,1,&row,1,&col,&v,ADD_VALUES);CHKERRQ(ierr);
  }

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
  ierr=MatSetOption(Jpre,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
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

