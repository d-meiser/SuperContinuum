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
#ifndef JACOBIAN_H
#define JACOBIAN_H

#include <petscdm.h>
#include <petscts.h>
#include <petscmat.h>
#include <ScExport.h>
#include <problem.h>
#include <fft.h>

struct FftData {
  ScFft fft;
  Vec xu;
  Vec yu;
  Vec zu;
  Vec xv;
  Vec yv;
  Vec zv;
};

struct JacobianCtx {
  PetscScalar        alpha;
  struct FftData     *fftData;
  struct ProblemSpec *problem;
};


SC_API PetscErrorCode scJacobianBuildConstantPart(DM da, Mat J, PetscBool fourthOrder);
SC_API PetscErrorCode scJacobianBuild(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, struct JacobianCtx *ctx);
SC_API PetscErrorCode scJacobianBuildPre(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, struct JacobianCtx *ctx);
SC_API PetscErrorCode scJacobianMatMult(Mat J, Vec x, Vec y);
SC_API PetscErrorCode scJacobianApply(struct JacobianCtx* ctx, Vec x, Vec y);

#endif
