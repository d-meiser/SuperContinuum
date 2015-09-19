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
#ifndef JACOBIAN_TEST_UTILS_H
#define JACOBIAN_TEST_UTILS_H

#include <ScTestUtilsExport.h>
#include <test_utils.h>
#include <jacobian.h>
#include <fft.h>

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
SC_TEST_UTILS_API PetscErrorCode jacobianSetup(struct JacobianFixture* fixture, PetscScalar alpha, PetscScalar gamma);
SC_TEST_UTILS_API PetscErrorCode jacobianTeardown(struct JacobianFixture* fixture);
typedef PetscScalar (*WaveForm)(PetscScalar);
typedef PetscErrorCode (*FillerMethod)(Vec, WaveForm);

SC_TEST_UTILS_API PetscScalar constant_func(PetscScalar x);
SC_TEST_UTILS_API PetscScalar sine_wave(PetscScalar x);
SC_TEST_UTILS_API PetscScalar gaussian(PetscScalar x);
SC_TEST_UTILS_API PetscErrorCode rightMovingWave(Vec X, WaveForm f);
SC_TEST_UTILS_API PetscErrorCode waveAtRest(Vec X, WaveForm f);
SC_TEST_UTILS_API PetscScalar constant_func(PetscScalar x);
SC_TEST_UTILS_API PetscErrorCode checkJacobianPreConsistency(struct JacobianFixture *fixture, FillerMethod filler, WaveForm f, PetscScalar alpha, PetscScalar gamma, PetscBool fourthOrder, PetscScalar tol, PetscBool view);

#endif
