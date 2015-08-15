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
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>
#include <ScExport.h>

/* Fixture for testing matrices */
struct MatFixture {
  PetscInt dim;
  PetscInt numComponents;
  DM da;
  DMDALocalInfo info;
  Mat m;
  Vec x;
  Vec y;
  PetscScalar *xarr;
  PetscScalar *yarr;
};

SC_API PetscErrorCode scMatSetup(struct MatFixture* fixture);
SC_API PetscErrorCode scMatTeardown(struct MatFixture* fixture);

#endif
