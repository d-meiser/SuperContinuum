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
#ifndef FD_H
#define FD_H

#include <petscdm.h>
#include <petscmat.h>
#include <ScExport.h>

SC_API PetscErrorCode scFdAddFirstDerivative(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
SC_API PetscErrorCode scFdAddFirstDerivativeFourthOrder(DM da, Mat m, PetscReal alpha, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
SC_API PetscErrorCode scFdAddSecondDerivative(DM da, Mat m, PetscReal alph, PetscReal hx, PetscInt rcomp, PetscInt ccomp);
SC_API PetscErrorCode scFdAddSecondDerivativeFourthOrder(DM da, Mat m, PetscReal alph, PetscReal hx, PetscInt rcomp, PetscInt ccomp);

#endif
