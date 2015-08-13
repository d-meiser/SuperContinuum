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
#include <fd.h>

static const char help[] = "Unit tests for FD methods.";

Ensure(first_derivative)
{
  Mat m;
  DM  dm;
  PetscErrorCode ierr;

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,5,2,1,NULL,&dm);CHKERRV(ierr);
  ierr = DMCreateMatrix(dm, &m);CHKERRV(ierr);
  ierr = scFdAddFirstDerivative(dm, m, 1.0, 1.0, 0, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(ierr);
  ierr = DMDestroy(&dm);CHKERRV(ierr);
  ierr = MatDestroy(&m);CHKERRV(ierr);
}

Ensure(second_derivative)
{
  Mat m;
  DM  dm;
  PetscErrorCode ierr;

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,5,2,1,NULL,&dm);CHKERRV(ierr);
  ierr = DMCreateMatrix(dm, &m);CHKERRV(ierr);
  ierr = scFdAddSecondDerivative(dm, m, 1.0, 1.0, 1, 0);CHKERRV(ierr);
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
  ierr = MatView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(ierr);
  ierr = DMDestroy(&dm);CHKERRV(ierr);
  ierr = MatDestroy(&m);CHKERRV(ierr);
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, help);
  TestSuite *suite = create_test_suite();
  add_test(suite, first_derivative);
  add_test(suite, second_derivative);
  int result = run_test_suite(suite, create_text_reporter());
  PetscFinalize();
  return result;
}
