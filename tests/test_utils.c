#include <test_utils.h>

PetscErrorCode scFdSetup(struct FDFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fixture->dim = 10;
  fixture->numComponents = 2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, fixture->dim, fixture->numComponents, 2, NULL, &fixture->da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(fixture->da,&fixture->info);CHKERRQ(ierr);
  ierr = DMCreateMatrix(fixture->da, &fixture->m);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fixture->da, &fixture->x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fixture->da, &fixture->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode scFdTeardown(struct FDFixture* fixture) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDestroy(&fixture->da);CHKERRQ(ierr);
  ierr = MatDestroy(&fixture->m);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->x);CHKERRQ(ierr);
  ierr = VecDestroy(&fixture->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

