#include <fft.h>
#include <petscdmda.h>

struct ScFft_ {
  DM da;
  Mat matFft;
};

#undef __FUNCT__
#define __FUNCT__ "scFftCreate"
PetscErrorCode scFftCreate(DM da, ScFft *fft)
{
  PetscErrorCode ierr;
  PetscInt       dim, dims[3];

  PetscFunctionBegin;
  if (!da) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Invalid da.");
  }
  ierr = DMDAGetInfo(da, &dim, &dims[0], &dims[1], &dims[2], 0, 0, 0, 0, 0, 0, 0, 0, 0);CHKERRQ(ierr);
  *fft = malloc(sizeof(**fft)); 
  (*fft)->da = da;
  ierr = MatCreateFFT(PETSC_COMM_WORLD, dim, dims, MATFFTW, &(*fft)->matFft);CHKERRQ(ierr);
  ierr = MatSetType((*fft)->matFft, MATFFTW);CHKERRQ(ierr);
  ierr = MatSetUp((*fft)->matFft);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftDestroy"
PetscErrorCode scFftDestroy(ScFft *fft)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&(*fft)->matFft);CHKERRQ(ierr);
  free(*fft);
  *fft = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftGetDM"
PetscErrorCode scFftGetDM(ScFft fft, DM *da)
{
  PetscFunctionBegin;
  *da = fft->da;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftTransform"
PetscErrorCode scFftTransform(Vec v, PetscInt i, Vec y)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "scFftCreateVecsFFTW"
PetscErrorCode scFftCreateVecsFFTW(ScFft fft, Vec *x, Vec *y, Vec *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecsFFTW(fft->matFft, x, y, z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
