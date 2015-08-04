PETSC_DIR=/home/scratch/petsc/float-rel
#PETSC_DIR=/home/scratch/petsc/float-dbg
include $(PETSC_DIR)/lib/petsc/conf/petscvariables
CC=/usr/gcc_trunk/bin/gcc
#CFLAGS+=$(PETSC_CC_INCLUDES) -g -O0 -fdiagnostics-color=auto -Wall
CFLAGS+=$(PETSC_CC_INCLUDES) -O3 -ffast-math -march=native -fdiagnostics-color=auto -Wall
LDFLAGS+=$(PETSC_LIB_BASIC) $(PETSC_EXTERNAL_LIB_BASIC)
LDLIBS+=$(PETSC_LIB_BASIC) $(PETSC_EXTERNAL_LIB_BASIC)

all: npp

.PHONY: clean
clean:
	rm -rf npp *h5
