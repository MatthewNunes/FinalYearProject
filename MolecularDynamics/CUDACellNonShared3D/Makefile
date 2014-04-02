
MAKER         = nvcc -gencode arch=compute_20,code=sm_20
OPTFLAGS    = -O
//PTXASFLAGS = --ptxas-options=-v
PTXASFLAGS =
DEBUG       = 
LIB_LIST    = -lm 

SUFFIX = cu

PROGRAM = moldyn

SRCS  = main.cu    input-parameters.cu read-input.cu \
        initialise-particles.cu pseudorand.cu loop-initialise.cu force.cu maxParticles.cu movout.cu \
        movea.cu moveb.cu sum-energies.cu hloop.cu scalet.cu tidyup.cu check_cells.cu output_particles.cu force_ij.cu

### End User configurable options ###

FFLAGS =  $(INCLUDE_DIR) $(OPTFLAGS)
LIBS = $(LIB_PATH) $(LIB_LIST)
FLIBS = $(FLIB_PATH) $(LIB_LIST)

OBJS = $(SRCS:.$(SUFFIX)=.o)

${PROGRAM}: ${OBJS}
	$(MAKER) $(OPTFLAGS) -o $(PROGRAM) $(OBJS) $(FLIBS)

${OBJS}:
	${MAKER} ${DEBUG} ${PTXASFLAGS} ${OPTFLAGS} ${INCLUDE_DIR} -c $*.${SUFFIX}

clean:
	rm -f ${OBJS}

Clean:
	rm -f ${PROGRAM} ${OBJS}
