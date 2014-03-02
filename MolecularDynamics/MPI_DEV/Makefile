MAKER         = mpicc
OPTFLAGS    =  -O
DEBUG       = 
LIB_LIST    = -lm 

SUFFIX = c

PROGRAM = moldyn

SRCS  = main.c    input-parameters.c read-input.c set-cell-neighbours.c \
        cold-start.c initialise-particles.c flush-buffer.c tohost.c     \
        pack-buffer.c set-chunk.c pseudorand.c outcon.c savcon.c        \
        loop-initialise.c force.c movout.c edgadd.c edgbuf.c migbuf.c   \
        lstadd.c pasbuf.c movea.c moveb.c sum-energies.c hloop.c scalet.c \
        tidyup.c edgprint.c

### End User configurable options ###

FFLAGS =  $(INCLUDE_DIR) $(OPTFLAGS)
LIBS = $(LIB_PATH) $(LIB_LIST)
FLIBS = $(FLIB_PATH) $(LIB_LIST)

OBJS = $(SRCS:.$(SUFFIX)=.o)

${PROGRAM}: ${OBJS}
	$(MAKER) $(OPTFLAGS) -o $(PROGRAM) $(OBJS) $(FLIBS)

${OBJS}:
	${MAKER} ${DEBUG} ${OPTFLAGS} ${INCLUDE_DIR} -c $*.${SUFFIX}

clean:
	rm -f ${OBJS}

Clean:
	rm -f ${PROGRAM} ${OBJS}
