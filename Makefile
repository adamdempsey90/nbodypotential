CC=gcc
NVCC=nvcc
INCFLAGS=-I/home/amd616/figtree/include -I/software/gsl/1.15RH6/include # -I$(C_INCLUDE_PATH)
#LDFLAGS=-L/home/amd616/figtree/lib -lgsl -lm -lann_figtree_version -lfigtree
LDFLAGS=-L/software/gsl/1.15RH6/lib -L/home/amd616/figtree/lib -lgsl -lgslcblas -lm -lann_figtree_version -lfigtree
#LDLOCS=$(subst :, -L,$(LD_LIBRARY_PATH))
CFLAGS=-O3 
LIBRARY_PATH=$(LD_LIBRARY_PATH)

BIN=bin
SRC=src

EXEC = pot 

_DEPS = potential.h 
DEPS = $(patsubst %,$(SRC)/%,$(_DEPS))

_OBJ =  main.o read_pars.o kde.o output.o solve.o potential.o
OBJ = $(patsubst %,$(BIN)/%,$(_OBJ))


$(BIN)/%.o: $(SRC)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCFLAGS)

$(BIN)/%.o: $(SRC)/%.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS) $(INCFLAGS)

$(EXEC): $(OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)


.PHONY: clean
clean:
	rm -f $(OBJ) $(EXEC) 
