CC=gcc
NVCC=nvcc
INCFLAGS=-I/home/amd616/figtree/include
LDFLAGS=-L/home/amd616/figtree/lib -lm -lann_figtree_version -lfigtree
CFLAGS=-O3 

BIN=bin
SRC=src

EXEC = pot 

_DEPS = potential.h 
DEPS = $(patsubst %,$(SRC)/%,$(_DEPS))

_OBJ =  main.o read_pars.o kde.o output.o potential.o
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
