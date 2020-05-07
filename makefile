.POSIX:
CC     = cc -std=c99
CFLAGS = -Wall -Wextra -O3 -g3
LDLIBS = -lm
N = 1000

all: cudaC4 pgrkcuda pgrkmpi pgrkserial pgrkomp

# recipes for connect four
cudaC4:
	nvcc -o ./bin/c4_cuda ./connect4/connect4.cu -lm

run:
	./bin/c4_cuda

connect4: connect4.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ connect4.c $(LDLIBS)

clean:
	rm -f connect4

# recipes for pagerank in serial, cuda, omp, mpi
pgrkserial:
	gcc -o ./bin/pgrkserial ./pagerank/pagerankserial.c

pgrkomp:
	gcc -o ./bin/pgrkomp -fopenmp ./pagerank/pagerankomp.c

pgrkcuda:
	nvcc -o ./bin/pgrkcuda ./pagerank/pagerankcuda.cu	

pgrkmpi:
	mpicc -o ./bin/pgrkmpi ./pagerank/pagerankmpi.c


# test to make sure cuda run well. Output should be 'Max error: 0.000000'
testcuda:
	nvcc -o ./bin/cudatest ./connect4/test_saxpy.cu
	./bin/cudatest


# recipes for running binaries
runmpi:
	mpiexec ./bin/pgrkmpi $(N)

runomp:
	./bin/pgrkomp $(N)

runserial:
	./bin/pgrkserial $(N)

runcuda:
	./bin/pgrkcuda $(N)

runall: runserial runmpi runomp runcuda

