.POSIX:
CC     = cc -std=c99
CFLAGS = -Wall -Wextra -O3 -g3
LDLIBS = -lm


cudaC4:
	nvcc -o ./bin/c4_cuda ./core/connect4.cu -lm

run:
	./bin/c4_cuda

test:
	gcc -std=c99 -Wall -o ./bin/c4_c ./core/connect4.c -lm
	./bin/c4_c

# test to make sure cuda run well. Output should be 'Max error: 0.000000'
testcuda:
	nvcc -o ./bin/cudatest ./core/test_saxpy.cu
	./bin/cudatest

connect4: connect4.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ connect4.c $(LDLIBS)

clean:
	rm -f connect4