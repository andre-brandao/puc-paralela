#!/bin/sh


# echo "Compiling mm.c"
# gcc mm.c -o mm

# echo "Testing mm.c"
# time ./mm


# echo "Compiling mm-omp.c"
# gcc mm-omp.c -o mm-omp -fopenmp

# echo "Testing mm-omp.c"
# time OMP_NUM_THREADS=4 ./mm-omp

echo "Compiling sieve.c"
gcc sieve.c -o sieve -fopenmp -lm

echo "Testing sieve.c"
# time ./sieve
perf stat -d ./sieve

echo "----------------"

echo "Compiling sieve-omp.c"
gcc sieve-omp.c -o sieve-omp -fopenmp -lm

echo "Testing sieve-omp.c"
OMP_NUM_THREADS=2 perf stat -d ./sieve-omp
# time OMP_NUM_THREADS=4 ./sieve-omp
