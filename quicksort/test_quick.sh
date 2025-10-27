

echo "Compiling quicksort.c"
gcc quicksort.c -O3 -o quicksort -fopenmp -lm

echo "Testing quicksort.c"
time ./quicksort

echo "----------------"

echo "Compiling quicksort-omp.c"
gcc quicksort-omp.c -O3 -o quicksort-omp -fopenmp -lm

echo "Testing quicksort-omp.c"
time ./quicksort-omp
