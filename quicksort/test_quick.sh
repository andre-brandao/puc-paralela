

echo "Compiling quicksort.c"
gcc quicksort.c -o3 quicksort -fopenmp -lm

echo "Testing quicksort.c"
time ./quicksort

echo "----------------"

echo "Compiling quicksort-omp.c"
gcc quicksort-omp.c -o3 quicksort-omp -fopenmp -lm

echo "Testing quicksort-omp.c"
time ./quicksort-omp
