

echo "Compiling silly_sort.c"
gcc silly_sort.c -O3 -o silly_sort -fopenmp -lm

echo "Testing silly_sort.c"
time ./silly_sort

echo "----------------"

echo "Compiling silly_sort-omp.c"
gcc silly_sort-omp.c -O3 -o silly_sort-omp -fopenmp -lm

echo "Testing silly_sort-omp.c"
time OMP_NUM_THREADS=4 ./silly_sort-omp

echo "----------------"

echo "Compiling silly_sort-gpu.c"
gcc silly_sort-gpu.c -O3 -o silly_sort-gpu -fopenmp  -lm

echo "Testing silly_sort-gpu.c"
time ./silly_sort-gpu
