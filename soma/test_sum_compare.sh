echo "Compiling sum_cuda-1.cu"
nvcc sum_cuda-1.cu -O3 -o sum_shared

echo "Testing sum_shared"
time ./sum_shared

echo "----------------"

echo "Compiling sum_cuda-1_noshared.cu"
nvcc sum_cuda-1_noshared.cu -O3 -o sum_noshared

echo "Testing sum_noshared"
time ./sum_noshared

echo "----------------"

if command -v nvprof >/dev/null 2>&1; then
  echo "Profiling sum_shared (nvprof)"
  nvprof ./sum_shared

  echo "----------------"

  echo "Profiling sum_noshared (nvprof)"
  nvprof ./sum_noshared
else
  echo "nvprof not found; skipping profiling runs"
fi
