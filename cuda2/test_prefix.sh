echo "Compiling soma_prefixos_gpu.cu"
nvcc soma_prefixos_gpu.cu -O3 -o soma_prefixos_gpu

echo "Testing soma_prefixos_gpu"
time ./soma_prefixos_gpu

echo "----------------"

echo "Compiling soma_prefixos_cpu.c"
gcc soma_prefixos_cpu.c -O3 -o soma_prefixos_cpu

echo "Testing soma_prefixos_cpu"
time ./soma_prefixos_cpu

echo "----------------"

if command -v nvprof >/dev/null 2>&1; then
  echo "Profiling soma_prefixos_gpu (nvprof)"
  nvprof ./soma_prefixos_gpu
else
  echo "nvprof not found; skipping profiling run"
fi
