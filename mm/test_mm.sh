#!/bin/bash

echo "========================================="
echo "Matrix Multiplication Performance Test"
echo "========================================="
echo ""

# CPU Sequential Version
echo "Compiling mm.c (CPU Sequential version)"
gcc mm.c -O3 -o mm -lm
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi

echo ""
echo "Testing mm.c (CPU Sequential version)"
echo "-------------------"
time ./mm
echo ""

echo "========================================="
echo ""

# OpenMP Version
echo "Compiling mm-omp.c (OpenMP version)"
gcc mm-omp.c -O3 -o mm-omp -fopenmp -lm
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi

echo ""
echo "Testing mm-omp.c (OpenMP version)"
echo "-------------------"
time ./mm-omp
echo ""

echo "========================================="
echo ""

# CUDA Version
echo "Compiling mm-cuda.cu (CUDA version)"
nvcc mm-cuda.cu -O3 -o mm-cuda
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi

echo ""
echo "Testing mm-cuda.cu (CUDA version)"
echo "-------------------"
time ./mm-cuda
echo ""

echo "========================================="
echo ""

# CUDA Profiling
echo "Profiling CUDA version with nvprof"
echo "-------------------"
echo ""
echo "Event: warps_launched"
nvprof --events warps_launched ./mm-cuda
echo ""
echo "-------------------"
echo ""
echo "Metric: warp_execution_efficiency"
nvprof --metrics warp_execution_efficiency ./mm-cuda
echo ""

echo "========================================="
echo "Test completed!"
echo "========================================="
