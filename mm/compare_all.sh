#!/bin/bash

echo "========================================="
echo "Matrix Multiplication - Complete Comparison"
echo "========================================="
echo ""
echo "This script compares three implementations:"
echo "  1. CPU Sequential (mm.c)"
echo "  2. CPU Parallel OpenMP (mm-omp.c)"
echo "  3. GPU CUDA (mm-cuda.cu)"
echo ""
echo "========================================="
echo ""

# Create results file
RESULTS_FILE="results_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to extract real time from time output
extract_time() {
    grep "real" | awk '{print $2}'
}

# ==========================================
# 1. CPU Sequential Version
# ==========================================
echo "=========================================" | tee -a $RESULTS_FILE
echo "1. CPU Sequential Version (mm.c)" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE

echo "Compiling..."
gcc mm.c -O3 -o mm -lm
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful" | tee -a $RESULTS_FILE
else
    echo "✗ Compilation failed" | tee -a $RESULTS_FILE
    exit 1
fi

echo "" | tee -a $RESULTS_FILE
echo "Running..." | tee -a $RESULTS_FILE
echo "-------------------" | tee -a $RESULTS_FILE
TIME_CPU=$( { time ./mm; } 2>&1 | grep real | awk '{print $2}' )
echo "Execution time: $TIME_CPU" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ==========================================
# 2. OpenMP Version
# ==========================================
echo "=========================================" | tee -a $RESULTS_FILE
echo "2. CPU Parallel OpenMP Version (mm-omp.c)" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE

echo "Compiling..."
gcc mm-omp.c -O3 -o mm-omp -fopenmp -lm
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful" | tee -a $RESULTS_FILE
else
    echo "✗ Compilation failed" | tee -a $RESULTS_FILE
    exit 1
fi

echo "" | tee -a $RESULTS_FILE
echo "Running..." | tee -a $RESULTS_FILE
echo "Number of OpenMP threads: $OMP_NUM_THREADS (set OMP_NUM_THREADS to change)" | tee -a $RESULTS_FILE
echo "-------------------" | tee -a $RESULTS_FILE
TIME_OMP=$( { time ./mm-omp; } 2>&1 | grep real | awk '{print $2}' )
echo "Execution time: $TIME_OMP" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ==========================================
# 3. CUDA Version
# ==========================================
echo "=========================================" | tee -a $RESULTS_FILE
echo "3. GPU CUDA Version (mm-cuda.cu)" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE

echo "Compiling..."
nvcc mm-cuda.cu -O3 -o mm-cuda
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful" | tee -a $RESULTS_FILE
else
    echo "✗ Compilation failed" | tee -a $RESULTS_FILE
    echo "Note: nvcc (CUDA compiler) must be installed" | tee -a $RESULTS_FILE
    exit 1
fi

echo "" | tee -a $RESULTS_FILE
echo "Running..." | tee -a $RESULTS_FILE
echo "-------------------" | tee -a $RESULTS_FILE
TIME_CUDA=$( { time ./mm-cuda; } 2>&1 | grep real | awk '{print $2}' )
echo "Execution time: $TIME_CUDA" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ==========================================
# 4. CUDA Profiling
# ==========================================
echo "=========================================" | tee -a $RESULTS_FILE
echo "4. CUDA Profiling with nvprof" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

echo "--- Event: warps_launched ---" | tee -a $RESULTS_FILE
nvprof --events warps_launched ./mm-cuda 2>&1 | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

echo "--- Metric: warp_execution_efficiency ---" | tee -a $RESULTS_FILE
nvprof --metrics warp_execution_efficiency ./mm-cuda 2>&1 | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

echo "--- Additional Metrics ---" | tee -a $RESULTS_FILE
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./mm-cuda 2>&1 | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ==========================================
# Summary
# ==========================================
echo "=========================================" | tee -a $RESULTS_FILE
echo "SUMMARY" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
echo "Matrix size: 2000x2000" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
echo "Execution Times:" | tee -a $RESULTS_FILE
echo "  CPU Sequential:  $TIME_CPU" | tee -a $RESULTS_FILE
echo "  CPU OpenMP:      $TIME_OMP" | tee -a $RESULTS_FILE
echo "  GPU CUDA:        $TIME_CUDA" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
echo "Results saved to: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE
echo "Comparison Complete!" | tee -a $RESULTS_FILE
echo "=========================================" | tee -a $RESULTS_FILE
