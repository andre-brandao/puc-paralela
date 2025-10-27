#!/bin/bash

echo "========================================="
echo "CUDA Matrix Multiplication Profiling"
echo "========================================="
echo ""

# Compile CUDA version if needed
if [ ! -f mm-cuda ]; then
    echo "Compiling mm-cuda.cu..."
    nvcc mm-cuda.cu -O3 -o mm-cuda
    if [ $? -ne 0 ]; then
        echo "✗ Compilation failed"
        exit 1
    fi
    echo "✓ Compilation successful"
    echo ""
fi

echo "========================================="
echo "Basic Execution Time"
echo "========================================="
time ./mm-cuda
echo ""

echo "========================================="
echo "Event: warps_launched"
echo "========================================="
nvprof --events warps_launched ./mm-cuda
echo ""

echo "========================================="
echo "Metric: warp_execution_efficiency"
echo "========================================="
nvprof --metrics warp_execution_efficiency ./mm-cuda
echo ""

echo "========================================="
echo "Additional Useful Metrics"
echo "========================================="
echo ""

echo "--- Global Memory Load/Store Efficiency ---"
nvprof --metrics gld_efficiency,gst_efficiency ./mm-cuda
echo ""

echo "--- Achieved Occupancy ---"
nvprof --metrics achieved_occupancy ./mm-cuda
echo ""

echo "--- IPC (Instructions Per Cycle) ---"
nvprof --metrics ipc ./mm-cuda
echo ""

echo "--- FLOPS (Floating Point Operations) ---"
nvprof --metrics flop_count_dp,flop_dp_efficiency ./mm-cuda
echo ""

echo "========================================="
echo "Summary Statistics"
echo "========================================="
nvprof --print-gpu-summary ./mm-cuda
echo ""

echo "========================================="
echo "Profiling Complete!"
echo "========================================="
