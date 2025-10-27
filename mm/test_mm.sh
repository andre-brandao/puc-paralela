#!/bin/bash

echo "=========================================="
echo "Matrix Multiplication Performance Tests"
echo "Intel Iris Plus Graphics G7"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}[1/6] Compiling mm.c (sequential version)${NC}"
gcc mm.c -O3 -o mm -lm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm.c (Sequential)${NC}"
    time ./mm
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo ""

echo -e "${BLUE}[2/6] Compiling mm-omp.c (OpenMP CPU version)${NC}"
gcc mm-omp.c -O3 -o mm-omp -fopenmp -lm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm-omp.c with 4 threads${NC}"
    time ./mm-omp
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo -e "${BLUE}GPU VERSIONS - Intel Iris Plus Graphics G7${NC}"
echo "=========================================="

# Try to detect Intel compiler or use GCC with offloading
COMPILER=""
COMPILE_FLAGS=""

if command -v icx &> /dev/null; then
    COMPILER="icx"
    COMPILE_FLAGS="-O3 -fiopenmp -fopenmp-targets=spir64 -lm"
    echo -e "${GREEN}Using Intel icx compiler with OpenMP offloading${NC}"
elif command -v icpx &> /dev/null; then
    COMPILER="icpx"
    COMPILE_FLAGS="-O3 -fiopenmp -fopenmp-targets=spir64 -lm"
    echo -e "${GREEN}Using Intel icpx compiler with OpenMP offloading${NC}"
elif command -v clang &> /dev/null; then
    COMPILER="clang"
    COMPILE_FLAGS="-O3 -fopenmp -fopenmp-targets=spir64 -lm"
    echo -e "${YELLOW}Using Clang with OpenMP offloading (may need Level Zero support)${NC}"
else
    COMPILER="gcc"
    COMPILE_FLAGS="-O3 -fopenmp -foffload=disable -lm"
    echo -e "${YELLOW}Warning: No Intel compiler found. Using GCC without GPU offloading${NC}"
    echo -e "${YELLOW}For Intel GPU support, install Intel oneAPI toolkit${NC}"
fi

echo ""
echo -e "${BLUE}[3/6] Compiling mm-gpu-distribute.c${NC}"
echo -e "      ${YELLOW}(Using: target teams distribute)${NC}"
$COMPILER mm-gpu-distribute.c $COMPILE_FLAGS -o mm-gpu-distribute 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm-gpu-distribute${NC}"
    time ./mm-gpu-distribute
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo ""

echo -e "${BLUE}[4/6] Compiling mm-gpu.c${NC}"
echo -e "      ${YELLOW}(Using: target teams distribute parallel for)${NC}"
$COMPILER mm-gpu.c $COMPILE_FLAGS -o mm-gpu 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm-gpu${NC}"
    time ./mm-gpu
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo ""

echo -e "${BLUE}[5/6] Compiling mm-gpu-distribute-parallel.c${NC}"
echo -e "      ${YELLOW}(Using: target teams distribute parallel for collapse(2))${NC}"
$COMPILER mm-gpu-distribute-parallel.c $COMPILE_FLAGS -o mm-gpu-distribute-parallel 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm-gpu-distribute-parallel${NC}"
    time ./mm-gpu-distribute-parallel
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo ""

echo -e "${BLUE}[6/6] Compiling mm-gpu-distribute-parallel-simd.c${NC}"
echo -e "      ${YELLOW}(Using: target teams distribute parallel for simd collapse(2))${NC}"
$COMPILER mm-gpu-distribute-parallel-simd.c $COMPILE_FLAGS -o mm-gpu-distribute-parallel-simd 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo -e "${YELLOW}Testing mm-gpu-distribute-parallel-simd${NC}"
    time ./mm-gpu-distribute-parallel-simd
else
    echo -e "${RED}✗ Compilation failed${NC}"
fi

echo ""
echo "=========================================="
echo -e "${BLUE}GPU PROFILING AND METRICS${NC}"
echo "=========================================="

echo ""
echo -e "${YELLOW}Checking for Intel profiling tools...${NC}"

# Check for Intel VTune
if command -v vtune &> /dev/null; then
    echo -e "${GREEN}✓ Intel VTune found${NC}"
    echo ""
    echo -e "${BLUE}Profiling with VTune (GPU Offload analysis)${NC}"
    echo "Running: vtune -collect gpu-offload -result-dir vtune_results ./mm-gpu-distribute-parallel"
    vtune -collect gpu-offload -result-dir vtune_results ./mm-gpu-distribute-parallel
    echo ""
    echo "To view results: vtune-gui vtune_results"
elif command -v advisor &> /dev/null; then
    echo -e "${GREEN}✓ Intel Advisor found${NC}"
    echo ""
    echo -e "${BLUE}Profiling with Advisor (Offload modeling)${NC}"
    advisor --collect=offload --project-dir=./advisor_results -- ./mm-gpu-distribute-parallel
    echo ""
    echo "To view results: advisor-gui ./advisor_results"
else
    echo -e "${YELLOW}⚠ No Intel profiling tools found (VTune or Advisor)${NC}"
    echo ""
    echo "For Intel GPU profiling, install Intel oneAPI Base Toolkit:"
    echo "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
fi

# Check for OpenCL profiling
if command -v clinfo &> /dev/null; then
    echo ""
    echo -e "${BLUE}OpenCL Device Information:${NC}"
    clinfo | grep -E "(Platform|Device Type|Device Name|Max compute units|Max work group size|Global memory)" | head -20
fi

# Environment variables for profiling
echo ""
echo -e "${BLUE}GPU Profiling with Environment Variables:${NC}"
echo ""
echo -e "${YELLOW}Running with LIBOMPTARGET_INFO=1 for debugging:${NC}"
if [ -f ./mm-gpu-distribute-parallel ]; then
    LIBOMPTARGET_INFO=1 ./mm-gpu-distribute-parallel 2>&1 | grep -i "device\|target\|kernel\|thread" | head -20
fi

echo ""
echo -e "${YELLOW}Running with OMP_TARGET_OFFLOAD=MANDATORY:${NC}"
if [ -f ./mm-gpu-distribute-parallel ]; then
    OMP_TARGET_OFFLOAD=MANDATORY ./mm-gpu-distribute-parallel 2>&1 | head -5
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPU offload successful${NC}"
    else
        echo -e "${RED}✗ GPU offload failed - may be running on CPU${NC}"
    fi
fi

echo ""
echo "=========================================="
echo -e "${BLUE}PERFORMANCE SUMMARY${NC}"
echo "=========================================="

echo ""
echo "OpenMP Directives Tested:"
echo "1. distribute                           - Distributes iterations across teams"
echo "2. distribute parallel for              - Adds parallelism within teams"
echo "3. distribute parallel for collapse(2)  - Parallelizes nested loops"
echo "4. distribute parallel for simd         - Adds SIMD vectorization"
echo ""
echo "Note: For accurate GPU metrics on Intel GPUs, use:"
echo "  - Intel VTune Profiler (vtune)"
echo "  - Intel Advisor (advisor)"
echo "  - Set LIBOMPTARGET_INFO=1 for debug output"
echo ""
echo "=========================================="
echo -e "${GREEN}Testing completed${NC}"
echo "=========================================="
