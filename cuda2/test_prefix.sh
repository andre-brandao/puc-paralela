#!/bin/bash
# Hint: make this script executable with: chmod +x test_prefix.sh

echo "========================================="
echo "Prefix-Sum (Soma de Prefixos) Test"
echo "========================================="
echo ""

# Move to the script directory to ensure relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE_FILE="soma_prefixos.cu"
BIN_NAME="soma_prefixos"

if [ ! -f "$SOURCE_FILE" ]; then
  echo "✗ Source file '$SOURCE_FILE' not found in $(pwd)"
  exit 1
fi

# CUDA Version
echo "Compiling $SOURCE_FILE (CUDA version)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc "$SOURCE_FILE" -O3 -o "$BIN_NAME"
  if [ $? -eq 0 ]; then
      echo "✓ Compilation successful"
  else
      echo "✗ Compilation failed"
      exit 1
  fi
else
  echo "✗ nvcc not found in PATH. Please install CUDA toolkit or add nvcc to PATH."
  exit 1
fi

echo ""
echo "Testing $SOURCE_FILE (CUDA version)"
echo "-------------------"
if [ -x "./$BIN_NAME" ]; then
  time "./$BIN_NAME"
  RUN_STATUS=$?
  if [ $RUN_STATUS -ne 0 ]; then
    echo "✗ Program exited with non-zero status ($RUN_STATUS)"
    exit $RUN_STATUS
  fi
else
  echo "✗ Binary '$BIN_NAME' not found or not executable."
  exit 1
fi
echo ""

echo "========================================="
echo ""

# CUDA Profiling (optional)
if command -v nvprof >/dev/null 2>&1; then
  echo "Profiling CUDA version with nvprof"
  echo "-------------------"
  echo ""
  echo "Event: warps_launched"
  nvprof --events warps_launched "./$BIN_NAME"
  echo ""
  echo "-------------------"
  echo ""
  echo "Metric: warp_execution_efficiency"
  nvprof --metrics warp_execution_efficiency "./$BIN_NAME"
  echo ""
else
  echo "nvprof not found. Skipping profiling."
  echo "Install NVIDIA nvprof (older CUDA) or consider using Nsight Systems/Compute."
fi

echo "========================================="
echo "Test completed!"
echo "========================================="
