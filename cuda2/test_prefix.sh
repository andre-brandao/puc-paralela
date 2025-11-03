#!/bin/bash
# Hint: make this script executable with: chmod +x test_prefix.sh

echo "========================================="
echo "Prefix-Sum (Soma de Prefixos) Test"
echo "========================================="
echo ""

# Move to the script directory to ensure relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_SOURCE_FILE="soma_prefixos_gpu.cu"
GPU_BIN="soma_prefixos_gpu"
CPU_SOURCE_FILE="soma_prefixos_cpu.c"
CPU_BIN="soma_prefixos_cpu"

if [ ! -f "$GPU_SOURCE_FILE" ] || [ ! -f "$CPU_SOURCE_FILE" ]; then
  echo "✗ Required sources not found in $(pwd)"
  echo "  - GPU source: $GPU_SOURCE_FILE"
  echo "  - CPU source: $CPU_SOURCE_FILE"
  exit 1
fi

# Compile GPU version
echo "Compiling $GPU_SOURCE_FILE (CUDA)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc "$GPU_SOURCE_FILE" -O3 -o "$GPU_BIN"
  if [ $? -eq 0 ]; then
      echo "✓ GPU compilation successful"
  else
      echo "✗ GPU compilation failed"
      exit 1
  fi
else
  echo "✗ nvcc not found in PATH. Please install CUDA toolkit or add nvcc to PATH."
  exit 1
fi
echo ""

# Compile CPU version
echo "Compiling $CPU_SOURCE_FILE (CPU)"
if command -v gcc >/dev/null 2>&1; then
  gcc "$CPU_SOURCE_FILE" -O3 -o "$CPU_BIN"
  if [ $? -eq 0 ]; then
      echo "✓ CPU compilation successful"
  else
      echo "✗ CPU compilation failed"
      exit 1
  fi
else
  echo "✗ gcc not found in PATH."
  exit 1
fi

echo ""
echo "Running GPU and CPU tests"
echo "-------------------"
if [ ! -x "./$GPU_BIN" ]; then
  echo "✗ Binary '$GPU_BIN' not found or not executable."
  exit 1
fi
if [ ! -x "./$CPU_BIN" ]; then
  echo "✗ Binary '$CPU_BIN' not found or not executable."
  exit 1
fi

echo ""
echo "[GPU] Running $GPU_BIN"
GPU_OUT=$(./"$GPU_BIN")
GPU_STATUS=$?
if [ $GPU_STATUS -ne 0 ]; then
  echo "✗ GPU program exited with non-zero status ($GPU_STATUS)"
  echo "$GPU_OUT"
  exit $GPU_STATUS
fi

echo ""
echo "[CPU] Running $CPU_BIN"
CPU_OUT=$(./"$CPU_BIN")
CPU_STATUS=$?
if [ $CPU_STATUS -ne 0 ]; then
  echo "✗ CPU program exited with non-zero status ($CPU_STATUS)"
  echo "$CPU_OUT"
  exit $CPU_STATUS
fi
echo ""

# Parse outputs
GPU_IN=$(echo "$GPU_OUT" | grep '^Entrada:' | sed 's/^Entrada: //')
GPU_SCAN=$(echo "$GPU_OUT" | grep '^GPU:' | sed 's/^GPU: //')
CPU_IN=$(echo "$CPU_OUT" | grep '^Entrada:' | sed 's/^Entrada: //')
CPU_SCAN=$(echo "$CPU_OUT" | grep '^CPU:' | sed 's/^CPU: //')

echo "GPU Entrada: $GPU_IN"
echo "GPU Result : $GPU_SCAN"
echo "CPU Entrada: $CPU_IN"
echo "CPU Result : $CPU_SCAN"
echo ""

# Compare
COMPARE_SCAN=""
MATCH_ICON=""
if [ -n "$GPU_IN" ] && [ -n "$GPU_SCAN" ]; then
  if [ "$GPU_IN" = "$CPU_IN" ] && [ -n "$CPU_SCAN" ]; then
    if [ "$GPU_SCAN" = "$CPU_SCAN" ]; then
      COMPARE_SCAN="✓ GPU result matches CPU result (same input)"
      MATCH_ICON="✓"
    else
      COMPARE_SCAN="✗ GPU result differs from CPU result (same input)"
      MATCH_ICON="✗"
    fi
  else
    # Compute CPU reference from GPU input
    CPU_REF=$(echo "$GPU_IN" | awk '{
      n=split($0,a," ");
      s=0;
      for(i=1;i<=n;i++){ s+=a[i]; printf i<n? s" ": s}
      print ""
    }')
    if [ "$GPU_SCAN" = "$CPU_REF" ]; then
      COMPARE_SCAN="✓ GPU result matches CPU reference (computed from GPU input)"
      MATCH_ICON="✓"
    else
      COMPARE_SCAN="✗ GPU result differs from CPU reference (computed from GPU input)"
      MATCH_ICON="✗"
    fi
  fi
else
  COMPARE_SCAN="✗ Unable to parse GPU output."
  MATCH_ICON="✗"
fi

echo "-------------------"
echo "Summary:"
echo "  Result: $COMPARE_SCAN"
echo "-------------------"

echo "========================================="
echo ""



# CUDA Profiling (optional)
if command -v nvprof >/dev/null 2>&1; then
  echo "Profiling CUDA version with nvprof"
  echo "-------------------"
  echo ""
  echo "Event: warps_launched"
  nvprof --events warps_launched "./$GPU_BIN"
  echo ""
  echo "-------------------"
  echo ""
  echo "Metric: warp_execution_efficiency"
  nvprof --metrics warp_execution_efficiency "./$GPU_BIN"
  echo ""
else
  echo "nvprof not found. Skipping profiling."
  echo "Install NVIDIA nvprof (older CUDA) or consider using Nsight Systems/Compute."
fi

echo "========================================="
echo "Test completed!"
echo "========================================="
