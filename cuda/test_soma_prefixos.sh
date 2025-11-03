#!/bin/bash

echo "========================================"
echo "TESTE DE SOMA DE PREFIXOS"
echo "Comparação: CPU (C) vs GPU (CUDA)"
echo "========================================"
echo ""

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ========================================
# VERSÃO 1: CPU SEQUENCIAL EM C
# ========================================
echo -e "${BLUE}[1/6] Compilando soma_prefixos_cpu.c (CPU sequencial)${NC}"
gcc soma_prefixos_cpu.c -O3 -o soma_prefixos_cpu -lm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilação bem-sucedida${NC}"
else
    echo "✗ Erro na compilação"
    exit 1
fi

echo ""
echo -e "${BLUE}[2/6] Testando soma_prefixos_cpu (N=1048576)${NC}"
echo "----------------------------------------"
time ./soma_prefixos_cpu
echo ""

echo "========================================"
echo ""

# ========================================
# VERSÃO 2: GPU CUDA SIMPLES
# ========================================
echo -e "${BLUE}[3/6] Compilando soma_prefixos.cu (GPU versão simples)${NC}"
nvcc soma_prefixos.cu -O3 -o soma_prefixos
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilação bem-sucedida${NC}"
else
    echo "✗ Erro na compilação"
    exit 1
fi

echo ""
echo -e "${BLUE}[4/6] Testando soma_prefixos (N=8 - demonstração)${NC}"
echo "----------------------------------------"
nvprof ./soma_prefixos
echo ""

echo "========================================"
echo ""

# ========================================
# VERSÃO 3: GPU CUDA OTIMIZADA
# ========================================
echo -e "${BLUE}[5/6] Compilando soma_prefixos_otimizado.cu (GPU otimizada)${NC}"
nvcc soma_prefixos_otimizado.cu -O3 -o soma_prefixos_otimizado
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilação bem-sucedida${NC}"
else
    echo "✗ Erro na compilação"
    exit 1
fi

echo ""
echo -e "${BLUE}[6/6] Testando soma_prefixos_otimizado (N=1048576)${NC}"
echo "----------------------------------------"
nvprof ./soma_prefixos_otimizado
echo ""

echo "========================================"
echo -e "${GREEN}TODOS OS TESTES CONCLUÍDOS!${NC}"
echo "========================================"
echo ""
echo -e "${YELLOW}RESUMO:${NC}"
echo "  1. soma_prefixos_cpu      - Implementação sequencial em C"
echo "  2. soma_prefixos          - CUDA versão simples (demonstração)"
echo "  3. soma_prefixos_otimizado - CUDA com memória compartilhada"
echo ""
echo "Para profiling detalhado com nvprof, execute:"
echo "  nvprof ./soma_prefixos"
echo "  nvprof ./soma_prefixos_otimizado"
echo ""
echo "Para comparar apenas CPU vs GPU otimizada:"
echo "  time ./soma_prefixos_cpu && time ./soma_prefixos_otimizado"
echo ""
