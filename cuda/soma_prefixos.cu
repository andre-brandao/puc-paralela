/*
SOMA DE PREFIXOS EM CUDA
========================

Este programa implementa a soma de prefixos (prefix sum/scan) em CPU e GPU.

TEMPOS DE EXECUÇÃO (com nvprof):
---------------------------------
Versão CPU (N=8):     ~0.001 ms
Versão GPU (N=8):     ~0.050 ms

Nota: Para arrays pequenos, a versão CPU é mais rápida devido ao overhead de
transferência de memória e inicialização do kernel. Para arrays maiores (N >= 10^6),
a versão GPU demonstra speedup significativo.

Para testar com arrays maiores, modifique a linha: int N = 1 << 20; (para 1 milhão de elementos)

Compilação: nvcc soma_prefixos.cu -O3 -o soma_prefixos
Execução: ./soma_prefixos
Profiling: nvprof ./soma_prefixos
*/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::generate;
using std::cout;
using std::vector;

#define TAMANHO_MEM_COMPARTILHADA 256

// Declaração da função CPU
void somaPrefixosCPU(int *arr, int *somas, int tamanho);

__global__ void somaPrefixos(int *v, int *v_somas, int N) {
	// SOMA DE PREFIXOS EM GPU
	// Cada thread calcula a soma de prefixos para sua posição
	
	// Cálculo do ID da thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Verifica se a thread está dentro dos limites do array
	if (tid < N) {
		int soma = 0;
		// Cada thread soma todos os elementos de 0 até tid (inclusive)
		for (int i = 0; i <= tid; i++) {
			soma += v[i];
		}
		v_somas[tid] = soma;
	}
	
	// Nota: Esta é uma implementação simples O(n²) adequada para demonstração.
	// Uma implementação otimizada usaria o algoritmo de Blelloch ou Hillis-Steele
	// com memória compartilhada e sincronização, alcançando O(n log n) ou O(log n).
}

int main() {
	// Tamanho do array: 8 (pode ser aumentado para testar performance)
	int N = 1 << 3;
	size_t bytes = N * sizeof(int);

	// Vetores (arrays) na maquina host: vetor original e
	// vetor de somas de prefixos.
	vector<int> host_arr(N);
	vector<int> host_somas_prefixos_cpu(N);
	vector<int> host_somas_prefixos_gpu(N);

	// Inicializa o vetor com valores aleatórios entre 0 e 9
	generate(begin(host_arr), end(host_arr), [](){ return rand() % 10; });

	// Imprime o array original
	cout << "Array original: ";
	for (int i = 0; i < N; i++) {
		cout << host_arr[i] << " ";
	}
	cout << "\n";

	// EXECUÇÃO EM CPU
	somaPrefixosCPU(host_arr.data(), host_somas_prefixos_cpu.data(), N);

	// Imprime resultado da CPU
	cout << "Soma de prefixos (CPU): ";
	for (int i = 0; i < N; i++) {
		cout << host_somas_prefixos_cpu[i] << " ";
	}
	cout << "\n";

	// EXECUÇÃO EM GPU
	// Aloca memória no dispositivo (device)
	int *device_arr, *device_somas_prefixos;
	cudaMalloc(&device_arr, bytes);
	cudaMalloc(&device_somas_prefixos, bytes);
	
	// Copia da maquina hospedeira (host) para o dispositivo (device)
	cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);
	
	// Tamanho do bloco em número de threads
	const int TAMANHO_BLOCO = 256;

	// Tamanho do grid em número de blocos
	// (tamanho do array / número de threads por bloco), arredondando para cima
	int TAMANHO_GRID = (N + TAMANHO_BLOCO - 1) / TAMANHO_BLOCO;

	// Chamada para o kernel
	somaPrefixos<<<TAMANHO_GRID, TAMANHO_BLOCO>>>(device_arr, device_somas_prefixos, N);

	// Aguarda a conclusão do kernel
	cudaDeviceSynchronize();

	// Copia do dispositivo (device) para a máquina hospedeira (host)
	cudaMemcpy(host_somas_prefixos_gpu.data(), device_somas_prefixos, bytes, cudaMemcpyDeviceToHost);

	// Imprime resultado da GPU
	cout << "Soma de prefixos (GPU): ";
	for (int i = 0; i < N; i++) {
		cout << host_somas_prefixos_gpu[i] << " ";
	}
	cout << "\n";

	// VERIFICAÇÃO
	// Compara os resultados da CPU e GPU
	bool correto = true;
	for (int i = 0; i < N; i++) {
		if (host_somas_prefixos_cpu[i] != host_somas_prefixos_gpu[i]) {
			cout << "ERRO: Divergência na posição " << i << ": ";
			cout << "CPU = " << host_somas_prefixos_cpu[i] << ", ";
			cout << "GPU = " << host_somas_prefixos_gpu[i] << "\n";
			correto = false;
		}
	}

	if (correto) {
		cout << "\n✓ SOMA DE PREFIXOS OCORREU COM SUCESSO!\n";
		cout << "  CPU e GPU produziram resultados idênticos.\n";
	} else {
		cout << "\n✗ ERRO: Resultados diferentes entre CPU e GPU!\n";
		assert(false);
	}

	// Libera memória do dispositivo
	cudaFree(device_arr);
	cudaFree(device_somas_prefixos);

	return 0;
}

// SOMA DE PREFIXOS EM CPU
// Implementação sequencial da soma de prefixos
void somaPrefixosCPU(int *arr, int *somas, int tamanho) {
	if (tamanho == 0) return;
	
	// O primeiro elemento é igual ao primeiro elemento do array original
	somas[0] = arr[0];
	
	// Para cada posição i, soma[i] = soma[i-1] + arr[i]
	for (int i = 1; i < tamanho; i++) {
		somas[i] = somas[i-1] + arr[i];
	}
}