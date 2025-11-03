/*
Neste exercício, você deve implementar uma soma de prefixos em CUDA.

Você deve completar o código em três pontos:

## SOMA DE PREFIXOS EM CPU ##

Implemente a soma de prefixos em CPU e execute sobre o vetor original.
Utilize a saída do algoritmo síncrono (em CPU) para verificar a
corretude da saída do algoritmo paralelo (em GPU)

## SOMA DE PREFIXOS EM GPU ##

Utilize seus conhecimentos em CUDA para implementar uma estratégia
de indexação de threads para computar a soma de prefixos em paralelo.

## VERIFICAÇÃO ##

Implemente algum tipo de verificação. Sugestão: confira
se a saída do algoritmo síncrono (CPU) bate com a saída do
algoritmo paralelo (GPU)
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

void somaPrefixosCPU(const int *arr, int *somas, int tamanho);

__global__ void somaPrefixos(int *v, int *v_somas) {
	// OBSERVAÇÃO: Não é necessário implementar
	// utilizando memória compartilhada, apesar de ser
	// a alternativa ótima (para que threads de um mesmo bloco)
	// não precisem acessar a memória global reiteradas vezes.

	// Cálculo do ID da thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// ## SOMA DE PREFIXOS EM GPU ##
	// Implementação básica (Hillis-Steele) dentro de um único bloco.
	// Requer que gridDim.x == 1 e blockDim.x <= TAMANHO_MEM_COMPARTILHADA.
	__shared__ int temp[TAMANHO_MEM_COMPARTILHADA];
	int local = threadIdx.x;

	// Carrega da memória global para a compartilhada
	temp[local] = v[tid];
	__syncthreads();

	for (int offset = 1; offset < blockDim.x; offset <<= 1) {
		int add = (local >= offset) ? temp[local - offset] : 0;
		__syncthreads();
		temp[local] += add;
		__syncthreads();
	}

	// Escreve o resultado (soma prefixos inclusiva)
	v_somas[tid] = temp[local];
}

int main() {
	// Tamanho do array: 8
	int N = 1 << 3;
	size_t bytes = N * sizeof(int);

	// Vetores (arrays) na maquina host: vetor original e
    // vetor reduzido.
	vector<int> host_arr(N);
	vector<int> host_somas_prefixos(N);

    // Inicializa o vetor , i. e., aloca memória na maquina host
    for (int i = 0; i < N; ++i) host_arr[i] = rand() % 10;

	// Aloca memória no dispositivo (device)
	int *device_arr, *device_somas_prefixos;
	cudaMalloc(&device_arr, bytes);
	cudaMalloc(&device_somas_prefixos, bytes);

	// Copia da maquina hospedeira (host) para o dispositivo (device)
	cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);

	// Tamanho do bloco em número de threads
	const int TAMANHO_BLOCO = 8;

	// Tamanho do grid em número de bloco
    // (tamanho do array / número de threads por bloco)
	int TAMANHO_GRID = N / TAMANHO_BLOCO;
	assert(TAMANHO_BLOCO <= TAMANHO_MEM_COMPARTILHADA);
	assert(TAMANHO_GRID == 1);

	// Chamadas para o kernel
	somaPrefixos<<<TAMANHO_GRID, TAMANHO_BLOCO>>>(device_arr, device_somas_prefixos);
	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);

	// Copia do dispositivo (device) para a máquina hospedeira (host)
	cudaMemcpy(host_somas_prefixos.data(), device_somas_prefixos, bytes, cudaMemcpyDeviceToHost);

	// Confere resultado
	// ## VERIFICAÇÃO ##
	// Calcula soma de prefixos em CPU e compara com resultado da GPU
	vector<int> host_somas_prefixos_cpu(N);
	somaPrefixosCPU(host_arr.data(), host_somas_prefixos_cpu.data(), N);

	bool ok = std::equal(host_somas_prefixos.begin(),
	                     host_somas_prefixos.end(),
	                     host_somas_prefixos_cpu.begin());

	if (!ok) {
		cout << "Erro: resultado da GPU difere do da CPU.\n";
		cout << "Vetor original: ";
		for (int i = 0; i < N; ++i) cout << host_arr[i] << " ";
		cout << "\nGPU: ";
		for (int i = 0; i < N; ++i) cout << host_somas_prefixos[i] << " ";
		cout << "\nCPU: ";
		for (int i = 0; i < N; ++i) cout << host_somas_prefixos_cpu[i] << " ";
		cout << "\n";
		assert(false);
	}

	cout << "SOMA DE PREFIXOS OCORREU COM SUCESSO.\n";

	return 0;
}

// ## SOMA DE PREFIXOS EM CPU ##
// Implementação sequencial (inclusiva)
void somaPrefixosCPU(const int *arr, int *somas, int tamanho) {
    int acc = 0;
    for (int i = 0; i < tamanho; ++i) {
        acc += arr[i];
        somas[i] = acc;
    }
}
