/*
SOMA DE PREFIXOS - VERSÃO CPU SEQUENCIAL EM C
==============================================

Este programa implementa a soma de prefixos (prefix sum/scan) de forma sequencial em CPU.

TEMPO DE EXECUÇÃO:
------------------
N=8:         ~0.001 ms
N=1000000:   ~2-3 ms

EXEMPLO:
--------
Array original:     [3, 1, 4, 1, 5, 9, 2, 6]
Soma de prefixos:   [3, 4, 8, 9, 14, 23, 25, 31]

Compilação: gcc soma_prefixos_cpu.c -O3 -o soma_prefixos_cpu -lm
Execução: ./soma_prefixos_cpu
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Função para medir tempo em milissegundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Implementação sequencial da soma de prefixos
void soma_prefixos_cpu(int *arr, int *somas, int tamanho) {
    if (tamanho == 0) return;
    
    // O primeiro elemento é igual ao primeiro elemento do array original
    somas[0] = arr[0];
    
    // Para cada posição i, soma[i] = soma[i-1] + arr[i]
    for (int i = 1; i < tamanho; i++) {
        somas[i] = somas[i-1] + arr[i];
    }
}

// Função para verificar a corretude da soma de prefixos
int verificar_soma_prefixos(int *arr, int *somas, int tamanho) {
    for (int i = 0; i < tamanho; i++) {
        int soma_esperada = 0;
        for (int j = 0; j <= i; j++) {
            soma_esperada += arr[j];
        }
        if (somas[i] != soma_esperada) {
            printf("ERRO na posição %d: esperado %d, obtido %d\n", 
                   i, soma_esperada, somas[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    // Tamanho do array (pode ser modificado para testes)
    int N = 1 << 20; // 1 milhão de elementos
    
    printf("====================================\n");
    printf("SOMA DE PREFIXOS - VERSÃO CPU (C)\n");
    printf("====================================\n\n");
    
    printf("Tamanho do array: %d elementos\n", N);
    printf("Memória utilizada: %.2f MB\n\n", (N * sizeof(int) * 2) / (1024.0 * 1024.0));
    
    // Aloca memória para os arrays
    int *arr = (int*)malloc(N * sizeof(int));
    int *somas = (int*)malloc(N * sizeof(int));
    
    if (arr == NULL || somas == NULL) {
        printf("ERRO: Falha ao alocar memória!\n");
        return 1;
    }
    
    // Inicializa o array com valores aleatórios entre 0 e 9
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 10;
    }
    
    // Imprime os primeiros elementos do array original
    printf("Primeiros 10 elementos do array: ");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");
    
    // Executa a soma de prefixos e mede o tempo
    printf("Executando soma de prefixos...\n");
    double inicio = get_time();
    soma_prefixos_cpu(arr, somas, N);
    double fim = get_time();
    double tempo_ms = fim - inicio;
    
    printf("Tempo de execução: %.3f ms\n\n", tempo_ms);
    
    // Imprime os primeiros resultados
    printf("Primeiros 10 resultados da soma: ");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("%d ", somas[i]);
    }
    printf("\n\n");
    
    // Verifica a corretude (apenas para arrays pequenos, pois é O(n²))
    if (N <= 10000) {
        printf("Verificando corretude...\n");
        if (verificar_soma_prefixos(arr, somas, N)) {
            printf("✓ Verificação bem-sucedida!\n");
        } else {
            printf("✗ Erro na verificação!\n");
            free(arr);
            free(somas);
            return 1;
        }
    } else {
        printf("Array grande - pulando verificação completa.\n");
        printf("Verificando apenas os primeiros 100 elementos...\n");
        if (verificar_soma_prefixos(arr, somas, 100)) {
            printf("✓ Verificação parcial bem-sucedida!\n");
        } else {
            printf("✗ Erro na verificação!\n");
            free(arr);
            free(somas);
            return 1;
        }
    }
    
    printf("\n");
    printf("Último elemento (soma total): %d\n", somas[N-1]);
    
    printf("\n====================================\n");
    printf("✓ SOMA DE PREFIXOS CONCLUÍDA!\n");
    printf("====================================\n");
    
    // Libera memória
    free(arr);
    free(somas);
    
    return 0;
}