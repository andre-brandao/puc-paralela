/*
  soma_prefixos_cpu.c
  Programa standalone em C para testar soma de prefixos (scan) apenas na CPU.

  Uso:
    ./soma_prefixos_cpu [--n N | -n N] [--seed S | -s S]

  Saída:
    - "Entrada: " seguido dos N valores originais
    - "CPU: "     seguido dos N valores com soma de prefixos (inclusiva)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

/* Implementação sequencial (inclusiva) */
static void somaPrefixosCPU(const int *arr, int *somas, int tamanho) {
    int acc = 0;
    for (int i = 0; i < tamanho; ++i) {
        acc += arr[i];
        somas[i] = acc;
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Uso: %s [--n N | -n N] [--seed S | -s S]\n", prog);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return -1;
    }
    if (v < -2147483648L || v > 2147483647L) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

int main(int argc, char **argv) {
    int N = 8;           /* padrão */
    unsigned int seed = 12345u; /* padrão */

    /* Parse de argumentos */
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if ((strcmp(arg, "--n") == 0) || (strcmp(arg, "-n") == 0)) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            int tmp;
            if (parse_int(argv[++i], &tmp) != 0 || tmp <= 0) {
                fprintf(stderr, "Valor inválido para N: %s\n", argv[i]);
                return 1;
            }
            N = tmp;
        } else if ((strcmp(arg, "--seed") == 0) || (strcmp(arg, "-s") == 0)) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            int tmp;
            if (parse_int(argv[++i], &tmp) != 0) {
                fprintf(stderr, "Valor inválido para seed: %s\n", argv[i]);
                return 1;
            }
            if (tmp < 0) {
                /* normaliza seed negativa para um unsigned */
                seed = (unsigned int)tmp;
            } else {
                seed = (unsigned int)tmp;
            }
        } else if ((strcmp(arg, "--help") == 0) || (strcmp(arg, "-h") == 0)) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Argumento desconhecido: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    int *arr = (int *)malloc((size_t)N * sizeof(int));
    int *somas = (int *)malloc((size_t)N * sizeof(int));
    if (!arr || !somas) {
        fprintf(stderr, "Falha ao alocar memória.\n");
        free(arr);
        free(somas);
        return 1;
    }

    /* Inicialização determinística com seed informada */
    srand(seed);
    for (int i = 0; i < N; ++i) {
        arr[i] = rand() % 10; /* valores pequenos como no exemplo original */
    }

    somaPrefixosCPU(arr, somas, N);

    /* Saídas solicitadas */
    printf("Entrada: ");
    for (int i = 0; i < N; ++i) {
        printf("%d%s", arr[i], (i + 1 < N) ? " " : "");
    }
    printf("\n");

    printf("CPU: ");
    for (int i = 0; i < N; ++i) {
        printf("%d%s", somas[i], (i + 1 < N) ? " " : "");
    }
    printf("\n");

    free(arr);
    free(somas);
    return 0;
}