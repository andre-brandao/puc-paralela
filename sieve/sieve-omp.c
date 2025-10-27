/**
 * Crivo de Eratóstenes Paralelo com OpenMP
 * Adaptado para usar reduction e políticas de escalonamento
 * Compilação: gcc sieve.c -o sieve -fopenmp -lm
 * Execução: time ./sieve
 *
 * Resultados:
 *  ./test.sh
 *  Compiling sieve.c
 *  Testing sieve.c
 *  5761455
 *
 *  real	0m1,032s
 *  user	0m0,966s
 *  sys	0m0,050s
 *  ----------------
 *  Compiling sieve-omp.c
 *  Testing sieve-omp.c
 *  Using 4 threads
 *  5761455
 *  Execution time: 0.6036 seconds
 *
 *  real	0m0,605s
 *  user	0m2,177s
 *  sys	0m0,055s
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int sieveOfEratosthenes(int n)
{
    // Create a boolean array "prime[0..n]" and initialize
    // all entries it as true. A value in prime[i] will
    // finally be false if i is Not a prime, else true.
    int primes = 0;
    bool *prime = (bool*) malloc((n+1)*sizeof(bool));
    int sqrt_n = sqrt(n);

    memset(prime, true, (n+1)*sizeof(bool));
    prime[0] = prime[1] = false; // 0 and 1 are not primes

    for (int p = 2; p <= sqrt_n; p++)
    {
        // If prime[p] is not changed, then it is a prime
        if (prime[p] == true)
        {
          // Update all multiples of p in parallel
            #pragma omp parallel for schedule(static, 1000)
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }

    // reucao paralela
    #pragma omp parallel for reduction(+:primes) schedule(dynamic, 10000)
    for (int p = 2; p <= n; p++)
        if (prime[p])
            primes++;

    free(prime);
    return primes;
}

int main()
{
    int n = 100000000;
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);

    double start_time = omp_get_wtime();

    int result = sieveOfEratosthenes(n);

    double end_time = omp_get_wtime();

    printf("%d\n", result);
    printf("Execution time: %.4f seconds\n", end_time - start_time);

    return 0;
}
