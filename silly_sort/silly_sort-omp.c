/**
➜ ./test_silly.sh
Compiling silly_sort.c
Testing silly_sort.c
test passed

real	0m1,136s
user	0m1,130s
sys	0m0,001s
----------------
Compiling silly_sort-omp.c
Testing silly_sort-omp.c
test passed

real	0m0,347s
user	0m1,373s
sys	0m0,001s*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
   int i, j, n = 30000;

   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));
   int *out = (int*) calloc(n, sizeof(int));

   // Initialize input array in the reverse order
   for(i=0; i < n; i++)
      in[i] = n-i;

   // Print input array
   //   for(i=0; i < n; i++)
   //      printf("%d ",in[i]);
   omp_set_num_threads(omp_get_max_threads());

   // #pragma omp parallel for schedule(auto) private(j)
   // #pragma omp parallel for schedule(guided) private(j)
   // #pragma omp parallel for schedule(static) private(j)
   #pragma omp parallel for schedule(dynamic) private(j)
   for(i=0; i < n; i++)
   {
      for(j=0; j < n; j++)
      {
         if(in[i] > in[j])
            pos[i]++;
      }
   }

   // Move elements to final position
   for(i=0; i < n; i++)
      out[pos[i]] = in[i];

   // print output array
   //   for(i=0; i < n; i++)
   //      printf("%d ",out[i]);

   // Check if answer is correct
   for(i=0; i < n; i++)
   {
      if(i+1 != out[i])
      {
         printf("test failed\n");
         exit(0);
      }
   }
   printf("test passed\n");

   // Clean up memory
   free(in);
   free(pos);
   free(out);

   return 0;
}
