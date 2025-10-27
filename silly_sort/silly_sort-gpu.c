/**
Compiling silly_sort.c
Testing silly_sort.c
test passed

real	0m1,087s
user	0m1,082s
sys	0m0,000s
----------------
Compiling silly_sort-omp.c
Testing silly_sort-omp.c
test passed

real	0m0,325s
user	0m1,283s
sys	0m0,002s
----------------
Compiling silly_sort-gpu.c
Testing silly_sort-gpu.c
test passed

real	0m0,275s
user	0m1,380s
sys	0m0,008s
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
   int i, j, n = 30000;

   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));
   int *out = (int*) calloc(n, sizeof(int));

   for(i=0; i < n; i++)
      in[i] = n-i;

   // Print input array
   //   for(i=0; i < n; i++)
   //      printf("%d ",in[i]);

   #pragma omp target teams distribute parallel for map(to: in[0:n]) map(tofrom: pos[0:n]) private(j)
   for(i=0; i < n; i++)
   {
      for(j=0; j < n; j++)
      {
         if(in[i] > in[j])
            pos[i]++;
      }
   }

   // Move elements to final position (on CPU)
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
