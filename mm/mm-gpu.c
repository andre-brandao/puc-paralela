/**
Intel Iris Plus Graphics G7 @ 1.10 GHz [Integrated]
------------------------------------------------------
Version: mm (Sequential - Baseline)
Directive: None (sequential code)
------------------------------------------------------
Compiling...
✓ Compilation successful

Executing...
Run 1:
real	0m22,376s

------------------------------------------------------
Version: mm-omp (OpenMP CPU - 4 threads)
Directive: #pragma omp parallel for
------------------------------------------------------
Compiling...
Run 1:
real	0m7,941s

======================================================
  GPU VERSIONS
======================================================

------------------------------------------------------
Version: mm-gpu-distribute
Directive: target teams distribute
------------------------------------------------------
Compiling...
Run 1:
real	0m31,837s

------------------------------------------------------
Version: mm-gpu
Directive: target teams distribute parallel for
------------------------------------------------------
Compiling...
Run 1:
real	0m15,144s

------------------------------------------------------
Version: mm-gpu-distribute-parallel
Directive: target teams distribute parallel for collapse(2)
------------------------------------------------------
Compiling...
Run 1:
real	0m11,801s

------------------------------------------------------
Version: mm-gpu-distribute-parallel-simd
Directive: target teams distribute parallel for simd collapse(2)
------------------------------------------------------
Compiling...
Run 1:
real	0m12,038s
*/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mm(double* a, double* b, double* c, int width)
{
#pragma omp target teams distribute parallel for map(to: a[0:width*width], b[0:width*width]) map(from: c[0:width*width])
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0;
      for (int k = 0; k < width; k++) {
        double x = a[i * width + k];
        double y = b[k * width + j];
        sum += x * y;
      }
      c[i * width + j] = sum;
    }
  }
}

int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));

  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }

  mm(a,b,c,width);

  free(a);
  free(b);
  free(c);

  //  for(int i = 0; i < width; i++) {
  //  for(int j = 0; j < width; j++) {
  //    printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
  //  }
  // }

  return 0;
}
