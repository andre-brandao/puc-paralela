#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void mm_kernel(double* a, double* b, double* c, int width)
{
  // Calculate row and column indices for this thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Boundary check
  if (row < width && col < width) {
    double sum = 0;
    // Inner loop executes in the kernel (per output element granularity)
    for (int k = 0; k < width; k++) {
      double x = a[row * width + k];
      double y = b[k * width + col];
      sum += x * y;
    }
    c[row * width + col] = sum;
  }
}

void mm(double* a, double* b, double* c, int width)
{
  double *d_a, *d_b, *d_c;
  size_t size = width * width * sizeof(double);
  
  // Allocate device memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);
  
  // Copy data from host to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  
  // Setup 2D grid and block dimensions
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
  
  // Launch kernel
  mm_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
  
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
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

  //  for(int i = 0; i < width; i++) {
  //  for(int j = 0; j < width; j++) {
  //    printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
  //  }
  // }
  
  free(a);
  free(b);
  free(c);
  
  return 0;
}