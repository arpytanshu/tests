#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays

__global__
void add(int n, float *x, float *y)
{
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // stride is the total number of threads in the grid.
  int stride = blockDim.x * gridDim.x;
  
  for (int i = index; i < n; i += stride)
    if (i < n){
      y[i] = x[i] + y[i];
    printf("id: %d \t \
    tId: (%d, %d, %d) \
    blockIds: (%d, %d, %d) \
    blockDims: (%d, %d, %d) \
    gridDims: (%d, %d, %d)\n", index, 
    threadIdx.x, threadIdx.y, threadIdx.z, 
    blockIdx.x, blockIdx.y, blockIdx.z, 
    blockDim.x, blockDim.y, blockDim.z, 
    gridDim.x, gridDim.y, gridDim.z);
  }
}


int main(int argc, char *argv[])
{

  int N = 18;
  int blockSize = 5;

  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i;
  }
  
  int numBlocks = (N + blockSize - 1) / blockSize; // 2e12
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}


/*
Usage: nvcc add.cu -o add && ./add
*/