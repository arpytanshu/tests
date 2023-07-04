#include <iostream>
#include <math.h>





void add_cpu(int n, float *x, float *y){
    for(int i=0; i<n; i++ ){
        y[i] = y[i] + x[i];
    }
}

// this kernel iterates over the entire array and
// performs addition of all elements.
__global__
void add_gpu1(int n, float *x, float *y){
    for(int i=0; i<n; i++ ){
        y[i] = y[i] + x[i];
    }
}


__global__
void add_gpu2a(int n, float *x, float *y){
    int index = threadIdx.x;
    int stride = blockDim.x;
    printf("tId: (%d) \
        blockIds: (%d) \
        blockDims: (%d) \
        gridDims: (%d)\n",
        threadIdx.x,
        blockIdx.x,
        blockDim.x,
        gridDim.x);
    for(int i=index; i<n; i+=stride ){
        y[i] = y[i] + x[i];
    }
}

__global__
void add_gpu2b(int n, float *x, float *y){
    int start_ix = blockIdx.x * blockDim.x;
    int end_ix = start_ix + blockDim.x + 1;
    for(int i=start_ix; i<end_ix; i++){
        y[i] = y[i] + x[i];
    }
}




int main(void){
    int N = 20;
    
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    
    for(int i=0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // add_cpu(N, x, y);
    // add_gpu1<<<1, 1>>>(N, x, y); 
    add_gpu2a<<<1, 5>>>(N, x, y);
    // add_gpu2b<<<1, 5>>>(N, x, y);
    
    cudaDeviceSynchronize();

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
nvcc block_thread_benchmark_adder.cu -o test 
__PREFETCH=off nsys profile -o noprefetch --stats=true ./test
*/