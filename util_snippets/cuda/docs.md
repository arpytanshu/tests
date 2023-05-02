cudaMallocManaged() - To allocate data in unified memory.
nvprof
cudaFree(pointer)


during execution there is a finer grouping of threads into warps.
SMs on the GPU execute instructions for each warp in SIMD fashion.
The warp size (effectively the SIMD width) of all current CUDA-capable GPUs is 32 threads.

