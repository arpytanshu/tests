# Cuda CheatSheet

<br>  

---  

<br>  

### **Function Type Qualifiers**

Function type qualifiers are used in function declarations and definitions.
They specify where the functions can be called from and where they can be executed.
Host is the CPU side, and device is the GPU side.

\> \_\_device__ : Executed on the device. Callable from the device only.  
\> \_\_global__ : Executed on the device. Callable from the host or from the device for devices of compute capability 3.x or higher. Must have void return type.  
\> \_\_host__  : Executed on the host. Callable from the host only (equivalent to declaring the function without any qualifiers).  

<br>  

---  

<br>   

### **Built-in Vector Types**

#### > <u>Types</u>
`charX`, `ucharX`, `shortX`, `intX`, `uintX`, `longX`, `ulongX`, `floatX`, where X = 1, 2, 3, or 4.  
`doubleX`, `longlongX`, `ulonglongX`, where X = 1, or 2.  
Note: dim3 is a uint3 with default components initalized to 1.


#### > <u>Constructor Function</u>
`make_<type name>` constructs the built-in vector type with type specified by replacing `<type name>` with one of the types above.

#### > <u>Component Access</u>
The 1st, 2nd, 3rd, and 4th components are accessible through the fields `x`, `y`, `z`, and `w`.

#### > <u>Example</u>
`int4 intVec = make_int4(0, 42, 3, 5)` creates an `int4` vector typed variable named `intVec` with the given int elements. intVec.z accesses its third element, 3.


<br>  

---  

<br>  


### **Built-in Variables**
Inside functions executed on the device (GPU), grid and block dimensions, and block and thread indices can be accessed using built-in variables listed below.

`gridDim`: Dimensions of the grid (dim3).  
`blockIdx` : Block index within the grid (uint3).  
`blockDim` : Dimensions of the block (dim3).  
`threadIdx` : Thread index within the block (uint3).  
`warpSize` : Warp size in threads (int).  

<br>  

---  

<br>  

### **Device Memory Management**

#### > <u>Allocating memory</u>
`cudaError_t cudaMalloc(void **devPtr, size_t size)`  
Allocates size bytes of linear memory on the device and points devPtr to the allocated memory.

#### > <u>Freeing memory</u>
`cudaError_t cudaFree(void *devPtr)`  
Frees the memory space pointed to by devPtr.

#### > <u>Transferring data</u>
`cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`  
Copies count bytes of data from the memory area pointed to by `src` to the memory area pointed 
to by `dst`.  
The direction of copy is specified by kind, and is one of:  
- `cudaMemcpyHostToHost`
- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`

<br>  

---  

<br>  

### **Kernel Launch**
A kernel function declared as `__global__ void Func(float *parameter)`  can be called without the optional arguments as 
`Func<<<numBlocks, threadsPerBlock>>>(parameter)` or with the optional arguments as `Func<<<numBlocks, threadsPerBlock, Ns, S>>>(parameter)`.

`numBlocks` is of type `dim3` and specifies the number of b locks,

`threadsPerBlock` is of type `dim3` and specifies the number of threads per block,

`Ns` is of type `size_t` and specifies bytes in shared memory (optional),

`S` is of type `cudaStream_t` and specifies associated stream (optional).

<br>  

---  

<br>  
