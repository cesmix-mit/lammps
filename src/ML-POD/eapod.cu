template <typename T>
__global__ void updateTwoBodyDescriptorsKernel(int nrbf2, int Nij, const int* tj, const int* idxi, const T* rbf, 
    const T* rbfx, const T* rbfy, const T* rbfz, T* d2, T* dd2, int Ni) 
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x; // Total number of threads

  while (index < nrbf2 * Nij) {
    int m = index / Nij;
    int n = index % Nij;

    int i2 = n + Nij * m;
    int i1 = n + Nij * m + Nij * nrbf2 * (tj[n] - 1);

    atomicAdd(&d2[idxi[n] + Ni * (m + nrbf2 * (tj[n] - 1))], rbf[i2]);
    dd2[0 + 3 * i1] += rbfx[i2];
    dd2[1 + 3 * i1] += rbfy[i2];
    dd2[2 + 3 * i1] += rbfz[i2];

    index += stride; // Move to the next set of elements
  }
}

template <typename T>
void updateTwoBodyDescriptors(T* d2, T* dd2, const T* rbf, const T* rbfx, const T* rbfy, const T* rbfz, const int* tj, const int* idxi, const int nrbf2, const int Ni, const int Nij)
{
  int threadsPerBlock = 256; // A typical number, but can be tuned for your specific GPU
  int totalThreads = nrbf2 * Nij;
  int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

  updateTwoBodyDescriptorsKernel<<<numBlocks, threadsPerBlock>>>(nrbf2, Nij, tj, idxi, rbf, rbfx, rbfy, rbfz, d2, dd2, Ni);
}


// #include <stdio.h>
// #include <cuda.h>

// // Assume the kernel function is defined here or in an included header file
// __global__ void updateDescriptorsKernel(int nrbf2, int Nij, const int* tj, const int* idxi, const float* rbf, const float* rbfx, const float* rbfy, const float* rbfz, float* d2, float* dd2, int Ni);

// void launchUpdateDescriptorsKernel(int nrbf2, int Nij, int* tj, int* idxi, float* rbf, float* rbfx, float* rbfy, float* rbfz, float* d2, float* dd2, int Ni) {
//     int *d_tj, *d_idxi;
//     float *d_rbf, *d_rbfx, *d_rbfy, *d_rbfz, *d_d2, *d_dd2;

//     // Allocate memory on the device
//     cudaMalloc((void**)&d_tj, Nij * sizeof(int));
//     cudaMalloc((void**)&d_idxi, Nij * sizeof(int));
//     cudaMalloc((void**)&d_rbf, Nij * nrbf2 * sizeof(float));
//     cudaMalloc((void**)&d_rbfx, Nij * nrbf2 * sizeof(float));
//     cudaMalloc((void**)&d_rbfy, Nij * nrbf2 * sizeof(float));
//     cudaMalloc((void**)&d_rbfz, Nij * nrbf2 * sizeof(float));
//     cudaMalloc((void**)&d_d2, Ni * nrbf2 * sizeof(float)); // Adjust size as needed
//     cudaMalloc((void**)&d_dd2, 3 * Nij * nrbf2 * sizeof(float)); // Adjust size as needed

//     // Copy data from host to device
//     cudaMemcpy(d_tj, tj, Nij * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_idxi, idxi, Nij * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rbf, rbf, Nij * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rbfx, rbfx, Nij * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rbfy, rbfy, Nij * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rbfz, rbfz, Nij * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_d2, d2, Ni * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dd2, dd2, 3 * Nij * nrbf2 * sizeof(float), cudaMemcpyHostToDevice);

//     // Define the kernel launch parameters
//     int threadsPerBlock = 256; 
//     int numBlocks = (256 * 256); // Adjust as needed

//     // Launch the kernel
//     updateDescriptorsKernel<<<numBlocks, threadsPerBlock>>>(nrbf2, Nij, d_tj, d_idxi, d_rbf, d_rbfx, d_rbfy, d_rbfz, d_d2, d_dd2, Ni);

//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();

//     // Copy results back to host
//     cudaMemcpy(d2, d_d2, Ni * nrbf2 * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(dd2, d_dd2, 3 * Nij * nrbf2 * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_tj);
//     cudaFree(d_idxi);
//     cudaFree(d_rbf);
//     cudaFree(d_rbfx);
//     cudaFree(d_rbfy);
//     cudaFree(d_rbfz);
//     cudaFree(d_d2);
//     cudaFree(d_dd2);
// }

// int main() {
//     // Assume the necessary arrays are defined and initialized here

//     // Example call to the function
//     launchUpdateDescriptorsKernel(nrbf2, Nij, tj, idxi, rbf, rbfx, rbfy, rbfz, d2, dd2, Ni);

//     // The results are now in d2 and dd2 arrays

//     return 0;
// }
