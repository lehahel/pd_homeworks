#include <CommonKernels.cuh>

__global__ void Reduce(int numElements, float* array, float* result) {
  extern __shared__ float shared_sums[];
  const size_t index = threadIdx.x;

  shared_sums[index] = array[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();

  for (int i = 1; i < blockDim.x; i *= 2) {
    if (index % (i * 2) == 0) {
      shared_sums[index] += shared_sums[index + i];
    }
    __syncthreads();
  }
  if (index == 0) {
    result[blockIdx.x] = shared_sums[index];
  }
}