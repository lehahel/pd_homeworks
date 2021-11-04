#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */
__global__ void ScalarMulBlock(
    int numElements, float* vector1, float* vector2, float *result) {
  extern __shared__ float shared_sums[];
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int shift = blockDim.x * gridDim.x;
  shared_sums[threadIdx.x] = vector1[start] * vector2[start];
  for (int i = start; i < numElements; i += shift) {
    atomicAdd(&(result[threadIdx.x]), shared_sums[threadIdx.x]);
  }
}

