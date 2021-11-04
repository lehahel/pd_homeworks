#include <KernelMul.cuh>

#include <iostream>
#include <algorithm>

#include <Utils.cuh>

__global__ void KernelMul(int numElements, float* x, float* y, float* result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements) {
      result[i] = x[i] * y[i];
  }
}
