#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(
    int height, int width, int pitch, float* A, float* B, float* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * (pitch / sizeof(float)) + column;
  result[index] = A[index] + B[index];
}
