#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(
    int height, int width, float* matrix, float* vector, float* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  float* matrix_item =
      (float*)(((char*)matrix + row * width) + column * sizeof(float));

  atomicAdd(&(result[row]), *matrix_item * vector[column]);
}

