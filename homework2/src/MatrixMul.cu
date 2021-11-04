#include <MatrixMul.cuh>

__global__ void MatrixMul(int heightA, int widthA, int widthB, float *matrixA,
                          float *matrixB, float *matrixResult) {
  //  extern __shared__ float shared_matrix[];

  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int column = blockIdx.x * blockDim.x + threadIdx.x;

  // float* A_item = (float*)(((char*)A + row * width) + column * sizeof(float));
  // float* B_item = (float*)(((char*)B + row * width) + column * sizeof(float));
  // float* res_item =
  //     (float*)(((char*)result + row * width) + column * sizeof(float));
}

