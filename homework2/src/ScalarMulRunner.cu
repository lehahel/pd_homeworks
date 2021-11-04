#include <ScalarMulRunner.cuh>

#include <CommonKernels.cuh>
#include <KernelMul.cuh>
#include <ScalarMul.cuh>

#include <Utils.cuh>

float ScalarMulTwoReductions(
    int numElements, float* vector1, float* vector2, int blockSize) {
  const int gridSize = (numElements + blockSize - 1) / blockSize;
  const int blockSizeBytes = blockSize * sizeof(float);

  float* multiplied = utils::MallocDevice(numElements);
  float* vector1_device = utils::ToDevice(vector1, numElements);
  float* vector2_device = utils::ToDevice(vector2, numElements);
  KernelMul<<<gridSize, blockSize>>>(
      numElements, vector1_device, vector2_device, multiplied);

  float* reduced = utils::MallocDevice(gridSize);
  Reduce<<<gridSize, blockSize, blockSizeBytes>>>(numElements, multiplied,
    reduced);

  float* result_device = utils::MallocDevice(1);
  Reduce<<<1, blockSize, blockSizeBytes>>>(numElements, reduced, result_device);

  float* result_ptr = utils::FromDevice(result_device, 1);
  float result = *result_ptr;

  utils::DeviceFree(vector1_device);
  utils::DeviceFree(vector2_device);
  utils::DeviceFree(multiplied);
  utils::DeviceFree(reduced);
  utils::DeviceFree(result_device);
  delete[] result_ptr;

  return result;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
  const int gridSize = (numElements + blockSize - 1) / blockSize;
  const int blockSizeBytes = blockSize * sizeof(float);

  float* vector1_device = utils::ToDevice(vector1, numElements);
  float* vector2_device = utils::ToDevice(vector2, numElements);

  float* multiplied = utils::MallocDevice(blockSize);
  ScalarMulBlock<<<gridSize, blockSize, blockSizeBytes>>>(
      numElements, vector1_device, vector2_device, multiplied);

  float* result_device = utils::MallocDevice(1);
  Reduce<<<1, blockSize, blockSizeBytes>>>(numElements, multiplied,
                                           result_device);

  float* result_ptr = utils::FromDevice(result_device, 1);
  float result = *result_ptr;

  utils::DeviceFree(multiplied);
  utils::DeviceFree(result_device);
  delete[] result_ptr;

  return result;
}

