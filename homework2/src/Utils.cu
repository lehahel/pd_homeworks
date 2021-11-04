#include "Utils.cuh"
#include <iostream>
namespace utils {

int GetBlockSize() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return roundl(sqrtl(deviceProp.maxThreadsPerBlock));
}

float* ToDevice(const float* data, size_t size) {
  float* result;
  cudaMalloc(&result, size * sizeof(float));
  cudaMemcpy(result, data, size * sizeof(float), cudaMemcpyHostToDevice);
  return result;
}

PtrWithPitch ToDevice2D(const float* data, size_t width, size_t height) {
  PtrWithPitch result;
  cudaMallocPitch(&result.ptr, &result.pitch, width * sizeof(float), height);
  cudaMemcpy2D(result.ptr, result.pitch, data, width * sizeof(float),
               width * sizeof(float), height, cudaMemcpyHostToDevice);
  return result;
}

float* FromDevice(const float* data, size_t size) {
  float* result = new float[size];
  cudaMemcpy(result, data, size * sizeof(float), cudaMemcpyDeviceToHost);
  return result;
}

float* FromDevice2D(PtrWithPitch data, size_t width, size_t height) {
  float* result = new float[width * height];
  cudaMemcpy2D(result, width * sizeof(float), data.ptr, data.pitch,
               width * sizeof(float), height, cudaMemcpyDeviceToHost);
  return result;
}

float* MallocDevice(size_t size) {
  float* result;
  cudaMalloc(&result, size * sizeof(float));
  return result;
}

PtrWithPitch MallocDevice2D(size_t width, size_t height) {
  PtrWithPitch result;
  cudaMallocPitch(&result.ptr, &result.pitch, width * sizeof(float), height);
  return result;
}

void DeviceFree(float* data) {
  cudaFree(data);
}

} // namespace utils
