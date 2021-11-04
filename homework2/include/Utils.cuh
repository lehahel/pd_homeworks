#pragma once

namespace utils {

struct PtrWithPitch {
  void* ptr;
  size_t pitch;
};

int GetBlockSize();

float* ToDevice(const float* data, size_t size);
PtrWithPitch ToDevice2D(const float* data, size_t width, size_t height);

float* FromDevice(const float* data, size_t size);
float* FromDevice2D(PtrWithPitch data, size_t width, size_t height);

float* MallocDevice(size_t size);
PtrWithPitch MallocDevice2D(size_t width, size_t height);

void DeviceFree(float* data);

} // namespace utils