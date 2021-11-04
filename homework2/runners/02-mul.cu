#include <iostream>
#include <random>
#include <algorithm>

#include <Helpers.cuh>

namespace {

float* StupidMul(float* x, float* y, size_t size) {
  float* result = new float[size];
  for (size_t i = 0; i < size; ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

bool AreEqual(float* l, float* r, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  using helpers::Timer;
  using helpers::Array;
  using helpers::GenerateRandomArray;

  const size_t size = 1024;
  Array x = GenerateRandomArray(size);
  Array y = GenerateRandomArray(size);

  Timer timer;

  timer.start();
  Array result = x * y;
  timer.stop();

  float* stupid = StupidMul(x.data(), y.data(), size);

  std::cout << "\e[1mTEST-02-MUL\e[0m";
  if (AreEqual(result.data(), stupid, size)) {
    std::cout << "\033[1;32m PASSED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  } else {
    std::cout << "\033[1;31m FAILED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  }

  delete[] stupid;
  return 0;
}
