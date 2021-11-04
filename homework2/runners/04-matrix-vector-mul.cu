#include <iostream>
#include <random>
#include <algorithm>

#include <Helpers.cuh>

namespace {

float* StupidMulMV(float* x, float* y, size_t width, size_t height) {
  float* result = new float[height]();
  for (size_t row = 0; row < height; ++row) {
    for (size_t column = 0; column < width; ++column) {
      result[row] += x[row * width + column] * y[column];
    }
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
  using helpers::Dot;
  using helpers::Timer;
  using helpers::Array;
  using helpers::Matrix;
  using helpers::GenerateRandomArray;
  using helpers::GenerateRandomMatrix;

  const size_t width = 1024;
  const size_t height = 512;
  Matrix x = GenerateRandomMatrix(width, height);
  Array y = GenerateRandomArray(width);

  Timer timer;

  timer.start();
  Array result = Dot(x, y);
  timer.stop();

  float* stupid = StupidMulMV(x.data(), y.data(), width, height);

  std::cout << "\e[1mTEST-04-MUL_MATRIX_VECTOR\e[0m";
  if (AreEqual(result.data(), stupid, height)) {
    std::cout << "\033[1;32m PASSED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  } else {
    std::cout << "\033[1;31m FAILED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  }

  delete[] stupid;
  return 0;
}
