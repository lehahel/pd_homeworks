#include <CosineVector.cuh>

#include <Helpers.cuh>

#include <cmath>

namespace {

float StupidProduct(float* x, float* y, size_t size) {
  float result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

float StupidCos(float* x, float* y, size_t size) {
  float x_len = sqrt(StupidProduct(x, x, size));
  float y_len = sqrt(StupidProduct(y, y, size));
  float prod = StupidProduct(x, y, size);
  return prod / (x_len * y_len);
}

} // namespace

int main() {
  using helpers::Timer;
  using helpers::Array;
  using helpers::Cos;
  using helpers::GenerateRandomArray;

  const size_t size = 5;
  Array x = GenerateRandomArray(size);
  Array y = GenerateRandomArray(size);

  Timer timer;

  timer.start();
  float result = Cos(x, y);
  timer.stop();

  float stupid = StupidCos(x.data(), y.data(), size);

  std::cout << "\e[1mTEST-05-1-PRODUCT1\e[0m";
  if (result == stupid) {
    std::cout << "\033[1;32m PASSED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  } else {
    std::cout << "\033[1;31m FAILED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  }
}

