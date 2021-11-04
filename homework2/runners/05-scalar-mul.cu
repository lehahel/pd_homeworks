#include <ScalarMulRunner.cuh>

#include <Helpers.cuh>

namespace {

float StupidProduct(float* x, float* y, size_t size) {
  float result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

} // namespace

int main() {
  using helpers::Timer;
  using helpers::Array;
  using helpers::Product;
  using helpers::Product2;
  using helpers::GenerateRandomArray;

  const size_t size = 5;
  Array x = GenerateRandomArray(size);
  Array y = GenerateRandomArray(size);

  Timer timer;

  timer.start();
  float result1 = Product(x, y);
  timer.stop();

  float stupid = StupidProduct(x.data(), y.data(), size);

  std::cout << "\e[1mTEST-05-1-PRODUCT1\e[0m";
  if (result1 == stupid) {
    std::cout << "\033[1;32m PASSED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  } else {
    std::cout << "\033[1;31m FAILED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  }

  timer.start();
  float result2 = Product2(x, y);
  timer.stop();


  std::cout << "\e[1mTEST-05-2-PRODUCT2\e[0m";
  if (result2 == stupid) {
    std::cout << "\033[1;32m PASSED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  } else {
    std::cout << "\033[1;31m FAILED\033[0m ";
    std::cout << "(time=" << timer.get_time() << "ms)" << std::endl;
  }

  return 0;
}

