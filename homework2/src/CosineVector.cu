#include <CosineVector.cuh>

#include <ScalarMulRunner.cuh>
#include <Utils.cuh>

#include <cmath>
#include <stdexcept>

class ZeroVectorError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

float CosineVector(int numElements, float* vector1, float* vector2) {
  const int blockSize = utils::GetBlockSize();
  float vector1_len =
      sqrt(ScalarMulTwoReductions(numElements, vector1, vector1, blockSize));
  float vector2_len =
      sqrt(ScalarMulTwoReductions(numElements, vector2, vector2, blockSize));
  float product =
      ScalarMulTwoReductions(numElements, vector1, vector2, blockSize);
  if (vector1_len == 0 || vector2_len == 0) {
    throw ZeroVectorError("Vectors should be non-zero");
  }
  return product / (vector1_len * vector2_len);
}

