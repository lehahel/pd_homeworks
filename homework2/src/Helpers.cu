#include "Helpers.cuh"

#include <iostream>
#include <random>
#include <algorithm>

#include <Utils.cuh>
#include <KernelAdd.cuh>
#include <KernelMul.cuh>
#include <KernelMatrixAdd.cuh>
#include <MatrixVectorMul.cuh>
#include <ScalarMulRunner.cuh>
#include <CosineVector.cuh>

namespace helpers {

using utils::GetBlockSize;

// Exception ///////////////////////////////////////////////////////////////////

class SizesNotEqualError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Helper functions ////////////////////////////////////////////////////////////

Array GenerateRandomArray(size_t size) {
  Array result(size);
  for (size_t i = 0; i < size; ++i) {
    result.data()[i] = rand() % 100;
  }
  return result;
}

Matrix GenerateRandomMatrix(size_t width, size_t height) {
  Matrix result(width, height);
  for (size_t i = 0; i < width * height; ++i) {
    result.data()[i] = rand() % 100;
  }
  return result;
}

// Timer ///////////////////////////////////////////////////////////////////////

Timer::Timer() {
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
}

void Timer::start() {
  cudaEventRecord(start_event);
}

void Timer::stop() {
  cudaEventRecord(stop_event);
}

float Timer::get_time() {
  cudaEventSynchronize(stop_event);
  float result = 0;
  cudaEventElapsedTime(&result, start_event, stop_event);
  return result;
}

// Array ///////////////////////////////////////////////////////////////////////

Array::Array(size_t size) : size_(size) {
  data_ = new float[size]();
}

Array::Array(Array&& array) : data_(array.data_), size_(array.size_) {
  array.data_ = nullptr;
}

Array& Array::operator=(Array&& array) {
  size_ = array.size_;
  data_ = array.data_;
  array.data_ = nullptr;
  return *this;
}

Array::Array(const CudaArray& array)
  : size_(array.size()),
    data_(utils::FromDevice(array.data(), array.size())) {};

Array::~Array() { if (data_) delete[] data_; }

size_t Array::size() const { return size_; }

float* Array::data() const { return data_; }

Array operator+(const Array& left, const Array& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Array sizes not equal");
  }
  const size_t size = left.size();
  int block_size = GetBlockSize();
  int grid_size = std::max(1ul, size / block_size);
  CudaArray result_device(size);
  KernelAdd<<<grid_size, block_size>>>(size, CudaArray(left).data(),
                                       CudaArray(right).data(),
                                       result_device.data());
  return Array(result_device);
}

Array operator*(const Array& left, const Array& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Array sizes not equal");
  }
  const size_t size = left.size();
  int block_size = GetBlockSize();
  int grid_size = std::max(1ul, size / block_size);
  CudaArray result_device(size);
  KernelMul<<<grid_size, block_size>>>(size, CudaArray(left).data(),
                                       CudaArray(right).data(),
                                       result_device.data());
  return Array(result_device);
}

float Product(const Array& left, const Array& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Array sizes not equal");
  }
  return ScalarMulTwoReductions(
      left.size(), left.data(), right.data(), GetBlockSize());
}

float Product2(const Array& left, const Array& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Array sizes not equal");
  }
  return ScalarMulSumPlusReduction(
    left.size(), left.data(), right.data(), GetBlockSize());
}

float Cos(const Array& left, const Array& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Array sizes not equal");
  }
  return CosineVector(left.size(), left.data(), right.data());
}

// CudaArray ///////////////////////////////////////////////////////////////////

CudaArray::CudaArray(size_t size) : size_(size) {
  data_ = utils::MallocDevice(size);
}

CudaArray::CudaArray(CudaArray&& array) : data_(array.data_),
                                          size_(array.size_) {
  array.data_ = nullptr;
}

CudaArray& CudaArray::operator=(CudaArray&& array) {
  data_ = array.data_;
  size_ = array.size_;
  array.data_ = nullptr;
  return *this;
}

CudaArray::CudaArray(const Array& array)
  : size_(array.size()),
    data_(utils::ToDevice(array.data(), array.size())) {};

CudaArray::~CudaArray() { if (data_) utils::DeviceFree(data_); }

size_t CudaArray::size() const { return size_; }

float* CudaArray::data() const { return data_; }

// Matrix //////////////////////////////////////////////////////////////////////

Matrix::Matrix(size_t width, size_t height)
    : size_(std::make_pair(width, height)) {
  data_ = new float[width * height]();
}

Matrix::Matrix(Matrix&& matrix) : size_(matrix.size_) {
  data_ = matrix.data_;
  matrix.data_ = nullptr;
}

Matrix::Matrix(const CudaMatrix& matrix)
  : size_(matrix.size()),
    data_(utils::FromDevice2D(matrix.data(), size_.first, size_.second)) {};

Matrix& Matrix::operator=(Matrix&& matrix) {
  data_ = matrix.data_;
  size_ = matrix.size_;
  matrix.data_ = nullptr;
  return *this;
}

std::pair<size_t, size_t> Matrix::size() const { return size_; }

float* Matrix::data() const { return data_; }

Matrix::~Matrix() { if (data_) delete[] data_; }

Matrix operator+(const Matrix& left, const Matrix& right) {
  if (left.size() != right.size()) {
    throw SizesNotEqualError("Matrix sizes not equal");
  }
  size_t width = left.size().first;
  size_t height = right.size().second;
  CudaMatrix result_device(width, height);

  int block_size = helpers::GetBlockSize();
  dim3 dim_block(block_size, block_size);
  dim3 dim_grid(std::max(1ul, width / dim_block.x),
                std::max(1ul, height / dim_block.y));

  CudaMatrix x_device(left);
  CudaMatrix y_device(right);

  KernelMatrixAdd<<<dim_grid, dim_block>>>(
      height, width, result_device.data().pitch, (float*)x_device.data().ptr,
      (float*)y_device.data().ptr, (float*)result_device.data().ptr);
  return Matrix(result_device);
}

Array Dot(const Matrix& left, const Array& right) {
  if (left.size().first != right.size()) {
    throw SizesNotEqualError("Matrix width is not equal to vector height");
  }
  size_t width = left.size().first;
  size_t height = left.size().second;

  int block_size = helpers::GetBlockSize();
  dim3 dim_block(block_size, block_size);
  dim3 dim_grid(std::max(1ul, width / dim_block.x),
                std::max(1ul, height / dim_block.y));

  CudaMatrix x_device(left);
  CudaArray y_device(right);
  CudaArray result_device(height);

  MatrixVectorMul<<<dim_grid, dim_block>>>(
     height, x_device.data().pitch, (float*)x_device.data().ptr,
     y_device.data(), result_device.data());

  return Array(result_device);
}

// CudaMatrix //////////////////////////////////////////////////////////////////

CudaMatrix::CudaMatrix(size_t width, size_t height)
  : size_(std::make_pair(width, height)),
    data_(utils::MallocDevice2D(width, height)) {};

CudaMatrix::CudaMatrix(CudaMatrix&& matrix) : size_(matrix.size_),
                                              data_(matrix.data_) {
  matrix.data_.ptr = nullptr;
}

CudaMatrix::CudaMatrix(const Matrix& matrix)
  : size_(matrix.size()),
    data_(utils::ToDevice2D(matrix.data(), size_.first, size_.second)) {};

CudaMatrix& CudaMatrix::operator=(CudaMatrix&& matrix) {
  size_ = matrix.size_;
  data_ = matrix.data_;
  matrix.data_.ptr = nullptr;
  return *this;
}

std::pair<size_t, size_t> CudaMatrix::size() const { return size_; }

utils::PtrWithPitch CudaMatrix::data() const { return data_; };

CudaMatrix::~CudaMatrix() { if (data_.ptr) cudaFree(data_.ptr); }

} // namespace helpers