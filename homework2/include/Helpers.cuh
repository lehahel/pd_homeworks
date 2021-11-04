#pragma once

#include <iostream>

#include <Utils.cuh>

namespace helpers {

class Array;
class CudaArray;
class Matrix;
class CudaMatrix;

Array GenerateRandomArray(size_t size);

Matrix GenerateRandomMatrix(size_t width, size_t height);

class Timer {
public:
 Timer();
 void start();
 void stop();
 float get_time();
private:
 cudaEvent_t start_event;
 cudaEvent_t stop_event;
};

class Array {
 public:
  Array() = delete;
  Array(size_t size);
  Array(const Array&) = delete;
  Array(Array&& array);
  Array(const CudaArray& array);

  Array& operator=(const Array&) = delete;
  Array& operator=(Array&&);

  size_t size() const;
  float* data() const;

  ~Array();

 private:
  size_t size_;
  float* data_;
};

Array operator+(const Array& left, const Array& right);
Array operator*(const Array& left, const Array& right);

float Product(const Array& left, const Array& right);
float Product2(const Array& left, const Array& right);
float Cos(const Array& left, const Array& right);

class CudaArray {
public:
  CudaArray() = delete;
  CudaArray(size_t size);
  CudaArray(const CudaArray&) = delete;
  CudaArray(CudaArray&& array);
  CudaArray(const Array& array);

  CudaArray& operator=(const CudaArray&) = delete;
  CudaArray& operator=(CudaArray&& array);

  size_t size() const;
  float* data() const;

  ~CudaArray();

 private:
  size_t size_;
  float* data_;
};

class Matrix {
public:
  Matrix() = delete;
  Matrix(size_t width, size_t height);
  Matrix(const Matrix&) = delete;
  Matrix(Matrix&& matrix);
  Matrix(const CudaMatrix& matrix);

  Matrix& operator=(const Matrix&) = delete;
  Matrix& operator=(Matrix&& matrix);

  std::pair<size_t, size_t> size() const;
  float* data() const;

  ~Matrix();

 private:
  std::pair<size_t, size_t> size_;
  float* data_;
};

Matrix operator+(const Matrix& left, const Matrix& right);
Array Dot(const Matrix& left, const Array& right);

class CudaMatrix {
public:
  CudaMatrix() = delete;
  CudaMatrix(size_t width, size_t height);
  CudaMatrix(const CudaMatrix&) = delete;
  CudaMatrix(CudaMatrix&& matrix);
  CudaMatrix(const Matrix& matrix);

  CudaMatrix& operator=(const CudaMatrix&) = delete;
  CudaMatrix& operator=(CudaMatrix&& matrix);

  std::pair<size_t, size_t> size() const;
  utils::PtrWithPitch data() const;

  ~CudaMatrix();

 private:
  std::pair<size_t, size_t> size_;
  utils::PtrWithPitch data_;
};

} // namespace helpers