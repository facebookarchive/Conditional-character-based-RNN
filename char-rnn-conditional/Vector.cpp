/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Vector.h"
#include "Matrix.h"
#include <math.h>
#include <cblas.h>
#include <float.h>

extern bool USE_BLAS;

Vector::Vector(int m) {
  m_ = m;
  data_ = new double[m];
}

Vector::Vector(const Vector& other) {
  m_ = other.m_;
  data_ = new double[m_];
  for (int i=0; i<m_; i++) {
    data_[i] = other.data_[i];
  }
}

Vector::~Vector() {
  delete[] data_;
}

void Vector::fillRandom() {
  for (int i=0; i<m_; i++) {
    data_[i] = uniRand();
  }
}

void Vector::fillValue(double a) {
  for (int i=0; i<m_; i++) {
    data_[i] = a;
  }
}

void Vector::print() {
  for (int i=0; i<m_; i++) {
    printf("%+5.2f ", data_[i]);
  }
  printf("\n");
}

void Vector::set(int i, double d) {
  assert(i < m_);
  data_[i] = d;
}

double Vector::get(int i) {
  assert(i < m_);
  return data_[i];
}

double Vector::max() {
  double M = -DBL_MAX;
  for (int i=0; i<m_; i++) {
    if (data_[i] > M) {
      M = data_[i];
    }
  }
  return M;
}

void Vector::scale(double a) {
  for (int i=0; i<m_; i++) {
    data_[i] *= a;
  }
}

void Vector::copy(Vector& b) {
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] = b.data_[i];
  }
}

// dot product between this and b
double Vector::dot(Vector& b) {
  assert(m_ == b.m_);
  double result = 0.0;
  for (int i=0; i<m_; i++) {
    result += data_[i] * b.data_[i];
  }
  return result;
}

// in place update as this = this + b
void Vector::addInPlace(Vector& b) {
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] += b.data_[i];
  }
}

// in place update as this = this * b
void Vector::timesInPlace(Vector& b) {
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] *= b.data_[i];
  }
}

// in place update as this = this + a * b
void Vector::addInPlace(double a, Vector& b) {
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] += a * b.data_[i];
  }
}

// dot product between this and b
double Vector::norm2() {
  double result = 0.0;
  for (int i=0; i<m_; i++) {
    result += data_[i] * data_[i];
  }
  return result;
}

// apply sigmoid to vector
void Vector::sigmoid() {
  for (int i=0; i<m_; i++) {
    data_[i] = 1.0 / (1 + exp(-data_[i]));
  }
}

void Vector::aTimesOneMinusA() {
  for (int i=0; i<m_; i++) {
    data_[i] = data_[i] * (1.0 - data_[i]);
  }
}

void Vector::softMax() {
  double sum = 0.0;
  double M = max();
  for (int i=0; i<m_; i++) {
    data_[i] = exp(data_[i] - M);
    sum += data_[i];
  }
  for (int i=0; i<m_; i++) {
    data_[i] /= sum;
  }
}

// vector becomes the j-th column of matrix A
void Vector::getColumn(Matrix& A, int j) {
  assert(m_ == A.m_);
  assert(j < A.n_);
  for (int i=0; i<m_; i++) {
    data_[i] = A.data_[i * A.n_ + j];
  }
}

void Vector::getRow(Matrix& A, int i) {
  assert(m_ == A.n_);
  assert(i>=0 && i<A.m_);
  for (int j=0; j<A.n_; j++) {
    data_[j] = A.data_[i * A.n_ + j];
  }
}

// store in object the output of a + b
void Vector::addVectors(Vector& a, Vector& b) {
  assert(m_ == a.m_);
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] = a.data_[i] + b.data_[i];
  }
}

// store in object the pointwise output of a * b
void Vector::timesVectors(Vector& a, Vector& b) {
  assert(m_ == a.m_);
  assert(m_ == b.m_);
  for (int i=0; i<m_; i++) {
    data_[i] = a.data_[i] * b.data_[i];
  }
}

// computes the GEMV this = a * matB * vecC + d * this
void Vector::matrixVector(double a, Matrix& matB, Vector& vecC, double d) {
  assert(m_ == matB.m_);
  assert(matB.n_ == vecC.m_);
  if (USE_BLAS) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, matB.m_, matB.n_, a, matB.data_,
        matB.n_, vecC.data_, 1, d, data_, 1);
  } else {
    int i = 0;
    int m = m_;
    int n = vecC.m_;

    for (i=0; i < m/8; i++) {
      data_[i * 8 + 0] *= d;
      data_[i * 8 + 1] *= d;
      data_[i * 8 + 2] *= d;
      data_[i * 8 + 3] *= d;

      data_[i * 8 + 4] *= d;
      data_[i * 8 + 5] *= d;
      data_[i * 8 + 6] *= d;
      data_[i * 8 + 7] *= d;
    }
    for (int ip = i*8; ip<m; ip++) {
      data_[ip] *= d;
    }

    double val0 = 0.0;
    double val1 = 0.0;
    double val2 = 0.0;
    double val3 = 0.0;

    double val4 = 0.0;
    double val5 = 0.0;
    double val6 = 0.0;
    double val7 = 0.0;

    for (i=0; i< m / 8; i++) {
      val0 = 0.0;
      val1 = 0.0;
      val2 = 0.0;
      val3 = 0.0;

      val4 = 0.0;
      val5 = 0.0;
      val6 = 0.0;
      val7 = 0.0;
      for (int j=0; j<n; j++) {
        val0 += matB.data_[(i * 8 + 0) * n + j] * vecC.data_[j];
        val1 += matB.data_[(i * 8 + 1) * n + j] * vecC.data_[j];
        val2 += matB.data_[(i * 8 + 2) * n + j] * vecC.data_[j];
        val3 += matB.data_[(i * 8 + 3) * n + j] * vecC.data_[j];

        val4 += matB.data_[(i * 8 + 4) * n + j] * vecC.data_[j];
        val5 += matB.data_[(i * 8 + 5) * n + j] * vecC.data_[j];
        val6 += matB.data_[(i * 8 + 6) * n + j] * vecC.data_[j];
        val7 += matB.data_[(i * 8 + 7) * n + j] * vecC.data_[j];
      }
      data_[i * 8 + 0] += val0;
      data_[i * 8 + 1] += val1;
      data_[i * 8 + 2] += val2;
      data_[i * 8 + 3] += val3;

      data_[i * 8 + 4] += val4;
      data_[i * 8 + 5] += val5;
      data_[i * 8 + 6] += val6;
      data_[i * 8 + 7] += val7;
    }
    for (int ip=i*8; ip < m; ip++) {
      for (int j = 0; j<n; j++) {
        data_[ip] += matB.data_[ip * n + j] * vecC.data_[j];
      }
    }
  }
}

// computes the GEMV this = a * matB^T * vecC + d * this
void Vector::matrixTVector(double a, Matrix& matB, Vector& vecC, double d) {
  assert(m_ == matB.n_);
  assert(matB.m_ == vecC.m_);
  if (USE_BLAS) {
    cblas_dgemv(CblasRowMajor, CblasTrans, matB.m_, matB.n_, a, matB.data_,
        matB.n_, vecC.data_, 1, d, data_, 1);
  } else {
    double mult = 0.0;
    int m = matB.m_;
    int n = matB.n_;
    for (int j=0; j<n; j++) {
      data_[j] *= d;
    }
    for (int i=0; i<m; i++) {
      mult = vecC.data_[i];
      for (int j=0; j<n; j++) {
        data_[j] += mult * matB.data_[i * n + j];
      }
    }
  }
}

// // computes the GEMV this = a * matB^T * vecC + d * this
// void Vector::matrixTVector(double a, Matrix& matB, Vector& vecC, double d) {
//   assert(m_ == matB.n_);
//   assert(matB.m_ == vecC.m_);
//   if (USE_BLAS) {
//     cblas_dgemv(CblasRowMajor, CblasTrans, matB.m_, matB.n_, a, matB.data_,
//         matB.n_, vecC.data_, 1, d, data_, 1);
//   } else {
//     int m = matB.m_;
//     int n = matB.n_;
//     for (int j=0; j<n; j++) {
//       data_[j] *= d;
//     }
//
//     double val0 = 0;
//     double val1 = 0;
//     double val2 = 0;
//     double val3 = 0;
//
//     double val4 = 0;
//     double val5 = 0;
//     double val6 = 0;
//     double val7 = 0;
//     int j = 0;
//     for (j=0; j<n/8; j++) {
//       val0 = 0.0;
//       val1 = 0.0;
//       val2 = 0.0;
//       val3 = 0.0;
//
//       val4 = 0.0;
//       val5 = 0.0;
//       val6 = 0.0;
//       val7 = 0.0;
//
//       for (int i=0; i<m; i++) {
//         val0 += vecC.data_[i] * matB.data_[i*n + j*8 + 0];
//         val1 += vecC.data_[i] * matB.data_[i*n + j*8 + 1];
//         val2 += vecC.data_[i] * matB.data_[i*n + j*8 + 2];
//         val3 += vecC.data_[i] * matB.data_[i*n + j*8 + 3];
//
//         val4 += vecC.data_[i] * matB.data_[i*n + j*8 + 4];
//         val5 += vecC.data_[i] * matB.data_[i*n + j*8 + 5];
//         val6 += vecC.data_[i] * matB.data_[i*n + j*8 + 6];
//         val7 += vecC.data_[i] * matB.data_[i*n + j*8 + 7];
//       }
//       data_[j*8 + 0] += val0;
//       data_[j*8 + 1] += val1;
//       data_[j*8 + 2] += val2;
//       data_[j*8 + 3] += val3;
//
//       data_[j*8 + 4] += val4;
//       data_[j*8 + 5] += val5;
//       data_[j*8 + 6] += val6;
//       data_[j*8 + 7] += val7;
//     }
//     for (int jp=j*8; jp<n; jp++) {
//       for (int i=0; i<m; i++) {
//         data_[jp] += vecC.data_[i] * matB.data_[i*n + jp];
//       }
//     }
//   }
// }
