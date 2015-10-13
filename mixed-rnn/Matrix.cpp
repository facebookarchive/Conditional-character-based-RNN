/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Matrix.h"
#include "Vector.h"
#include <cblas.h>
#include <algorithm>

using namespace std;

extern bool USE_BLAS;

Matrix::Matrix(int m, int n) {
  m_ = m;
  n_ = n;
  data_ = new double[m*n];
}

Matrix::Matrix(const Matrix& a) {
  m_ = a.m_;
  n_ = a.n_;
  data_ = new double[m_ * n_];
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a.data_[i * n_ + j];
    }
  }
}

Matrix::~Matrix() {
  delete[] data_;
}

void Matrix::fillRandom() {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = uniRand();
    }
  }
}

void Matrix::fillRandn() {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = 1 * randn();
    }
  }
}

void Matrix::fillRandom(double a) {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a * uniRand();
    }
  }
}

void Matrix::fillRandn(double a) {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a * randn();
    }
  }
}

void Matrix::addDiag(double a) {
  for (int i=0; i<m_; i++) {
    data_[i * n_ + i] += a;
  }
}

void Matrix::setDiag(double a) {
  for (int i=0; i<m_; i++) {
    data_[i * n_ + i] = a;
  }
}



void Matrix::fillValue(double a) {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a;
    }
  }
}

void Matrix::print() {
  int m = std::min(m_, 10);
  int n = std::min(n_, 10);
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%+8.5f ", data_[i * n_ + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void Matrix::scale(double a) {
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] *= a;
    }
  }
}

void Matrix::copy(Matrix& a) {
  assert(m_ == a.m_);
  assert(n_ == a.n_);
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a.data_[i * n_ + j];
    }
  }
}

double Matrix::frobenius() {
  double result = 0.0;
  double d = 0.0;
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      d = data_[i * n_ + j];
      result += d*d;
    }
  }
  return result;
}

double Matrix::dotProduct(Matrix& b) {
  assert(n_ == b.n_);
  assert(m_ == b.m_);
  double result = 0.0;
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      result += data_[i * n_ + j] * b.data_[i * n_ + j];
    }
  }
  return result;
}

void Matrix::addInPlace(Matrix& b) {
  assert(m_ == b.m_);
  assert(n_ == b.n_);
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] += b.data_[i * n_ + j];
    }
  }
}

void Matrix::addInPlace(double a, Matrix& b) {
  assert(m_ == b.m_);
  assert(n_ == b.n_);
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] += a * b.data_[i * n_ + j];
    }
  }
}

// adds the values in vecA times double a to row i in this matrix
void Matrix::addColumn(int j, double a, Vector& vecA) {
  assert(j>=0 && j<n_);
  assert(vecA.m_ == m_);
  for (int i=0; i<m_; i++) {
    data_[i * n_ + j] += a * vecA.data_[i];
  }
}

// adds the values in vecA times double a to row i in this matrix
void Matrix::addRow(int i, double a, Vector& vecA) {
  assert(i>=0 && i<m_);
  assert(vecA.m_ == n_);
  for (int j=0; j<n_; j++) {
    data_[i * n_ + j] += a * vecA.data_[j];
  }
}

// adds the values in vecA times double a to row i in this matrix
void Matrix::addRow(int i, double a, Matrix& matA) {
  assert(i>=0 && i<m_);
  assert(matA.m_ == m_);
  assert(matA.n_ == n_);
  for (int j=0; j<n_; j++) {
    data_[i * n_ + j] += a * matA.data_[i * n_ + j];
  }
}

// filling ith row in the matrix with value a
void Matrix::fillRow(int i, double a) {
  assert(i>=0 && i<m_);
  for (int j=0; j<n_; j++) {
    data_[i * n_ + j] = a;
  }
}

// stores in object the output of a + b
void Matrix::addMatrices(Matrix& a, Matrix& b) {
  assert(m_ == a.m_);
  assert(n_ == a.n_);
  assert(m_ == b.m_);
  assert(n_ == b.n_);
  for (int i=0; i<m_; i++) {
    for (int j=0; j<n_; j++) {
      data_[i * n_ + j] = a.data_[i * n_ + j] + b.data_[i * n_ + j];
    }
  }
}

void Matrix::vectorVectorT(double a, Vector& vecB, Vector& vecC) {
  assert(m_ == vecB.m_);
  assert(n_ == vecC.m_);
  if (USE_BLAS) {
    cblas_dger(CblasRowMajor, vecB.m_, vecC.m_, a, vecB.data_, 1,
        vecC.data_, 1, data_, n_);
  } else {
    for (int i=0; i<m_; i++) {
      for (int j=0; j<n_; j++) {
        data_[i * n_ + j] += a * vecB.data_[i] * vecC.data_[j];
      }
    }
  }
}

