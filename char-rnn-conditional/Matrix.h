/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "Utils.h"
#include <stdio.h>
#include <assert.h>
#include <fstream>

class Vector;

class Matrix {
  public:
    Matrix();
    Matrix(int, int);
    Matrix(const Matrix&);
    ~Matrix();
    void readMatrix(std::ifstream&);
    void fillRandom();
    void fillRandom(double);
    void fillRandn();
    void fillRandn(double);
    void addDiag(double);
    void setDiag(double);
    void fillValue(double);
    void print();

    void scale(double);
    void copy(Matrix&);
    double frobenius();
    double dotProduct(Matrix&);
    void addInPlace(Matrix&);
    void addInPlace(double, Matrix&);

    void addColumn(int, double, Vector&);
    void addRow(int, double, Vector&);
    void addRow(int, double, Matrix&);
    void fillRow(int, double);

    void addMatrices(Matrix&, Matrix&);
    void vectorVectorT(double, Vector&, Vector&);
    double* data_;
    int m_;
    int n_;
};

#endif
