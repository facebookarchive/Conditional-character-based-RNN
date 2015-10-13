/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include "Utils.h"
#include <stdio.h>
#include <assert.h>

class Matrix;

class Vector {
  public:
    Vector(int);
    Vector(const Vector&);
    ~Vector();
    void fillRandom();
    void fillValue(double);
    void print();

    void set(int, double);
    double get(int);
    double max();

    // level 1 operations
    void scale(double);
    void copy(Vector&);
    double dot(Vector&);
    void addInPlace(Vector&);
    void addInPlace(double, Vector&);
    void timesInPlace(Vector&);
    double norm2();
    void sigmoid();
    void aTimesOneMinusA();
    void softMax();

    void getColumn(Matrix&, int);
    void getRow(Matrix&, int);

    void addVectors(Vector&, Vector&);
    void timesVectors(Vector&, Vector&);

    void matrixVector(double, Matrix&, Vector&, double);
    void matrixTVector(double, Matrix&, Vector&, double);

    int m_;
    double* data_;
};

#endif
