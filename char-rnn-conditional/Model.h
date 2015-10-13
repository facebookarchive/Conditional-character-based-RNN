/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>

class Model {
  public:
    int m_;
    int d_;

    // parameters
    Matrix R_;
    Matrix A_;
    std::unordered_map<std::wstring, Matrix> U_;

    // gradients
    Matrix gR_;
    Matrix gA_;
    std::unordered_map<std::wstring, Matrix> gU_;

    // perturbation
    Matrix dR_;
    Matrix dA_;
    std::unordered_map<std::wstring, Matrix> dU_;

    std::set<std::wstring> ngramHistory_;

    Model(int, int);
    Model(const Model&);
    ~Model();
    void copy(Model&);
    void addHistory(std::wstring);
    void resetGradients();
    void resetDeltas();
    void update(double);
    void initialize(char*);
    void pickDeltas();
    void addDeltas(double);
    double gradTDelta();
};

#endif
