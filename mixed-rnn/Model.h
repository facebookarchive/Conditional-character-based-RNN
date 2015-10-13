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
#include <vector>
#include <set>

class Model {
  public:
    int mw;
    int mc;
    int dc;
    int dwV1;
    int dwV2;

    double alpha_;

    // parameters
    Matrix Rw_;
    Matrix Aw_;
    Matrix Uw_;
    Matrix Rc_;
    Matrix Ac_;
    Matrix Uc_;
    Matrix Ic_;
    Matrix Q_;

    // gradients
    Matrix gRw_;
    Matrix gAw_;
    Matrix gUw_;
    Matrix gRc_;
    Matrix gAc_;
    Matrix gUc_;
    Matrix gIc_;
    Matrix gQ_;

    // list of words whose embedings are updated
    std::set<int> updatedWordsList_;

    // perturbation
    Matrix dRw_;
    Matrix dAw_;
    Matrix dUw_;
    Matrix dRc_;
    Matrix dAc_;
    Matrix dUc_;
    Matrix dIc_;
    Matrix dQ_;

    Model(int, int, int, int, int, double);
    Model(const Model&);
    ~Model();
    void copy(Model&);
    void resetGradients();
    void resetDeltas();
    void update(double);
    void initialize(char*);
    void pickDeltas();
    void addDeltas(double);
    double gradTDelta();
};

#endif
