/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef WORDMODULE_H
#define WORDMODULE_H

#include "Model.h"
#include "Vector.h"
#include <vector>
#include <string>
#include <unordered_map>

const int MAX_WORD_LENGTH = 30;

class WordModule {
  private:
    // reference to shared model
    Model& model_;

    // temporary results in dimension d and m
    Vector dTemp_;
    Vector mTemp_;

  public:
    // word level variables
    int xt_;
    int xtp1_;
    std::wstring history_;
    Vector ht_;
    Vector yt_;
    Vector lambda_;

    WordModule(Model&);
    ~WordModule();
    double forward(int, int, std::wstring, Vector&);
    double computeEntropy(Vector&);
    double computeProbability(int, int, std::wstring, Vector&);
    void backward(Vector&, Vector&, Vector&);
    int generate(int, std::wstring, Vector&);
};

#endif
