/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef RNN_H
#define RNN_H

#include "Model.h"
#include "Vector.h"
#include "WordModule.h"
#include "DataProvider.h"
#include <vector>
#include <unordered_map>

class Rnn {
  private:
    Model& model_;
    std::unordered_map<wchar_t, int>& char2int_;
    std::unordered_map<int, wchar_t>& int2char_;
    std::vector<WordModule2> net_;
    WordModule2 generator_;
    Vector firstWordHidden_;
    Vector firstCharHidden_;
    Vector lastWordHidden_;
    Vector lastCharHidden_;
    Vector lastLambda_;
    Vector lastMu_;
    const double MY_MAX_VALUE = 1000.0 * 1000.0 * 1000.0 * 1000.0;
    int T_;
    int step_;
    double lr_;
    double lr0_;
  public:
    Rnn(Model&, std::unordered_map<wchar_t, int>&,
        std::unordered_map<int, wchar_t>&, int, double);
    void reset();
    void updateLearningRate(double);
    double getLr();
    void lineSearch();
    void gradientCheck();
    void printContent();
    void forward(int, int, std::string&, bool, double&, double&, int&);
    void backward();
    void computeEntropy(double&, double&);
    void train(DataProvider&, bool, double&, double&, double&, int&);
    void eval(DataProvider&, double&, double&, int&);
    void generate(DataProvider&);
};

#endif
