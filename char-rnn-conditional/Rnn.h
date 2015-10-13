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
    std::vector<WordModule> net_;
    WordModule generator_;
    Vector firstHidden_;
    Vector lastHidden_;
    Vector lastLambda_;
    int T_;
    int step_;
    double lr_;
    double lr0_;
  public:
    Rnn(Model&, int, double);
    void reset();
    void lineSearch();
    void gradientCheck();
    void printContent();
    void forward(int, int, std::wstring, bool, double&);
    void backward();
    void updateLearningRate(double);
    double getLr();
    double computeEntropy();
    double printSomeProbabilities(int, int, std::wstring);
    void printAllProbabilities(DataProvider&);
    double train(DataProvider&, double&);
    double eval(DataProvider&);
    void generate(DataProvider&);
};

#endif
