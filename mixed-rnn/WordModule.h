/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef WORDMODULE2_H
#define WORDMODULE2_H

#include "Model.h"
#include "Vector.h"
#include <vector>
#include <string>
#include <unordered_map>

const int MAX_WORD_LENGTH = 200;

class WordModule2 {
  private:
    // reference to shared model
    Model& model_;
    std::unordered_map<wchar_t, int>& char2int_;
    std::unordered_map<int, wchar_t>& int2char_;

    // temporary results in dimension d and m
    Vector dcTemp_;
    Vector mcTemp_;
    Vector dwTemp_;
    Vector mwTemp_;

    // character level variables
  public:
    std::vector<Vector> hp_;
    std::vector<Vector> mup_;

  private:
    std::vector<Vector> yp_;
    std::vector<Vector> nup_;
    std::vector<int> cp_;
    Vector Yt_;

  public:
    // word level variables
    int wt_;
    int wtp1_;
    Vector Ht_;
    Vector lambda_;

    int lastChar;
    WordModule2(Model&, std::unordered_map<wchar_t, int>&,
        std::unordered_map<int, wchar_t>&);
    ~WordModule2();
    void loadData(int, int, std::string&);
    void forward(Vector&, Vector&, double&, double&);
    void backward(Vector&, Vector&, Vector&, Vector&, Vector&, Vector&);
    void printChars();
    std::string generate(int, Vector&, Vector&);
};

#endif
