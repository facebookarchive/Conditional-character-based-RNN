/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "WordModule.h"
#include <iostream>
#include <math.h>

WordModule::WordModule(Model& modelRef)
    : model_(modelRef),
      dTemp_(modelRef.d_),
      mTemp_(modelRef.m_),
      ht_(modelRef.m_),
      yt_(modelRef.d_),
      lambda_(modelRef.m_) {

  xt_ = 0;
  xtp1_ = 0;
  dTemp_.fillValue(0.0);
  mTemp_.fillValue(0.0);
}

WordModule::~WordModule() {
}

// forward takes as input the previous hidden
double WordModule::forward(int xt, int xtp1, std::wstring hist, Vector& htm1) {
  xt_ = xt;
  xtp1_ = xtp1;
  history_ = hist;
  double entropy = 0;

  ht_.getRow(model_.A_, xt_);
  ht_.matrixVector(1.0, model_.R_, htm1, 1.0);
  ht_.sigmoid();


  // if the model does not contain the current history, init with random
  if (model_.U_.count(history_) == 0) {
    model_.addHistory(history_);
  }
  Matrix &temp = model_.U_.at(history_);
  yt_.matrixVector(1.0, temp, ht_, 0.0);
  yt_.softMax();

  entropy -= log(yt_.get(xtp1)) / log(2.0);

  return entropy;
}

// compute a forward without changing things
double WordModule::computeProbability(int xt, int xtp1, std::wstring hist,
                                      Vector& htm1) {
  xt_ = xt;
  xtp1_ = xtp1;
  history_ = hist;


  ht_.getRow(model_.A_, xt_);
  ht_.matrixVector(1.0, model_.R_, htm1, 1.0);
  ht_.sigmoid();

  yt_.matrixVector(1.0, model_.U_[history_], ht_, 0.0);
  yt_.softMax();

  return yt_.get(xtp1_);
}

// compute a forward without changing things
double WordModule::computeEntropy(Vector& htm1) {
  double entropy = 0;

  ht_.getRow(model_.A_, xt_);
  ht_.matrixVector(1.0, model_.R_, htm1, 1.0);
  ht_.sigmoid();

  yt_.matrixVector(1.0, model_.U_[history_], ht_, 0.0);
  yt_.softMax();

  entropy -= log(yt_.get(xtp1_)) / log(2.0);

  return entropy;
}

void WordModule::backward(Vector& htm1, Vector& htp1, Vector& lambdatp1) {
  lambda_.fillValue(0.0);

  // computing derivatives of the output
  mTemp_.copy(htp1);
  mTemp_.aTimesOneMinusA();
  mTemp_.timesInPlace(lambdatp1);

  // computing derivatives of the hidden
  dTemp_.fillValue(0.0);
  dTemp_.set(xtp1_, 1.0);
  dTemp_.addInPlace(-1.0, yt_);
  dTemp_.scale(1 / log(2.0));

  lambda_.matrixTVector(1.0, model_.U_[history_], dTemp_, 0.0);
  lambda_.matrixTVector(1.0, model_.R_, mTemp_, 1.0);

  // computing derivatives of the output
  mTemp_.copy(ht_);
  mTemp_.aTimesOneMinusA();
  mTemp_.timesInPlace(lambda_);

  // computing the gradients
  model_.ngramHistory_.insert(history_);
  model_.gU_[history_].vectorVectorT(-1.0, dTemp_, ht_);
  model_.gR_.vectorVectorT(-1.0, mTemp_, htm1);
  model_.gA_.addRow(xt_, -1.0, mTemp_);
}

int WordModule::generate(int ct, std::wstring hist, Vector& htm1) {
  ht_.getRow(model_.A_, ct);
  ht_.matrixVector(1.0, model_.R_, htm1, 1.0);
  ht_.sigmoid();

  if (model_.U_.count(hist) == 0) {
    model_.addHistory(hist);
  }
  yt_.matrixVector(1.0, model_.U_[hist], ht_, 0.0);
  yt_.softMax();

  return sampleFromVector(yt_);
}
