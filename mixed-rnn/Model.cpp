/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Model.h"
#include <iostream>
#include <cstring>

Model::Model(int mWord,
            int mChar,
            int dWordV1,
            int dWordV2,
            int dChar,
            double alpha)
    : Rw_(mWord, mWord),
      Aw_(dWordV1, mWord), // storing the transpose for efficient lookup
      Uw_(dWordV2, mWord),
      Rc_(mChar, mChar),
      Ac_(dChar, mChar), // storing the transpose for efficient lookup
      Uc_(dChar, mChar),
      Ic_(mChar, mWord),
      Q_(mChar, mWord),
      gRw_(mWord, mWord),
      gAw_(dWordV1, mWord),
      gUw_(dWordV2, mWord),
      gRc_(mChar, mChar),
      gAc_(dChar, mChar),
      gUc_(dChar, mChar),
      gIc_(mChar, mWord),
      gQ_(mChar, mWord),
      dRw_(mWord, mWord),
      dAw_(dWordV1, mWord),
      dUw_(dWordV2, mWord),
      dRc_(mChar, mChar),
      dAc_(dChar, mChar),
      dUc_(dChar, mChar),
      dIc_(mChar, mWord),
      dQ_(mChar, mWord){

  mw = mWord;
  mc = mChar;
  dwV1 = dWordV1;
  dwV2 = dWordV2;
  dc = dChar;
  alpha_ = alpha;
}

Model::Model(const Model& other)
    : Rw_(other.Rw_),
      Aw_(other.Aw_),
      Uw_(other.Uw_),
      Rc_(other.Rc_),
      Ac_(other.Ac_),
      Uc_(other.Uc_),
      Ic_(other.Ic_),
      Q_(other.Q_),
      gRw_(other.gRw_),
      gAw_(other.gAw_),
      gUw_(other.gUw_),
      gRc_(other.gRc_),
      gAc_(other.gAc_),
      gUc_(other.gUc_),
      gIc_(other.gIc_),
      gQ_(other.gQ_),
      dRw_(other.dRw_),
      dAw_(other.dAw_),
      dUw_(other.dUw_),
      dRc_(other.dRc_),
      dAc_(other.dAc_),
      dUc_(other.dUc_),
      dIc_(other.dIc_),
      dQ_(other.dQ_) {
  mw = other.mw;
  mc = other.mc;
  dwV1 = other.dwV1;
  dwV2 = other.dwV2;
  dc = other.dc;
  alpha_ = other.alpha_;
}

Model::~Model() {
}

void Model::copy(Model& other) {
  // copy the models
  Rw_.copy(other.Rw_);
  Aw_.copy(other.Aw_);
  Uw_.copy(other.Uw_);

  Rc_.copy(other.Rc_);
  Ac_.copy(other.Ac_);
  Uc_.copy(other.Uc_);

  Ic_.copy(other.Ic_);
  Q_.copy(other.Q_);

  // copy the gradients
  gRw_.copy(other.gRw_);
  gAw_.copy(other.gAw_);
  gUw_.copy(other.gUw_);

  gRc_.copy(other.gRc_);
  gAc_.copy(other.gAc_);
  gUc_.copy(other.gUc_);

  gIc_.copy(other.gIc_);
  gQ_.copy(other.gQ_);

  // set the delta to zero
  resetDeltas();

  alpha_ = other.alpha_;
}

void Model::resetGradients() {
  gRw_.fillValue(0.0);

  // whiping only the gradients in the list of words
  for (auto it=updatedWordsList_.begin(); it!=updatedWordsList_.end(); ++it) {
    gAw_.fillRow(*it, 0.0);
  }
  updatedWordsList_.clear();
  // gAw_.fillValue(0.0);
  if (alpha_ > 0.01) {
    gUw_.fillValue(0.0);
  }

  gRc_.fillValue(0.0);
  gAc_.fillValue(0.0);
  gUc_.fillValue(0.0);

  gIc_.fillValue(0.0);
  gQ_.fillValue(0.0);
}

void Model::resetDeltas() {
  dRw_.fillValue(0.0);
  dAw_.fillValue(0.0);
  dUw_.fillValue(0.0);

  dRc_.fillValue(0.0);
  dAc_.fillValue(0.0);
  dUc_.fillValue(0.0);

  dIc_.fillValue(0.0);
  dQ_.fillValue(0.0);
}

void Model::update(double gamma) {
  Rw_.addInPlace(-gamma, gRw_);

  // updating only the words that were seen recently
  for (auto it=updatedWordsList_.begin(); it!=updatedWordsList_.end(); ++it) {
    Aw_.addRow(*it, -gamma, gAw_);
  }
  if (alpha_ > 0.01) {
    Uw_.addInPlace(-gamma, gUw_);
  }

  Rc_.addInPlace(-gamma, gRc_);
  Ac_.addInPlace(-gamma, gAc_);
  Uc_.addInPlace(-gamma, gUc_);

  Ic_.addInPlace(-gamma, gIc_);
  Q_.addInPlace(-gamma, gQ_);
}

void Model::initialize(char* type) {
  if (strcmp(type, "diagonal")==0) {
    Rw_.fillRandn(0.001);
    Rw_.setDiag(0.95);
    Rc_.fillRandn(0.001);
    Rc_.setDiag(0.95);
  } else {
    Rw_.fillRandn();
    Rc_.fillRandn();
  }
  Aw_.fillRandn();
  Uw_.fillRandn();
  Ac_.fillRandn();
  Uc_.fillRandn();
  Ic_.fillRandn();
  Q_.fillRandn();

  resetGradients();
  resetDeltas();
}

void Model::pickDeltas() {
  dRw_.fillRandn(0.1);
  dAw_.fillRandn(0.1);
  dUw_.fillRandn(0.1);

  dRc_.fillRandn(0.1);
  dAc_.fillRandn(0.1);
  dUc_.fillRandn(0.1);

  dIc_.fillRandn(0.1);
  dQ_.fillRandn(0.1);
}

void Model::addDeltas(double gamma) {
  Rw_.addInPlace(gamma, dRw_);
  Aw_.addInPlace(gamma, dAw_);
  Uw_.addInPlace(gamma, dUw_);

  Rc_.addInPlace(gamma, dRc_);
  Ac_.addInPlace(gamma, dAc_);
  Uc_.addInPlace(gamma, dUc_);

  Ic_.addInPlace(gamma, dIc_);
  Q_.addInPlace(gamma, dQ_);
}

double Model::gradTDelta() {
  double result = 0.0;
  result += gRw_.dotProduct(dRw_);
  result += gAw_.dotProduct(dAw_);

  if (alpha_ > 0.01) {
    result += gUw_.dotProduct(dUw_);
  }

  result += gRc_.dotProduct(dRc_);
  result += gAc_.dotProduct(dAc_);
  result += gUc_.dotProduct(dUc_);

  // result += gIc_.dotProduct(dIc_);
  result += gQ_.dotProduct(dQ_);
  return result;
}
