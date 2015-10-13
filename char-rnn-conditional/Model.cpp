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
#include <fstream>
#include <string.h>

Model::Model(int m,
            int d)
    : R_(m, m),
      A_(d, m),
      gR_(m, m),
      gA_(d, m),
      dR_(m, m),
      dA_(d, m) {
  m_ = m;
  d_ = d;
}

Model::Model(const Model& other)
    : R_(other.R_),
      A_(other.A_),
      U_(other.U_),
      gR_(other.gR_),
      gA_(other.gA_),
      gU_(other.gU_),
      dR_(other.dR_),
      dA_(other.dA_),
      dU_(other.dU_) {
  m_ = other.m_;
  d_ = other.d_;
}

Model::~Model() {
}

void Model::copy(Model& other) {
  R_.copy(other.R_);
  gR_.copy(other.gR_);
  A_.copy(other.A_);
  gA_.copy(other.gA_);

  // clearing the output layer and re-building with provided model
  U_.clear();
  gU_.clear();
  dU_.clear();
  for (auto it=other.U_.begin(); it!=other.U_.end(); ++it) {
    Matrix paramTemp(it->second); // temporary copy of the other matrix
    U_.insert({it->first, paramTemp});
    Matrix gradTemp(other.gU_[it->first]);
    gU_.insert({it->first, gradTemp});
    Matrix deltaTemp(other.dU_[it->first]);
    deltaTemp.fillValue(0.0);
    dU_.insert({it->first, deltaTemp});
  }
}

void Model::addHistory(std::wstring history) {
  Matrix paramTemp(d_, m_);
  paramTemp.fillRandn();
  Matrix gradTemp(d_, m_);
  gradTemp.fillValue(0.0);
  Matrix deltaTemp(d_, m_);
  deltaTemp.fillValue(0.0);
  U_.insert({history, paramTemp});
  gU_.insert({history, gradTemp});
  dU_.insert({history, deltaTemp});
}

void Model::resetGradients() {
  gR_.fillValue(0.0);
  gA_.fillValue(0.0);
  for (auto it=ngramHistory_.begin(); it!=ngramHistory_.end(); ++it) {
    gU_[*it].fillValue(0.0);
  }
  ngramHistory_.clear();
}

void Model::resetDeltas() {
  dR_.fillValue(0.0);
  dA_.fillValue(0.0);
  for (auto it=dU_.begin(); it!=dU_.end(); ++it) {
    it->second.fillValue(0.0);
  }
}

void Model::update(double gamma) {
  R_.addInPlace(-gamma, gR_);
  A_.addInPlace(-gamma, gA_);

  for (auto it=ngramHistory_.begin(); it!=ngramHistory_.end(); ++it) {
    U_[*it].addInPlace(-gamma, gU_[*it]);
  }
}

void Model::initialize(char* type) {
  if (strcmp(type, "diagonal")==0) {
    std::cout << "diagonal init" << std::endl;
    R_.fillRandn(0.01);
    R_.setDiag(0.95);
  } else {
    R_.fillRandn();
  }
  A_.fillRandn();
}

void Model::pickDeltas() {
  resetDeltas();
  dR_.fillRandn();
  dA_.fillRandn();
  for (auto it=dU_.begin(); it!=dU_.end(); ++it) {
    it->second.fillRandn();
  }
}

void Model::addDeltas(double gamma) {
  R_.addInPlace(gamma, dR_);
  A_.addInPlace(gamma, dA_);
  for (auto it=dU_.begin(); it!=dU_.end(); ++it) {
    U_[it->first].addInPlace(gamma, dU_[it->first]);
  }
}

double Model::gradTDelta() {
  double result = 0.0;
  result += gR_.dotProduct(dR_);
  result += gA_.dotProduct(dA_);
  for (auto it=ngramHistory_.begin(); it!=ngramHistory_.end(); ++it) {
    result += gU_[*it].dotProduct(dU_[*it]);
  }
  return result;
}

