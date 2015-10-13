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
#include <float.h>

WordModule2::WordModule2(Model& modelRef,
    std::unordered_map<wchar_t,int>& char2int,
    std::unordered_map<int, wchar_t>& int2char)
    : model_(modelRef),
      char2int_(char2int),
      int2char_(int2char),
      dcTemp_(modelRef.dc),
      mcTemp_(modelRef.mc),
      dwTemp_(modelRef.dwV2),
      mwTemp_(modelRef.mw),
      hp_(MAX_WORD_LENGTH, Vector(modelRef.mc)),
      mup_(MAX_WORD_LENGTH, Vector(modelRef.mc)),
      yp_(MAX_WORD_LENGTH, Vector(modelRef.dc)),
      cp_(MAX_WORD_LENGTH, 0),
      Yt_(modelRef.dwV2),
      Ht_(modelRef.mw),
      lambda_(modelRef.mw) {
  wt_ = 0;
  wtp1_ = 0;
  dcTemp_.fillValue(0.0);
  mcTemp_.fillValue(0.0);
  dwTemp_.fillValue(0.0);
  mwTemp_.fillValue(0.0);
}

WordModule2::~WordModule2() {
}

void WordModule2::loadData(int w, int wtp1, std::string& string_tp1) {
  wt_ = w;
  wtp1_ = wtp1;

  wchar_t wideChars[1000];
  size_t nChars = mbstowcs(wideChars, string_tp1.c_str(), 1000);

  // lastChar is the length of the string plus one for the first char
  lastChar = nChars + 1;
  cp_[0] = char2int_['_'];
  for (int i=0; i<lastChar; i++) {
    cp_[i+1] = char2int_[wideChars[i]];
  }
  cp_[lastChar+1] = char2int_['_'];
}

// forward takes as input the previous hidden
void WordModule2::forward(Vector& Htm1,
                          Vector& htm1P,
                          double& wordEntropy,
                          double& charEntropy) {

  Ht_.getRow(model_.Aw_, wt_);
  Ht_.matrixVector(1.0, model_.Rw_, Htm1, 1.0);
  Ht_.sigmoid();

  if (model_.alpha_ > 0.01) {
    Yt_.matrixVector(1.0, model_.Uw_, Ht_, 0.0);
    Yt_.softMax();
    wordEntropy -= log(Yt_.get(wtp1_) + DBL_MIN) / log(2.0);
  }

  for (int i=0; i<lastChar; i++) {
    // computing character hidden
    hp_[i].getRow(model_.Ac_, cp_[i]);
    if (i==0) {
      hp_[i].matrixVector(1.0, model_.Rc_, htm1P, 1.0);
    } else {
      hp_[i].matrixVector(1.0, model_.Rc_, hp_[i-1], 1.0);
    }
    hp_[i].matrixVector(1.0, model_.Q_, Ht_, 1.0);
    hp_[i].sigmoid();

    // computing character output
    yp_[i].matrixVector(1.0, model_.Uc_, hp_[i], 0.0);
    yp_[i].softMax();
    charEntropy -= log(yp_[i].get(cp_[i+1]) + DBL_MIN) / log(2.0);
    // std::cout << charEntropy << " " << yp_[i].get(cp_[i+1]) << std::endl;
  }
}

void WordModule2::backward(Vector& Htm1, Vector& htm1P, Vector& Htp1,
                           Vector& lambdatp1, Vector& htp10, Vector& mutp10)  {
  lambda_.fillValue(0.0);

  for (int i=lastChar-1; i>=0; i--) {
    // printf("backward: %d:%d\n", i, cp_[i]);

    if (i==lastChar-1) {
      mcTemp_.copy(htp10);
      mcTemp_.aTimesOneMinusA();
      mcTemp_.timesInPlace(mutp10);
    } else {
      mcTemp_.copy(hp_[i+1]);
      mcTemp_.aTimesOneMinusA();
      mcTemp_.timesInPlace(mup_[i+1]);
    }

    dcTemp_.fillValue(0.0);
    dcTemp_.set(cp_[i+1], 1.0);
    dcTemp_.addInPlace(-1.0, yp_[i]);
    dcTemp_.scale((1.0 - model_.alpha_) / log(2.0));

    mup_[i].matrixTVector(1.0, model_.Uc_, dcTemp_, 0.0);
    mup_[i].matrixTVector(1.0, model_.Rc_, mcTemp_, 1.0);

    // computing the contribution of all the hiddens to the word hidden (Q)
    if (i < lastChar-1) {
      lambda_.matrixTVector(1.0, model_.Q_, mcTemp_, 1.0);
    }

    // computing the gradients
    model_.gUc_.vectorVectorT(-1.0, dcTemp_, hp_[i]);
    if (i < lastChar-1) {
      model_.gQ_.vectorVectorT(-1.0, mcTemp_, Ht_);
      model_.gRc_.vectorVectorT(-1.0, mcTemp_, hp_[i]);
      model_.gAc_.addRow(cp_[i+1], -1.0, mcTemp_);
    }
  }

  // contribution of next hidden to word hidden
  mwTemp_.copy(Htp1);
  mwTemp_.aTimesOneMinusA();
  mwTemp_.timesInPlace(lambdatp1);
  lambda_.matrixTVector(1.0, model_.Rw_, mwTemp_, 1.0);

  if (model_.alpha_ > 0.01) {
    // contribution of the prediction to hidden
    dwTemp_.fillValue(0.0);
    dwTemp_.set(wtp1_, 1.0);
    dwTemp_.addInPlace(-1.0, Yt_);
    dwTemp_.scale(model_.alpha_ / log(2.0));
    lambda_.matrixTVector(1.0, model_.Uw_, dwTemp_, 1.0);

    // compute the output gradient
    model_.gUw_.vectorVectorT(-1.0, dwTemp_, Ht_);
  }

  // contribution of first char to word hidden
  mcTemp_.copy(hp_[0]);
  mcTemp_.aTimesOneMinusA();
  mcTemp_.timesInPlace(mup_[0]);
  lambda_.matrixTVector(1.0, model_.Q_, mcTemp_, 1.0);

  model_.gAc_.addRow(cp_[0], -1.0, mcTemp_);

  mwTemp_.copy(Ht_);
  mwTemp_.aTimesOneMinusA();
  mwTemp_.timesInPlace(lambda_);

  // gradient of the word parameters
  model_.gQ_.vectorVectorT(-1.0, mcTemp_, Ht_);
  model_.gRw_.vectorVectorT(-1.0, mwTemp_, Htm1);
  model_.gRc_.vectorVectorT(-1.0, mcTemp_, htm1P);

  model_.updatedWordsList_.insert(wt_);
  model_.gAw_.addRow(wt_, -1.0, mwTemp_);

}

void WordModule2::printChars() {
  for (int i=0; i<lastChar; i++) {
    printf("%d ", cp_[i]);
  }
  printf("%d ", cp_[lastChar]);
  printf("\n");
}

std::string WordModule2::generate(int wordId, Vector& Htm1, Vector& htm1P) {
  Ht_.getRow(model_.Aw_, wordId);
  Ht_.matrixVector(1.0, model_.Rw_, Htm1, 1.0);
  Ht_.sigmoid();

  int charId = char2int_['_'];
  std::string res;
  wchar_t c;
  char mbs[16];
  int nBytes = 0;

  int i = 0;
  while ((charId!=char2int_['_'] || i==0) && i<MAX_WORD_LENGTH) {
    // computing character hidden
    hp_[i].getRow(model_.Ac_, charId);
    if (i==0) {
      hp_[i].matrixVector(1.0, model_.Rc_, htm1P, 1.0);
    } else {
      hp_[i].matrixVector(1.0, model_.Rc_, hp_[i-1], 1.0);
    }
    hp_[i].matrixVector(1.0, model_.Q_, Ht_, 1.0);
    hp_[i].sigmoid();

    // computing character output
    yp_[i].matrixVector(1.0, model_.Uc_, hp_[i], 0.0);
    yp_[i].softMax();
    charId = sampleFromVector(yp_[i]);
    c = int2char_[charId];
    nBytes = wctomb(mbs, c);
    for (int j=0; j<nBytes; j++) {
      res.push_back(mbs[j]);
    }
    i++;
  }
  lastChar = i;

  if (i==MAX_WORD_LENGTH && charId!=char2int_['_']) {
    res.push_back('_');
  }

  return res;

}
