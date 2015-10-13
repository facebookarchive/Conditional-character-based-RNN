/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Rnn.h"
#include <math.h>
#include <iostream>
#include <float.h>
#include <chrono>

extern bool VERBOSE;

Rnn::Rnn(Model& modelRef, std::unordered_map<wchar_t, int>& char2int,
      std::unordered_map<int, wchar_t>& int2char, int T, double learningRate)
    : model_(modelRef),
      char2int_(char2int),
      int2char_(int2char),
      generator_(modelRef, char2int, int2char),
      firstWordHidden_(modelRef.mw),
      firstCharHidden_(modelRef.mc),
      lastWordHidden_(modelRef.mw),
      lastCharHidden_(modelRef.mc),
      lastLambda_(modelRef.mw),
      lastMu_(modelRef.mc) {
  T_ = T;
  lr_ = learningRate;
  lr0_ = lr_;
  for (int t=0; t<T_; t++) {
    WordModule2 wm(modelRef, char2int, int2char);
    net_.push_back(wm);
  }
  reset();
}

void Rnn::reset() {
  firstWordHidden_.fillValue(0.0);
  firstCharHidden_.fillValue(0.0);
  lastWordHidden_.fillValue(0.0);
  lastCharHidden_.fillValue(0.0);
  lastLambda_.fillValue(0.0);
  lastMu_.fillValue(0.0);
  step_ = 0;
}

void Rnn::updateLearningRate(double shrinkVal) {
  lr_ /= shrinkVal;
  if (lr_ < 0.000001 * lr0_) {
    lr_ = 0.000001 * lr0_;
  }
}

double Rnn::getLr() {
  return lr_;
}

void Rnn::lineSearch() {
  // saving the model
  Model modelSave(model_);

  double wordEntropy = 0.0;
  double charEntropy = 0.0;

  for (int i=0; i<20; i++) {
    model_.update(0.001);
    computeEntropy(wordEntropy, charEntropy);
    printf("%8.3f ", model_.alpha_ * wordEntropy
        + (1.0 - model_.alpha_) * charEntropy);
  }
  printf("\n");

  // returning to original model
  model_.copy(modelSave);
}

void Rnn::gradientCheck() {
  // saving the model
  Model modelSave(model_);
  // computing initial cost
  double initWordEntropy = 0.0;
  double initCharEntropy = 0.0;
  computeEntropy(initWordEntropy, initCharEntropy);
  double initEntropy = model_.alpha_ * initWordEntropy
      + (1.0 - model_.alpha_) * initCharEntropy;
  // printf("%8.3f\n", initEntropy);

  // storage for linearization and difference
  int nSteps = 30;
  double* linearization = new double[nSteps];
  double* difference = new double[nSteps];

  double maxPow = 2;
  double minPow = -7;
  double c = pow(10.0, (maxPow - minPow) / (nSteps-1));
  double t = - minPow / log10(c);

  double wordEntropy = 0.0;
  double charEntropy = 0.0;
  // model_.pickDeltas();

  for (int j=0; j<4; j++) {
    // pick random direction
    model_.copy(modelSave);
    model_.pickDeltas();

    for (int i=0; i<nSteps; i++) {
      double gamma = pow(c, (double)i-t);
      model_.addDeltas(gamma);
      computeEntropy(wordEntropy, charEntropy);
      double entropy = model_.alpha_ * wordEntropy
          + (1.0 - model_.alpha_) * charEntropy;
      difference[i] = entropy - initEntropy;
      linearization[i] = gamma * model_.gradTDelta();
      model_.addDeltas(-gamma);
    }

    for (int i=0; i<nSteps; i++) {
      printf("%+10.2e ", linearization[i]);
    }
    std::cout << std::endl;
    for (int i=0; i<nSteps; i++) {
      printf("%+10.2e ", difference[i]);
    }
    std::cout << std::endl;
    for (int i=0; i<nSteps; i++) {
      printf("%+10.3f ", difference[i] / linearization[i]);
    }
    // std::cout << std::endl;
    std::cout << std::endl;
  }

  delete[] linearization;
  delete[] difference;

  // returning to original model
  model_.copy(modelSave);
  computeEntropy(wordEntropy, charEntropy);
  // checking that the entropy is the same as at the beginning
}

void Rnn::backward() {
  model_.resetGradients();
  for (int t=T_-1; t>=0; t--) {
    // printf("step:%3d\n", t);
    if (t == T_ - 1) {
      int Ptm1 = net_[t-1].lastChar;
      net_[t].backward(net_[t-1].Ht_, net_[t-1].hp_[Ptm1 - 1], lastWordHidden_,
                        lastLambda_, lastCharHidden_, lastMu_);
    } else if (t == 0) {
      net_[t].backward(firstWordHidden_, firstCharHidden_, net_[t+1].Ht_,
                        net_[t+1].lambda_, net_[t+1].hp_[0], net_[t+1].mup_[0]);
    } else {
      int Ptm1 = net_[t-1].lastChar;
      net_[t].backward(net_[t-1].Ht_, net_[t-1].hp_[Ptm1 - 1], net_[t+1].Ht_,
                        net_[t+1].lambda_, net_[t+1].hp_[0], net_[t+1].mup_[0]);
    }
  }

  // printContent();
  // gradientCheck();
  // lineSearch();
  model_.update(lr_ / T_);
}

void Rnn::computeEntropy(double& wordEntropy, double& charEntropy) {
  charEntropy = 0.0;
  wordEntropy = 0.0;
  for (int i=0; i<T_; i++) {
    if (i == 0) {
      net_[0].forward(firstWordHidden_, firstCharHidden_, wordEntropy, charEntropy);
    } else {
      int Ptm1 = net_[i - 1].lastChar;
      net_[i].forward(net_[i - 1].Ht_, net_[i - 1].hp_[Ptm1 - 1], wordEntropy, charEntropy);
    }
  }
}

void Rnn::printContent() {
  for (int i=0; i<T_; i++) {
    printf("token:%d ", net_[i].wt_);
    printf("next_token:%d ", net_[i].wtp1_);
    net_[i].printChars();
  }
  printf("\n");
}

void Rnn::forward(int w, int wtp1, std::string& stp1, bool train,
                  double& wordEntropy, double& charEntropy, int& nChars) {
  net_[step_].loadData(w, wtp1, stp1);
  if (step_ == 0) {
    net_[0].forward(firstWordHidden_, firstCharHidden_, wordEntropy, charEntropy);
  } else {
    int Ptm1 = net_[step_ - 1].lastChar;
    net_[step_].forward(net_[step_ - 1].Ht_, net_[step_ - 1].hp_[Ptm1-1], wordEntropy, charEntropy);
  }
  // nChars counts the number of letters in the word plus the space
  nChars += net_[step_].lastChar;
  step_++;
  if (step_ == T_) {
    if (train) {
      backward();
    }
    // copying the usefull hiddens
    firstWordHidden_.copy(net_[step_ - 1].Ht_);
    int Ptm1 = net_[step_ - 1].lastChar;
    firstCharHidden_.copy(net_[step_ - 1].hp_[Ptm1 - 1]);
    step_ = 0;
  }
}

void Rnn::train(DataProvider& dp,
                bool doTrain,
                double& seconds,
                double& wordEntropy,
                double& charEntropy,
                int& nChars) {
  dp.initIterator();
  int nWords = dp.getNumTokens();
  wordEntropy = 0.0;
  charEntropy = 0.0;
  double loss = 0.0;
  nChars = 0;
  int now = 0;
  int next = 0;
  std::string nextWord;
  auto tic = std::chrono::steady_clock::now();
  auto toc = std::chrono::steady_clock::now();
  seconds = 0.0;
  for (int i=0; i<nWords; i++) {
    if ( i%10000 == 0 && i>0 ) {
      toc = std::chrono::steady_clock::now();
      seconds = std::chrono::duration_cast<std::chrono::milliseconds>
          (toc - tic).count() / 1000.0;
      if (VERBOSE) {
        printf("train=%d token %8d/%8d ", doTrain, i, nWords);
        printf("char-entropy=%12.6e ", charEntropy / nChars);
        printf("word-entropy=%12.6e ", charEntropy / i);
        printf("word-model-entropy=%12.6e ", wordEntropy / i);
        printf("loss=%12.6e ", loss);
        printf("time=%-8.0f ", seconds);
        printf("sec/epoch=%-8.0f ", seconds / i * nWords);
        printf("words/sec=%-8.0f ", i / seconds);
        printf("chars/sec=%-8.0f ", nChars / seconds);
        std::cout << std::endl;
        if (doTrain) {
          // generate(dp);
        }
      }
    }
    dp.getToken(now, next, nextWord);
    forward(now, next, nextWord, doTrain, wordEntropy, charEntropy, nChars);
    loss = model_.alpha_ * wordEntropy + (1.0 - model_.alpha_) * charEntropy;
  }
}

void Rnn::eval(DataProvider& dp,
                double& wordEntropy,
                double& charEntropy,
                int& nChars) {
  double time = 0.0;
  train(dp, false, time, wordEntropy, charEntropy, nChars);
}

void Rnn::generate(DataProvider& dp) {
  int wordId = 0;
  std::string word;
  dp.randomWord(wordId, word);
  std::cout << word << " ";

  Vector Htm1(firstWordHidden_);
  Vector htm1P(firstCharHidden_);

  for (int i=0; i<20; i++) {
    word = generator_.generate(wordId, Htm1, htm1P);
    word.pop_back(); // removing the underscore

    Htm1.copy(generator_.Ht_);
    htm1P.copy(generator_.hp_[generator_.lastChar - 1]);

    wordId = dp.getWordId(word);
    if (wordId != 0) {
      std::cout << "#" << word << " ";
    } else {
      std::cout << word << " ";
    }
  }
  std::cout << std::endl;
}
