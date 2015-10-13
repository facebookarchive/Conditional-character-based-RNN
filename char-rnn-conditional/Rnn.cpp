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
#include <chrono>

extern bool VERBOSE;

Rnn::Rnn(Model& modelRef, int T, double learningRate)
    : model_(modelRef),
      generator_(modelRef),
      firstHidden_(modelRef.m_),
      lastHidden_(modelRef.m_),
      lastLambda_(modelRef.m_) {
  T_ = T;
  step_ = 0;
  lr_ = learningRate;
  lr0_ = learningRate;
  for (int t=0; t<T_; t++) {
    WordModule wm(modelRef);
    net_.push_back(wm);
  }
  firstHidden_.fillValue(0.0);
  lastHidden_.fillValue(0.0);
  lastLambda_.fillValue(0.0);
}

void Rnn::reset() {
  firstHidden_.fillValue(0.0);
  lastHidden_.fillValue(0.0);
  lastLambda_.fillValue(0.0);
  step_ = 0;
}

void Rnn::lineSearch() {
  // saving the model
  Model modelSave(model_);

  for (int i=0; i<20; i++) {
    model_.update(0.001);
    printf("%8.3f ", computeEntropy());
  }
  printf("\n");

  // returning to original model
  model_.copy(modelSave);
}

void Rnn::gradientCheck() {
  // saving the model
  Model modelSave(model_);
  // computing initial cost
  double initEntropy = computeEntropy();
  // printf("%8.3f\n", initEntropy);

  // storage for linearization and difference
  int nSteps = 30;
  double* linearization = new double[nSteps];
  double* difference = new double[nSteps];

  double maxPow = 2;
  double minPow = -7;
  double c = pow(10.0, (maxPow - minPow) / (nSteps-1));
  double t = - minPow / log10(c);

  for (int j=0; j<10; j++) {
    // pick random direction
    model_.pickDeltas();

    for (int i=0; i<nSteps; i++) {
      double gamma = pow(c, (double)i-t);
      model_.addDeltas(gamma);
      double entropy = computeEntropy();
      difference[i] = entropy - initEntropy;
      linearization[i] = gamma * model_.gradTDelta();
      model_.addDeltas(-gamma);
    }

    for (int i=0; i<nSteps; i++) {
      printf("%+10.3f ", difference[i] / linearization[i]);
    }
    std::cout << std::endl;
  }

  delete[] linearization;
  delete[] difference;

  model_.copy(modelSave);
}

void Rnn::backward() {
  model_.resetGradients();
  for (int t=T_-1; t>=0; t--) {
    if (t == T_ - 1) {
      net_[t].backward(net_[t-1].ht_, lastHidden_, lastLambda_);
    } else if (t == 0) {
      net_[t].backward(firstHidden_, net_[t+1].ht_, net_[t+1].lambda_);
    } else {
      net_[t].backward(net_[t-1].ht_, net_[t+1].ht_, net_[t+1].lambda_);
    }
  }
  // gradientCheck();
  // lineSearch();
  model_.update(lr_ / T_);
  model_.resetGradients();
}

void Rnn::updateLearningRate(double shrinkVal) {
  lr_ /= shrinkVal;
  if (lr_ < 0.0001 * lr0_) {
    lr_ = 0.0001 * lr0_;
  }
}

double Rnn::getLr() {
  return lr_;
}

double Rnn::computeEntropy() {
  double entropy = 0.0;
  for (int i=0; i<T_; i++) {
    if (i == 0) {
      entropy += net_[0].computeEntropy(firstHidden_);
    } else {
      entropy += net_[i].computeEntropy(net_[i - 1].ht_);
    }
  }
  return entropy;
}

void Rnn::forward(int xt, int xtp1, std::wstring history, bool train,
                  double& entropy) {
  if (step_ == 0) {
    entropy += net_[0].forward(xt, xtp1, history, firstHidden_);
  } else {
    entropy += net_[step_].forward(xt, xtp1, history, net_[step_ - 1].ht_);
  }
  step_++;
  if (step_ == T_) {
    if (train) {
      backward();
    }
    firstHidden_.copy(net_[step_ - 1].ht_);
    step_ = 0;
  }
}

double Rnn::printSomeProbabilities(int xt, int xtp1, std::wstring history) {
  double p = 0;
  if (step_ == 0) {
    p = net_[0].computeProbability(xt, xtp1, history, firstHidden_);
  } else {
    p = net_[step_].computeProbability(xt, xtp1, history, net_[step_ - 1].ht_);
  }
  step_++;
  if (step_ == T_) {
    firstHidden_.copy(net_[step_ - 1].ht_);
    step_ = 0;
  }
  return p;
}

void Rnn::printAllProbabilities(DataProvider& dp) {
  dp.initIterator();
  int nTokens = dp.getNumTokens();
  int now = 0;
  int next = 0;
  double p = 0.0;
  std::wstring history;

  for (int i=0; i<nTokens; i++) {
    dp.getToken(now, next);
    history = dp.getHistory();
    p = printSomeProbabilities(now, next, history);
    printf("%d\t%.10f\t%lc\n", next, p, dp.getChar(next));
  }
  std::cout << std::endl;
}

double Rnn::train(DataProvider& dp, double& seconds) {
  dp.initIterator();
  int nTokens = dp.getNumTokens();
  double entropy = 0.0;
  int now = 0;
  int next = 0;
  std::wstring history;
  auto tic = std::chrono::steady_clock::now();
  auto toc = std::chrono::steady_clock::now();
  seconds = 0.0;
  for (int i=0; i<nTokens; i++) {
    if ( VERBOSE && i%100000 == 0 && i>0 ) {
      toc = std::chrono::steady_clock::now();
      seconds = std::chrono::duration_cast<std::chrono::milliseconds>
        (toc - tic).count() / 1000.0;
      printf("train token %8d/%8d ", i, nTokens);
      printf("char entropy=%8.6e ", entropy/i);
      printf("time=%5.3e ", seconds);
      printf("sec/epoch=%-8.0f ", seconds / i * nTokens);
      printf("chars/sec=%5.3e ", i / seconds);
      std::cout << std::endl;
      // generate(dp);
    }
    dp.getToken(now, next);
    history = dp.getHistory();
    forward(now, next, history, true, entropy);
  }
  toc = std::chrono::steady_clock::now();
  seconds = std::chrono::duration_cast<std::chrono::milliseconds>
      (toc - tic).count() / 1000.0;
  return entropy / nTokens;
}

double Rnn::eval(DataProvider& dp) {
  dp.initIterator(); // put iterator at begining of dataset
  reset(); // reset hidden
  int nTokens = dp.getNumTokens(); // get number of words
  double entropy = 0.0;
  int now = 0;
  int next = 0;
  std::wstring history;
  for (int i=0; i<nTokens; i++) {
    if ( VERBOSE && (i % 100000 == 0) ) {
      printf("test token %8d/%8d entropy=%8.6e\n", i, nTokens, entropy/i);
    }
    dp.getToken(now, next);
    history = dp.getHistory();
    forward(now, next, history, false, entropy);
  }
  return entropy / nTokens;
}

void Rnn::generate(DataProvider& dp) {
  Vector htm1(firstHidden_);
  int nChars = dp.getNumChars();
  int ct = (int)(uniRand() * nChars);

  std::wstring history;
  std::wstring res;

  for (int i=0; i<200; i++) {
    wchar_t ch = dp.getChar(ct);
    res.push_back(ch);
    history.push_back(ch);
    if (history.size() > dp.ngramOrder_) {
      history.erase(0, 1);
    }

    ct = generator_.generate(ct, history, htm1);
    htm1.copy(generator_.ht_);
  }
  std::wcout << res << std::endl;
}
