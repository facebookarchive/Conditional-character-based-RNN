/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Utils.h"

double uniRand() {
  return (rand() + 1.0) / (1.0 + RAND_MAX);
}

double randn() {
  double res = 0.0;
  for (int i=0; i<10; i++) {
    res += (uniRand() * 2.0 - 1.0);
  }
  res = res / 10;
  return res;
}

int intRand(int m, int M) {
  return m + (rand() % (M-m));
}

// sample from vector of probabilities
int sampleFromVector(Vector& p) {
  double* cum = new double[p.m_];
  cum[0] = p.get(0);
  for (int i=1; i<p.m_; i++) {
    cum[i] = cum[i-1] + p.get(i);
  }
  double r = uniRand();
  int sample = 0;
  for (int i=0; i<p.m_; i++) {
    if (cum[i] >= r) {
      sample = i;
      break;
    }
  }
  delete[] cum;
  return sample;
}
