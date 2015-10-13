/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include "Vector.h"

class Vector;

double uniRand();
double randn();
int intRand(int, int);
int sampleFromVector(Vector&);

#endif
