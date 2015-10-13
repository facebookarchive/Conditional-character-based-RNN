/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "Matrix.h"
#include "Vector.h"
#include "Model.h"
#include "Rnn.h"
#include "DataProvider.h"
#include "WordModule.h"
#include "Utils.h"
#include <iostream>
#include <string.h>
#include <float.h>

bool USE_BLAS = false;
bool VERBOSE = true;

int main(int argc, char** argv) {
  int nhidw = 200;
  int nhidc = 100;
  int bptt = 10;
  int nepoch = 1;
  int V1 = 1;
  int V2 = 2000;
  double lr = 0.005;
  double shrinkVal = 1.5;
  std::string trainFile;
  std::string validFile;
  std::string testFile;
  double alpha = 0.5;
  int seed = 1;
  char init[100];
  strcpy(init, "gaussian");


  int ai = 1;
  while(ai < argc) {
    if( strcmp( argv[ai], "--blas") == 0) {
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      USE_BLAS = strcmp(argv[ai+1], "true")==0;
    }
    else if( strcmp( argv[ai], "--trainFile") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      std::string stemp(argv[ai+1]);
      trainFile = stemp;
    }
    else if( strcmp( argv[ai], "--validFile") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      std::string stemp(argv[ai+1]);
      validFile = stemp;
    }
    else if( strcmp( argv[ai], "--testFile") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      std::string stemp(argv[ai+1]);
      testFile = stemp;
    }
    else if( strcmp( argv[ai], "--init") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      strcpy(init, argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--V1") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      V1 = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--V2") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      V2 = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--alpha") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      alpha = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--nhidc") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      nhidc = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--nhidw") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      nhidw = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--bptt") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      bptt = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--nepoch") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      nepoch = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--seed") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      seed = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--lr") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      lr = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--shrinkVal") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      shrinkVal = atof(argv[ai+1]);
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }

  srand(seed);

  DataProvider dpTrain;
  DataProvider dpValid;
  DataProvider dpTest;

  dpTrain.readFromFile(trainFile, V1, V2);
  dpValid.readFromFile(validFile, dpTrain);
  dpTest.readFromFile(testFile, dpTrain);

  // get the basic stats on the dataset
  int numWordsV0 = 0;
  int numWordsV1 = 0;
  int numWordsV2 = 0;
  dpTrain.getNumWords(numWordsV0, numWordsV1, numWordsV2);
  int numChars = dpTrain.getNumChars();
  printf("number of different characters: %d\n", numChars);
  printf("Vocabulary size... V0=%-10d V1=%-10d V2=%-10d\n",
      numWordsV0, numWordsV1, numWordsV2);

  double oovRateValid = dpValid.countOutOfVocab() * 100.0 / dpValid.getNumTokens();
  double oovRateTest = dpTest.countOutOfVocab() * 100.0 / dpTest.getNumTokens();
  printf("OOV rate on validation set: %5.2f\n", oovRateValid);
  printf("OOV rate on test set: %5.2f\n", oovRateTest);

  // dpTrain.printDictionary();

  // initializing the model parameters
  Model m(nhidw, nhidc, numWordsV1, numWordsV2, numChars, alpha);
  m.initialize(init);
  m.resetGradients();

  // obtaining the character table
  std::unordered_map<wchar_t, int> charTable = dpTrain.getCharTable();

  // creating a network
  Rnn network(m, charTable, dpTrain.int2char_, bptt, lr);

  double trainWordEntropy = 0.0;
  double validWordEntropy = 0.0;
  double testWordEntropy = 0.0;
  double trainCharEntropy = 0.0;
  double validCharEntropy = 0.0;
  double testCharEntropy = 0.0;
  double validLoss = 0.0;
  double prevValidLoss = DBL_MAX;
  int nTrainChars = 0;
  int nValidChars = 0;
  int nTestChars = 0;
  bool doShrink = false;

  double trainTime = 0.0;
  for (int e=0; e<nepoch; e++) {
    network.train(dpTrain, true, trainTime, trainWordEntropy, trainCharEntropy, nTrainChars);
    network.eval(dpValid, validWordEntropy, validCharEntropy, nValidChars);
    network.eval(dpTest, testWordEntropy, testCharEntropy, nTestChars);
    validLoss = alpha * validWordEntropy + (1.0 - alpha) * validCharEntropy;
    printf("json_stats: {");
    printf("\"alpha\": %f, ", alpha);
    printf("\"V1\": %d, ", V1);
    printf("\"V2\": %d, ", V2);
    printf("\"nValidChars\": %d, ", nValidChars);
    printf("\"nValidWords\": %d, ", dpValid.getNumTokens());
    printf("\"nTestChars\": %d, ", nTestChars);
    printf("\"nTestWords\": %d, ", dpTest.getNumTokens());
    printf("\"nhidc\": %d, ", nhidc);
    printf("\"nhidw\": %d, ", nhidw);
    printf("\"bptt\": %d, ", bptt);
    printf("\"seed\": %d, ", seed);
    printf("\"lr\": %f, ", lr);
    printf("\"shrinkVal\": %f, ", shrinkVal);
    printf("\"init\": \"%s\", ", init);
    printf("\"epoch\": %d, ", e);
    printf("\"train_time\": %f, ", trainTime);
    // logging train entropy
    printf("\"train_word_model_entropy\": %f, ",
        trainWordEntropy / dpTrain.getNumTokens());
    printf("\"train_word_entropy\": %f, ",
        trainCharEntropy / dpTrain.getNumTokens());
    printf("\"train_char_entropy\": %f, ", trainCharEntropy / nTrainChars);
    // logging valid entropy
    printf("\"valid_word_model_entropy\": %f, ",
        validWordEntropy / dpValid.getNumTokens());
    printf("\"valid_word_entropy\": %f, ",
        validCharEntropy / dpValid.getNumTokens());
    printf("\"valid_char_entropy\": %f, ", validCharEntropy / nValidChars);
    // logging test entropy
    printf("\"test_word_model_entropy\": %f, ",
        testWordEntropy / dpTest.getNumTokens());
    printf("\"test_word_entropy\": %f, ",
        testCharEntropy / dpTest.getNumTokens());
    printf("\"test_char_entropy\": %f", testCharEntropy / nTestChars);
    printf("}\n");

    // checking for increase of validation entropy
    if (doShrink || (1.001 * validLoss - prevValidLoss > 0)) {
      network.updateLearningRate(shrinkVal);
      printf("decreasing the learning rate to %f.", network.getLr());
      std::cout << std::endl;
      doShrink = true;
    }
    prevValidLoss = validLoss;
  }

  return 0;
}
