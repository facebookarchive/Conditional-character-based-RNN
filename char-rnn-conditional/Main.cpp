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
#include <time.h>
#include <string.h>
#include <float.h>
#include <math.h>

bool VERBOSE = true;
bool USE_BLAS = false;

std::string getHash(int argc, char** argv) {
  std::hash<std::string> str_hash;
  std::string modelString;
  for (int i=0; i<argc; i++) {
    modelString += argv[i];
  }
  time_t rawtime;
  time(&rawtime);
  std::string hash1 = std::to_string(str_hash(ctime(&rawtime)));
  std::string hash2 = std::to_string(str_hash(modelString));
  return hash1 + hash2;
}

int main(int argc, char** argv) {
  int nhid = 100;
  int bptt = 30;
  int ngram = 30;
  int minFreq = 40;
  int nepoch = 10;
  double lr = 0.1;
  double shrinkVal = 2.0;
  std::string trainFile;
  std::string validFile;
  std::string testFile;
  char init[100];
  strcpy(init, "gaussian");

  std::string modelHash = getHash(argc, argv);

  int ai = 1;
  while(ai < argc) {
    if( strcmp( argv[ai], "--nhid") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n", argv[ai]);
        return - 1;
      }
      nhid = atoi(argv[ai+1]);
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
    else if( strcmp( argv[ai], "--bptt") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      bptt = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--ngram") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      ngram = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--minFreq") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      minFreq = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "--nepoch") == 0){
      if (ai + 1 >= argc) {
        printf("error need argument for option %s\n",argv[ai]);
        return - 1;
      }
      nepoch = atoi(argv[ai+1]);
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

  if (trainFile.size()==0 || validFile.size()==0 || testFile.size()==0) {
    fprintf(stderr, "Pleas provide training, validation and test files!\n");
    return -1;
  }

  DataProvider dp_train(ngram, minFreq);
  DataProvider dp_valid(ngram, minFreq);
  DataProvider dp_test(ngram, minFreq);

  dp_train.readFromFile(trainFile);
  dp_valid.readFromFile(validFile, dp_train);
  dp_test.readFromFile(testFile, dp_train);


  int numChars = dp_train.getNumChars();
  int numWords = dp_train.getNumWords();
  int numTrainTokens = dp_train.getNumTokens();
  int numValidTokens = dp_valid.getNumTokens();
  int numTestTokens = dp_test.getNumTokens();

  // initializing the model parameters
  Model model(nhid, numChars);
  model.initialize(init);
  model.resetGradients();

  // creating a network
  Rnn network(model, bptt, lr);

  double train_time = 0.0;
  double train_entropy = 0.0;
  double valid_entropy = 0.0;
  double prev_valid_entropy = DBL_MAX;
  double test_entropy = 0.0;
  double entropy_thresh = 0.001;
  bool doShrink = false;
  for (int e=0; e<nepoch; e++) {
    train_entropy = network.train(dp_train, train_time);
    valid_entropy = network.eval(dp_valid);
    test_entropy = network.eval(dp_test);

    printf("json_stats: {");
    printf("\"nhid\": %d, ", nhid);
    printf("\"hash\": %s, ", modelHash.c_str());
    printf("\"bptt\": %d, ", bptt);
    printf("\"ngram\": %d, ", ngram);
    printf("\"minFreq\": %d, ", minFreq);
    printf("\"lr\": %f, ", lr);
    printf("\"shrinkVal\": %f, ", shrinkVal);
    printf("\"epoch\": %d, ", e);
    printf("\"train_time\": %f, ", train_time);
    printf("\"train_char_entropy\": %f, ", train_entropy);
    printf("\"train_logprob\": %f, ",
        train_entropy * numTrainTokens * log(2.0) / log(10.0));
    printf("\"valid_char_entropy\": %f, ", valid_entropy);
    printf("\"valid_logprob\": %f, ",
        valid_entropy * numValidTokens * log(2.0) / log(10.0));
    printf("\"test_char_entropy\": %f, ", test_entropy);
    printf("\"test_logprob\": %f",
        test_entropy * numTestTokens * log(2.0) / log(10.0));
    printf("}\n");

    // checking for increase of validation entropy
    if (doShrink || 1.001 * valid_entropy - prev_valid_entropy > 0) {
      network.updateLearningRate(shrinkVal);
      printf("decreasing the learning rate to %f.\n", network.getLr());
      doShrink = true;
    }
    prev_valid_entropy = valid_entropy;
  }


  return 0;
}
