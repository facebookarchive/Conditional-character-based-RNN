/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "DataProvider.h"
#include <iostream>
#include <stdio.h>
#include <array>
#include <algorithm>

DataProvider::DataProvider() {
  word2int_.insert({"<unk>", 0});
  int2word_.insert({0, "<unk>"});
  lastIdx_ = 1;
  currIdx_ = tokens_.begin();
}

DataProvider::~DataProvider() {
}

int DataProvider::addWord(std::string s, bool isTrain) {
  int count = word2int_.count(s);
  if (count == 0) {
    word2int_.insert({s, lastIdx_});
    int2word_.insert({lastIdx_, s});
    tokens_.push_back(lastIdx_);
    wordCount_.insert({lastIdx_, 1});
    if (!isTrain) {
      outOfVocabulary_.insert(lastIdx_);
      restrictedVocab1_.insert({lastIdx_, 0});
      restrictedVocab2_.insert({lastIdx_, 0});
    }
    lastIdx_++;
    return lastIdx_-1;
  } else {
    int idx = word2int_[s];
    wordCount_[idx] ++;
    tokens_.push_back(idx);
    return idx;
  }
}

int DataProvider::getNumTokens() {
  return tokens_.size();
}

// return the size of the original vocabulary
void DataProvider::getNumWords(int& numWordsV0,
                                int& numWordsV1,
                                int& numWordsV2) {
  numWordsV0 = word2int_.size();
  numWordsV1 = V1_;
  numWordsV2 = V2_;
}

// return size of char vocabulary
int DataProvider::getNumChars() {
  return char2int_.size();
}

// counting how many tokens are OOV
int DataProvider::countOutOfVocab() {
  int nOOV = 0;
  for (auto it=tokens_.begin(); it!=tokens_.end(); ++it) {
    if (outOfVocabulary_.find(*it)!=outOfVocabulary_.end()) {
      nOOV ++;
    }
  }
  return nOOV;
}

// return the word id in the input vocabulary V1
// if not present, provide the id of <unk>
int DataProvider::getWordId(std::string str) {
  if (word2int_.count(str) == 0) {
    return 0;
  } else {
    int idx = word2int_[str];
    return restrictedVocab1_[idx];
  }
}

void DataProvider::printDictionary() {
  int k = 0;
  for (auto it = int2word_.begin(); it != int2word_.end(); ++it) {
    if (restrictedVocab2_[it->first] > 0) {
      printf("%-50s\t\t%8d ", it->second.c_str(),  wordCount_[it->first]);
      printf("V0:%8d ", it->first);
      printf("V1:%8d ", restrictedVocab1_[it->first]);
      printf("V2:%8d ", restrictedVocab2_[it->first]);
      printf("\n");
    }
  }
}

void DataProvider::printTokens() {
  for (auto it = tokens_.begin(); it != tokens_.end(); ++it) {
    std::cout << *it << std::endl;
  }
}

void DataProvider::initIterator() {
  currIdx_ = tokens_.begin();
}

void DataProvider::getToken(int& now, int& next, std::string& nextWord) {
  now = *currIdx_;
  currIdx_++;
  if (currIdx_ == tokens_.end()) {
    currIdx_ = tokens_.begin();
  }
  next = *currIdx_;
  nextWord = getWord(next);

  // remapping to the restricted vocabs
  now = restrictedVocab1_[now];
  next = restrictedVocab2_[next];
}

std::string DataProvider::getWord(int i) {
  return int2word_[i];
}

void DataProvider::printCharTable() {
  char mbs[16];
  int nBytes;
  for (auto it = char2int_.begin(); it != char2int_.end(); ++it) {
    nBytes = wctomb(mbs, it->first);
    printf("%lc : %x : %d : ", it->first, it->first, it->second);
    for (int i=0; i<nBytes; i++) {
      printf("%x ", 0xff & mbs[i]);
    }
    printf("\n");
  }
}

void DataProvider::computeRestrictedVocabs(int V1, int V2) {

  // sorting the word occurences
  std::vector<int> counts;
  for (auto it=wordCount_.begin(); it!=wordCount_.end(); ++it) {
    counts.push_back(it->second);
  }
  std::sort(counts.begin(), counts.end(), [](int a, int b) {
      return b < a;
  });

  int k1 = 0;
  int k2 = 0;
  for (auto it=wordCount_.begin(); it!=wordCount_.end(); ++it) {
    if (it->second <= V1) {
      restrictedVocab1_.insert({it->first, 0});
      restrictedVocab2_.insert({it->first, 0});
    } else if (it->second < counts[V2]) {
      k1++;
      restrictedVocab1_.insert({it->first, k1});
      restrictedVocab2_.insert({it->first, 0});
    } else {
      k1++;
      k2++;
      restrictedVocab1_.insert({it->first, k1});
      restrictedVocab2_.insert({it->first, k2});
    }
    // std::cout << int2word_[it->first] << " " << it->second;
    // std::cout << " V1=" << restrictedVocab1_[it->first];
    // std::cout << " V2=" << restrictedVocab2_[it->first] << std::endl;
  }
  V1_ = k1;
  V2_ = k2;
}

void DataProvider::readFromFile(std::string fname, int V1, int V2) {
  std::cout << "Loading data from file: " << fname;
  std::locale::global(std::locale(""));
  std::wifstream ifs(fname);
  wchar_t c;
  char mbs[16];
  int nBytes;
  int maxWordLength = 0;

  char2int_.insert({'_', 0});
  int2char_.insert({0, '_'});
  int k = 1;
  std::string str;
  while (ifs.get(c)) {
    if (c == '_') {
      addWord(str, true);
      if (str.length() > maxWordLength) {
        maxWordLength = str.length();
      }
      str.clear();
    } else {
      if (char2int_.count(c)==0) {
        char2int_.insert({c, k});
        int2char_.insert({k, c});
        k++;
      }
      nBytes = wctomb(mbs, c);
      for (int i=0; i<nBytes; i++) {
        str.push_back(mbs[i]);
      }
    }
  }
  ifs.close();

  computeRestrictedVocabs(V1, V2);

  std::cout << " done." << std::endl;
}

void DataProvider::readFromFile(std::string fname, DataProvider& train) {
  std::cout << "Loading data from file: " << fname << std::endl;
  word2int_ = train.word2int_;
  int2word_ = train.int2word_;
  wordCount_ = train.wordCount_;
  char2int_ = train.char2int_;
  int2char_ = train.int2char_;
  restrictedVocab1_ = train.restrictedVocab1_;
  restrictedVocab2_ = train.restrictedVocab2_;
  V1_ = train.V1_;
  V2_ = train.V2_;
  lastIdx_ = train.lastIdx_;

  std::wifstream ifs(fname);
  wchar_t c;
  char mbs[16];
  int nBytes = 0;

  int k = 0;
  std::string str;
  while (ifs.get(c)) {
    if (c == '_') {
      addWord(str, false);
      str.clear();
    } else {
      nBytes = wctomb(mbs, c);
      if (char2int_.count(c)==0) {
        printf("Unseen char! %lc : %x : ", c, c);
        for (int i=0; i<nBytes; i++) {
          printf("%x ", 0xff & mbs[i]);
        }
        printf("\n");
      } else {
        for (int i=0; i<nBytes; i++) {
          str.push_back(mbs[i]);
        }
      }
    }
  }
  ifs.close();
  std::cout << "Done." << std::endl;
}

std::unordered_map<wchar_t, int>& DataProvider::getCharTable() {
  return char2int_;
}

void DataProvider::randomWord(int& wordId, std::string& word) {
  int N = int2word_.size();
  int id = intRand(0, N);
  wordId = restrictedVocab1_[id];
  word = getWord(id);
}
