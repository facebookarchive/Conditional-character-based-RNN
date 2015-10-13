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

DataProvider::DataProvider(int ngramOrder, int minFreq) {
  currIdx_ = tokens_.begin();
  nWords_ = 1;
  ngramOrder_ = ngramOrder;
  minFreq_ = minFreq;
}

DataProvider::~DataProvider() {
}

int DataProvider::getNumTokens() {
  return tokens_.size();
}

int DataProvider::getNumWords() {
  return nWords_;
}

int DataProvider::getNumChars() {
  return char2int_.size();
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

void DataProvider::printDictionary() {
  for (auto it = char2int_.begin(); it != char2int_.end(); ++it) {
    std::cout << it->first << " -> " << it->second << std::endl;
  }
}

void DataProvider::printTokens() {
  for (auto it = tokens_.begin(); it != tokens_.end(); ++it) {
    std::cout << *it << std::endl;
  }
}

wchar_t DataProvider::getChar(int x) {
  return int2char_[x];
}

void DataProvider::initIterator() {
  currIdx_ = tokens_.begin();
}

void DataProvider::getToken(int& now, int& next) {
  now = *currIdx_;

  history_.push_back(int2char_[now]);
  if (history_.size() > ngramOrder_) {
    history_.erase(0, 1);
  }

  // print ngram for debugging
  // printf("now: %4d ", now);
  // for (auto it=history_.begin(); it!=history_.end(); ++it) {
  //   printf("%4d -> ", *it);
  // }
  // printf("\n");

  currIdx_++;
  if (currIdx_ == tokens_.end()) {
    currIdx_ = tokens_.begin();
  }
  next = *currIdx_;
}

std::wstring DataProvider::getHistory() {
  std::wstring res = history_;
  while (res.length()>0) {
    if (validNgrams_.count(res)==0) {
      res.erase(0, 1);
    } else {
      break;
    }
  }
  // std::wcout << res << std::endl;
  return res;
}

void DataProvider::readFromFile(std::string fname) {
  std::locale::global(std::locale(""));
  std::wifstream ifs(fname);
  wchar_t c;
  int k = 0;

  std::unordered_map<std::wstring, int> ngramCount;

  std::wstring ngramHistory;

  while (ifs.get(c)) {
    if (c == '_') {
      nWords_++;
    }
    if (char2int_.count(c)==0) {
      char2int_.insert({c, k});
      int2char_.insert({k, c});
      tokens_.push_back(k);
      k++;
    } else {
      int idx = char2int_[c];
      tokens_.push_back(idx);
    }

    // updating the ngram history
    ngramHistory.push_back(c);
    if (ngramHistory.length() > ngramOrder_) {
      ngramHistory.erase(0, 1);
    }
    if (ngramHistory.length() == ngramOrder_) {
      for (int j=0; j<ngramOrder_; j++) {

        std::wstring temp = ngramHistory.substr(j, ngramOrder_-j);
        if (ngramCount.count(temp) == 0) {
          ngramCount.insert({temp, 1});
        } else {
          ngramCount[temp]++;
        }
      }
    }
  }

  for (auto it=ngramCount.begin(); it!=ngramCount.end(); ++it) {
    if (it->first.size()==1 || it->second > minFreq_) {
      validNgrams_.insert(it->first);
    }
  }
  std::cout << validNgrams_.size() << std::endl;
  ifs.close();
}

void DataProvider::readFromFile(std::string fname, DataProvider& train) {
  char2int_ = train.char2int_;
  int2char_ = train.int2char_;
  validNgrams_ = train.validNgrams_;

  std::wifstream ifs(fname);
  wchar_t c;
  int k = 0;
  while (ifs.get(c)) {
    if (char2int_.count(c)==0) {
      std::cout << "oh!" << std::endl;
    } else {
      int idx = char2int_[c];
      tokens_.push_back(idx);
    }
  }
  ifs.close();
}
