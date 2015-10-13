/*
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <set>
#include <list>
#include <fstream>
#include "Utils.h"

class DataProvider {
  private:
    std::unordered_map<std::string, int> word2int_;
    std::unordered_map<int, std::string> int2word_;
    std::unordered_map<int, int> wordCount_;

    std::unordered_map<int, int> restrictedVocab1_;
    std::unordered_map<int, int> restrictedVocab2_;
    int V1_;
    int V2_;

    std::unordered_set<int> outOfVocabulary_;

    std::list<int> tokens_;
    int lastIdx_;
    std::list<int>::iterator currIdx_;
  public:
    std::unordered_map<wchar_t, int> char2int_;
    std::unordered_map<int, wchar_t> int2char_;

    DataProvider();
    ~DataProvider();
    int getNumTokens();
    void getNumWords(int&, int&, int&);
    int getNumChars();
    int countOutOfVocab();
    int getWordId(std::string);
    int addWord(std::string, bool);
    void printDictionary();
    void printTokens();
    void initIterator();
    void getToken(int&, int&, std::string&);
    std::string getWord(int);
    void printCharTable();
    void computeRestrictedVocabs(int, int);
    void readFromFile(std::string, int, int);
    void readFromFile(std::string, DataProvider&);
    std::unordered_map<wchar_t, int>& getCharTable();
    void randomWord(int&, std::string&);
};

#endif
