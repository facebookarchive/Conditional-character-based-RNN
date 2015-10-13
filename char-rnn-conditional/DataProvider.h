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
#include <string>
#include <unordered_set>
#include <list>
#include <fstream>

class DataProvider {
  private:
    std::unordered_map<wchar_t, int> char2int_;
    std::unordered_map<int, wchar_t> int2char_;
    std::list<int> tokens_;
    std::list<int>::iterator currIdx_;
    int nWords_;
    std::wstring history_;
    std::unordered_set<std::wstring> validNgrams_;

  public:
    int ngramOrder_;
    int minFreq_;
    DataProvider(int, int);
    ~DataProvider();
    int getNumTokens();
    int getNumWords();
    int getNumChars();
    int addChar(wchar_t);
    wchar_t getChar(int);
    void printCharTable();
    void printDictionary();
    void printTokens();
    void initIterator();
    void getToken(int&, int&);
    std::wstring getHistory();
    void readFromFile(std::string);
    void readFromFile(std::string, DataProvider&);
};

#endif
