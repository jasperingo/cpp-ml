#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"
#include "Algorithms.hpp"

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "All options not provided" << std::endl;
    return 1;
  }

  // doKNN(argv[1]);
  
  std::cout << "CMD args count: " << argc << std::endl;

  std::string algorithmOption = "--algorithm";
  std::string datasetOption = "--dataset";
  std::string labelColumnOption = "--label-column";
  std::string labelColumnIsDigitOption = "--label-column-digit";
  std::string featureColumnsOption = "--feature-columns";
  std::string featureColumnsAreDigitOption = "--feature-columns-digit";

  std::string algorithm;
  std::string dataset;
  std::string labelColumn;
  std::string labelColumnIsDigit;
  std::string featureColumns;
  std::string featureColumnsAreDigit;

  char ** itr;

  char** end = argv + argc;

  itr = std::find(argv, end, algorithmOption);

  if (itr != end && ++itr != end) {
    algorithm = *itr;
  } else {
    std::cout << algorithmOption << " option not found" << std::endl;
    return 1;
  }

  itr = std::find(argv, end, datasetOption);

  if (itr != end && ++itr != end) {
    dataset = *itr;
  } else {
    std::cout << datasetOption << " option not found" << std::endl;
    return 1;
  }

  itr = std::find(argv, end, labelColumnOption);

  if (itr != end && ++itr != end) {
    labelColumn = *itr;
  } else {
    std::cout << labelColumnOption << " option not found" << std::endl;
    return 1;
  }

  itr = std::find(argv, end, labelColumnIsDigitOption);

  if (itr != end && ++itr != end) {
    labelColumnIsDigit = *itr;
  } else {
    std::cout << labelColumnIsDigitOption << " option not found" << std::endl;
    return 1;
  }

  itr = std::find(argv, end, featureColumnsOption);

  if (itr != end && ++itr != end) {
    featureColumns = *itr;
  } else {
    std::cout << featureColumnsOption << " option not found" << std::endl;
    return 1;
  }

  itr = std::find(argv, end, featureColumnsAreDigitOption);

  if (itr != end && ++itr != end) {
    featureColumnsAreDigit = *itr;
  } else {
    std::cout << featureColumnsAreDigitOption << " option not found" << std::endl;
    return 1;
  }

  std::string optionPart;
  std::vector<std::string> featureColumnsSplit;
  std::stringstream featureColumnsStream(featureColumns);

  while(std::getline(featureColumnsStream, optionPart, ',')) {
    featureColumnsSplit.push_back(optionPart);
  }

  std::vector<std::string> featureColumnsIsDigitSplit;
  std::stringstream featureColumnsIsDigitStream(featureColumnsAreDigit);

  while(std::getline(featureColumnsIsDigitStream, optionPart, ',')) {
    featureColumnsIsDigitSplit.push_back(optionPart);
  }

  const size_t featureColumnsSize = featureColumnsSplit.size();
  const size_t featureColumnsIsDigitSize = featureColumnsIsDigitSplit.size();

  if (featureColumnsSize != featureColumnsIsDigitSize) {
    std::cout << featureColumnsOption << " size: " << featureColumnsSize 
      << " is not equal to " << featureColumnsAreDigitOption << " size: " << featureColumnsIsDigitSize << std::endl;
    return 1;
  }

  unsigned int featureColumnsDigits[100];
  bool featureColumnsAreDigitBools[100];

  for (size_t i = 0; i < featureColumnsSize; i++) {
    std::string columnString = featureColumnsSplit[i];
    std::string columnIsDigitString = featureColumnsIsDigitSplit[i];

    double column = std::atof(columnString.c_str());

    if (column == 0 && columnString != "0") {
      std::cout << "None digit value provided for " << featureColumnsOption << " = " << columnString << std::endl;
      return 1;
    } else {
      featureColumnsDigits[i] = (unsigned int) column;
    }

    if (columnIsDigitString == "1" || columnIsDigitString == "true") {
      featureColumnsAreDigitBools[i] = true;
    } else if (columnIsDigitString == "0" || columnIsDigitString == "false") {
      featureColumnsAreDigitBools[i] = false;
    } else {
      std::cout << "None boolean value provided for " << featureColumnsAreDigitOption << " = " << columnIsDigitString << std::endl;
      return 1;
    }
  }

  bool labelColumnIsDigitBool;

  double labelColumnDigit = std::atof(labelColumn.c_str());

  if (labelColumnDigit == 0 && labelColumn != "0") {
    std::cout << "None digit value provided for " << labelColumnOption << std::endl;
    return 1;
  }

  if (labelColumnIsDigit == "1" || labelColumnIsDigit == "true") {
    labelColumnIsDigitBool = true;
  } else if (labelColumnIsDigit == "0" || labelColumnIsDigit == "false") {
    labelColumnIsDigitBool = false;
  } else {
    std::cout << "None boolean value provided for " << labelColumnOption << std::endl;
    return 1;
  }

  std::cout << "Done " << std::endl;
}
