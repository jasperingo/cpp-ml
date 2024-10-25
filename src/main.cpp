#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"
#include "KNNHandler.hpp"
#include "DecisionTreeHandler.hpp"
#include "LinearRegressionHandler.hpp"
#include "LogisticRegressionHandler.hpp"

int main(int argc, char* argv[]) {
  if (argc < 13) {
    std::cout << "All options not provided" << std::endl;
    return 1;
  }

  std::string algorithmOption = "--algorithm";
  std::string datasetOption = "--dataset";
  std::string labelColumnOption = "--label-column";
  std::string labelColumnIsDigitOption = "--label-column-digit";
  std::string featureColumnsOption = "--feature-columns";
  std::string featureColumnsAreDigitOption = "--feature-columns-digit";

  std::string learningRateOption = "--learning-rate";
  std::string maxNumberOfIterationsOption = "--max-iterations";

  std::string maxDepthOption = "--max-depth";
  std::string minSampleSplitOption = "--min-sample-split";

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

  std::vector<unsigned int> featureColumnsDigits;
  std::vector<bool> featureColumnsAreDigitBools;

  for (size_t i = 0; i < featureColumnsSize; i++) {
    std::string columnString = featureColumnsSplit[i];
    std::string columnIsDigitString = featureColumnsIsDigitSplit[i];

    double column = std::atof(columnString.c_str());

    if (column == 0 && columnString != "0") {
      std::cout << "None digit value provided for " << featureColumnsOption << " = " << columnString << std::endl;
      return 1;
    } else {
      featureColumnsDigits.push_back((unsigned int) column);
    }

    if (columnIsDigitString == "1" || columnIsDigitString == "true") {
      featureColumnsAreDigitBools.push_back(true);
    } else if (columnIsDigitString == "0" || columnIsDigitString == "false") {
      featureColumnsAreDigitBools.push_back(false);
    } else {
      std::cout << "None boolean value provided for " << featureColumnsAreDigitOption << " = " << columnIsDigitString << std::endl;
      return 1;
    }
  }

  bool labelColumnIsDigitBool;

  unsigned int labelColumnDigit = (unsigned int) std::atof(labelColumn.c_str());

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

  if (algorithm == "knn") {
    return handleKNN(dataset, labelColumnDigit, labelColumnIsDigitBool, featureColumnsDigits, featureColumnsAreDigitBools);
  } else if (algorithm == "linear-regression" || algorithm == "logistic-regression") {
    
    bool learningRateProvided = false;
    bool maxNumberOfIterationsProvided = false;

    std::string learningRate;
    std::string maxNumberOfIterations;

    double learningRateDigit;
    int maxNumberOfIterationsDigit;

    itr = std::find(argv, end, learningRateOption);

    if (itr != end && ++itr != end) {
      learningRate = *itr;
      learningRateProvided = true;
      learningRateDigit = std::atof(learningRate.c_str());

      if (learningRateDigit <= 0) {
        std::cout << learningRateOption << " cannot be less than 0.1" << std::endl;
        return 1;
      }
    }

    itr = std::find(argv, end, maxNumberOfIterationsOption);

    if (itr != end && ++itr != end) {
      maxNumberOfIterations = *itr;
      maxNumberOfIterationsProvided = true;

      maxNumberOfIterationsDigit = (int) std::atof(maxNumberOfIterations.c_str());

      if (maxNumberOfIterationsDigit < 10) {
        std::cout << maxNumberOfIterationsOption << " cannot be less than 10" << std::endl;
        return 1;
      }
    }

    if (algorithm == "linear-regression") {
      if (learningRateProvided && maxNumberOfIterationsProvided) {
        return handleLinearRegression(
          dataset, 
          labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          learningRateDigit, 
          maxNumberOfIterationsDigit
        );
      }

      if (learningRateProvided) {
        return handleLinearRegression(
          dataset, labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          learningRateDigit
          );
      }

      if (maxNumberOfIterationsProvided) {
        return handleLinearRegression(
          dataset, 
          labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          maxNumberOfIterationsDigit
        );
      }

      return handleLinearRegression(dataset, labelColumnDigit, labelColumnIsDigitBool, featureColumnsDigits, featureColumnsAreDigitBools);
    } else if (algorithm == "logistic-regression") {
      if (learningRateProvided && maxNumberOfIterationsProvided) {
        return handleLogisticRegression(
          dataset, 
          labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          learningRateDigit, 
          maxNumberOfIterationsDigit
        );
      }

      if (learningRateProvided) {
        return handleLogisticRegression(
          dataset, labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          learningRateDigit
          );
      }

      if (maxNumberOfIterationsProvided) {
        return handleLogisticRegression(
          dataset, 
          labelColumnDigit, 
          labelColumnIsDigitBool, 
          featureColumnsDigits, 
          featureColumnsAreDigitBools, 
          maxNumberOfIterationsDigit
        );
      }

      return handleLogisticRegression(dataset, labelColumnDigit, labelColumnIsDigitBool, featureColumnsDigits, featureColumnsAreDigitBools);
    }
  } else if (algorithm == "decision-tree") {
  
    bool maxDepthProvided = false;
    bool minSampleSplitProvided = false;

    std::string maxDepth;
    std::string minSampleSplit;

    unsigned int maxDepthDigit;
    unsigned int minSampleSplitDigit;

    itr = std::find(argv, end, maxDepthOption);

    if (itr != end && ++itr != end) {
      maxDepth = *itr;
      maxDepthProvided = true;
      maxDepthDigit = (unsigned int) std::atof(maxDepth.c_str());

      if (maxDepthDigit < 10) {
        std::cout << maxDepthOption << " cannot be less than 10" << std::endl;
        return 1;
      }
    }

    itr = std::find(argv, end, minSampleSplitOption);

    if (itr != end && ++itr != end) {
      minSampleSplit = *itr;
      minSampleSplitProvided = true;
      minSampleSplitDigit = (unsigned int) std::atof(minSampleSplit.c_str());

      if (minSampleSplitDigit < 2) {
        std::cout << minSampleSplitOption << " cannot be less than 2" << std::endl;
        return 1;
      }
    }

    if (maxDepthProvided && minSampleSplitProvided) {
      return handleDecisionTree(
        dataset, 
        labelColumnDigit, 
        labelColumnIsDigitBool, 
        featureColumnsDigits, 
        featureColumnsAreDigitBools, 
        maxDepthDigit, 
        minSampleSplitDigit
      );
    }

    if (maxDepthProvided) {
      return handleDecisionTreeWithMaxDepth(
        dataset, 
        labelColumnDigit, 
        labelColumnIsDigitBool, 
        featureColumnsDigits, 
        featureColumnsAreDigitBools, 
        maxDepthDigit
      );
    }

    if (minSampleSplitProvided) {
      return handleDecisionTreeWithMinSampleSplit(
        dataset, 
        labelColumnDigit, 
        labelColumnIsDigitBool, 
        featureColumnsDigits, 
        featureColumnsAreDigitBools,
        minSampleSplitDigit
      );
    }
    
    return handleDecisionTree(dataset, labelColumnDigit, labelColumnIsDigitBool, featureColumnsDigits, featureColumnsAreDigitBools);
  }

  std::cout << "Provided algorithm: " << algorithm << " has no implementation" << std::endl;
  return 1;
}
