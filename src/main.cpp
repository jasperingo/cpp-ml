#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"
#include "KNNHandler.hpp"
#include "RandomForestHandler.hpp"
#include "DecisionTreeHandler.hpp"
#include "LinearRegressionHandler.hpp"
#include "LogisticRegressionHandler.hpp"

const std::string algorithmOption = "--algorithm";
const std::string datasetOption = "--dataset";
const std::string labelColumnOption = "--label-column";
const std::string labelColumnIsDigitOption = "--label-column-digit";
const std::string featureColumnsOption = "--feature-columns";
const std::string featureColumnsAreDigitOption = "--feature-columns-digit";

const std::string numberOfKOption = "--number-of-k";

const std::string learningRateOption = "--learning-rate";
const std::string maxNumberOfIterationsOption = "--max-iterations";

const std::string maxDepthOption = "--max-depth";
const std::string minSampleSplitOption = "--min-sample-split";
const std::string numberOfFeaturesOption = "--number-of-features";
const std::string numberOfTreesOption = "--number-of-trees";

std::tuple<std::string, bool, double> findArgument(
  char ** argsBegin, 
  char** argsEnd, 
  std::string option, 
  bool isDigit = false, 
  bool isRequired = true, 
  std::function<bool(double)> validator = nullptr,
  std::string validatorError = ""
) {
  std::string value;
  double valueDigit = 0.0;
  bool valueProvided = false;

  char ** itr = std::find(argsBegin, argsEnd, option);

  if (itr != argsEnd && ++itr != argsEnd) {
    value = *itr;
    valueProvided = true;
  }

  if (isDigit) {
    valueDigit = std::atof(value.c_str());
  }

  if (isRequired && !valueProvided) {
    throw std::runtime_error(option + " option not found");
  }

  if (valueProvided && validator != nullptr) {
    if (validator(valueDigit)) {
      throw std::runtime_error(validatorError);
    }
  }

  return std::tuple<std::string, bool, double>(value, valueProvided, valueDigit);
}

int main(int argc, char* argv[]) {
  char** argvEnd = argv + argc;

  try {
    std::tuple<std::string, bool, double> algorithmResult = findArgument(argv, argvEnd, algorithmOption);
    std::string algorithm = std::get<0>(algorithmResult);

    std::tuple<std::string, bool, double> datasetResult = findArgument(argv, argvEnd, datasetOption);
    std::string dataset = std::get<0>(datasetResult);

    std::tuple<std::string, bool, double> labelColumnResult = findArgument(argv, argvEnd, labelColumnOption, true);
    std::string labelColumn = std::get<0>(labelColumnResult);
    unsigned int labelColumnDigit = (unsigned int) std::get<2>(labelColumnResult);

    if (labelColumnDigit == 0 && labelColumn != "0") {
      throw std::runtime_error("None digit value provided for " + labelColumnOption);
    }

    std::tuple<std::string, bool, double> labelColumnIsDigitResult = findArgument(argv, argvEnd, labelColumnIsDigitOption);
    std::string labelColumnIsDigit = std::get<0>(labelColumnIsDigitResult);

    bool labelColumnIsDigitBool;

    if (labelColumnIsDigit == "1" || labelColumnIsDigit == "true") {
      labelColumnIsDigitBool = true;
    } else if (labelColumnIsDigit == "0" || labelColumnIsDigit == "false") {
      labelColumnIsDigitBool = false;
    } else {
      throw std::runtime_error("None boolean value provided for " + labelColumnOption);
    }

    std::tuple<std::string, bool, double> featureColumnsResult = findArgument(argv, argvEnd, featureColumnsOption);
    std::string featureColumns = std::get<0>(featureColumnsResult);

    std::tuple<std::string, bool, double> featureColumnsAreDigitResult = findArgument(argv, argvEnd, featureColumnsAreDigitOption);
    std::string featureColumnsAreDigit = std::get<0>(featureColumnsAreDigitResult);

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
      throw std::runtime_error(featureColumnsOption + " size: " + std::to_string(featureColumnsSize) 
        + " is not equal to " + featureColumnsAreDigitOption + " size: " + std::to_string(featureColumnsIsDigitSize));
    }

    std::vector<unsigned int> featureColumnsDigits;
    std::vector<bool> featureColumnsAreDigitBools;

    for (size_t i = 0; i < featureColumnsSize; i++) {
      std::string columnString = featureColumnsSplit[i];
      std::string columnIsDigitString = featureColumnsIsDigitSplit[i];

      double column = std::atof(columnString.c_str());

      if (column == 0 && columnString != "0") {
        throw std::runtime_error("None digit value provided for " + featureColumnsOption + " = " + columnString);
      } else {
        featureColumnsDigits.push_back((unsigned int) column);
      }

      if (columnIsDigitString == "1" || columnIsDigitString == "true") {
        featureColumnsAreDigitBools.push_back(true);
      } else if (columnIsDigitString == "0" || columnIsDigitString == "false") {
        featureColumnsAreDigitBools.push_back(false);
      } else {
        throw std::runtime_error("None boolean value provided for " + featureColumnsAreDigitOption + " = " + columnIsDigitString);
      }
    }

    CSVETL csvETL(dataset);

    if (!csvETL.load(labelColumnDigit, labelColumnIsDigitBool, featureColumnsDigits, featureColumnsAreDigitBools)) {
      throw std::runtime_error("Dataset: " + dataset + " not loaded ");
    }

    std::cout << "Dataset: " << dataset << " loaded " << std::endl;

    csvETL.printDatasetSize();

    if (algorithm == "knn") {

      std::tuple<std::string, bool, double> numberOfKResult = findArgument(
        argv, 
        argvEnd, 
        numberOfKOption, 
        true, 
        false, 
        [](double value) { return value <= 0; }, 
        numberOfKOption + " cannot be less than or equal to 0"
      );

      bool numberOfKProvided = std::get<1>(numberOfKResult);
      unsigned int numberOfKDigit = (unsigned int) std::get<2>(numberOfKResult);

      KNNConfig knnConfig(csvETL);

      if (numberOfKProvided) {
        knnConfig.numberOfK = numberOfKDigit;
      }

      handleKNN(knnConfig);

    } else if (algorithm == "linear-regression" || algorithm == "logistic-regression") {

      std::tuple<std::string, bool, double> learningRateResult = findArgument(
        argv, 
        argvEnd, 
        learningRateOption, 
        true, 
        false, 
        [](double value) { return value <= 0; }, 
        learningRateOption + " cannot be less than or equal to 0"
      );

      bool learningRateProvided = std::get<1>(learningRateResult);
      double learningRateDigit = std::get<2>(learningRateResult);

      std::tuple<std::string, bool, double> maxNumberOfIterationsResult = findArgument(
        argv, 
        argvEnd, 
        maxNumberOfIterationsOption, 
        true, 
        false, 
        [](double value) { return value < 10; }, 
        maxNumberOfIterationsOption + " cannot be less than 10"
      );

      bool maxNumberOfIterationsProvided = std::get<1>(maxNumberOfIterationsResult);
      int maxNumberOfIterationsDigit = (int) std::get<2>(maxNumberOfIterationsResult);

      if (algorithm == "linear-regression") {

        LinearRegressionConfig linearRegressionConfig(csvETL);

        if (learningRateProvided) {
          linearRegressionConfig.learningRate = learningRateDigit;
        }

        if (maxNumberOfIterationsProvided) {
          linearRegressionConfig.maxNumberOfIterations = maxNumberOfIterationsDigit;
        }

        handleLinearRegression(linearRegressionConfig);

      } else if (algorithm == "logistic-regression") {

        LogisticRegressionConfig logisticRegressionConfig(csvETL);

        if (learningRateProvided) {
          logisticRegressionConfig.learningRate = learningRateDigit;
        }

        if (maxNumberOfIterationsProvided) {
          logisticRegressionConfig.maxNumberOfIterations = maxNumberOfIterationsDigit;
        }

        handleLogisticRegression(logisticRegressionConfig);

      }

    } else if (algorithm == "decision-tree" || algorithm == "random-forest") {

      std::tuple<std::string, bool, double> maxDepthResult = findArgument(
        argv, 
        argvEnd, 
        maxDepthOption, 
        true, 
        false, 
        [](double value) { return value < 10; }, 
        maxDepthOption + " cannot be less than 10"
      );

      bool maxDepthProvided = std::get<1>(maxDepthResult);
      unsigned int maxDepthDigit = (unsigned int) std::get<2>(maxDepthResult);

      std::tuple<std::string, bool, double> minSampleSplitResult = findArgument(
        argv, 
        argvEnd, 
        minSampleSplitOption, 
        true, 
        false, 
        [](double value) { return value < 2; }, 
        minSampleSplitOption + " cannot be less than 2"
      );

      bool minSampleSplitProvided = std::get<1>(minSampleSplitResult);
      unsigned int minSampleSplitDigit = (unsigned int) std::get<2>(minSampleSplitResult);

      std::tuple<std::string, bool, double> numberOfFeaturesResult = findArgument(
        argv, 
        argvEnd, 
        numberOfFeaturesOption, 
        true, 
        false, 
        [featureColumnsSize](double value) { return value < 2 || value > featureColumnsSize; }, 
        numberOfFeaturesOption + " cannot be less than 2 nor greater than the number of features"
      );

      bool numberOfFeaturesProvided = std::get<1>(numberOfFeaturesResult);
      unsigned int numberOfFeaturesDigit = (unsigned int) std::get<2>(numberOfFeaturesResult);

      if (algorithm == "decision-tree") {
        DecisionTreeConfig decisionTreeConfig(csvETL);

        if (maxDepthProvided) {
          decisionTreeConfig.maxDepth = maxDepthDigit;
        }

        if (minSampleSplitProvided) {
          decisionTreeConfig.minSampleSplit = minSampleSplitDigit;
        }

        if (numberOfFeaturesProvided) {
          decisionTreeConfig.numberOfFeatures = numberOfFeaturesDigit;
        }
        
        handleDecisionTree(decisionTreeConfig);
        
      } else {
        
        std::tuple<std::string, bool, double> numberOfTreesResult = findArgument(
          argv, 
          argvEnd, 
          numberOfTreesOption, 
          true, 
          false, 
          [](double value) { return value < 5; }, 
          numberOfTreesOption + " cannot be less than 5"
        );

        bool numberOfTreesProvided = std::get<1>(numberOfTreesResult);
        unsigned int numberOfTreesDigit = (unsigned int) std::get<2>(numberOfTreesResult);
        
        RandomForestConfig randomForestConfig(csvETL);
        
        if (maxDepthProvided) {
          randomForestConfig.maxDepth = maxDepthDigit;
        }

        if (minSampleSplitProvided) {
          randomForestConfig.minSampleSplit = minSampleSplitDigit;
        }

        if (numberOfFeaturesProvided) {
          randomForestConfig.numberOfFeatures = numberOfFeaturesDigit;
        }

        if (numberOfTreesProvided) {
          randomForestConfig.numberOfTrees = numberOfTreesDigit;
        }

        handleRandomForest(randomForestConfig);
      }

    } else {
      throw std::runtime_error("Provided algorithm: " + algorithm + " has no implementation");
    }

  } catch (std::runtime_error& error) {
    std::cout << error.what() << std::endl;
    return 1;
  }

  return 0;
}
