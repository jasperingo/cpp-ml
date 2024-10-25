#ifndef CPP_ML_DECISION_TREE_HANDLER_H_
#define CPP_ML_DECISION_TREE_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "CSVETL.hpp"
#include "DecisionTree.hpp"

int handleDecisionTree(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  unsigned int maxDepth,
  unsigned int minSampleSplit
) {
  CSVETL csvETL(datasetPath);

  if (!csvETL.load(labelColumn, labelColumIsDigit, featureColumns, featureColumnsAreDigits)) {
    std::cout << "Dataset: " << datasetPath << " not loaded " << std::endl;
    return 1;
  }

  std::cout << "Dataset: " << datasetPath << " loaded " << std::endl;

  Eigen::MatrixXd X = csvETL.getFeatures();

  Eigen::VectorXd y = csvETL.getLabels();

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;
  std::cout << "Number of labels: " << y.size() << std::endl;

  // std::cout << "Features:" << std::endl << std::endl;

  // std::cout << X << std::endl << std::endl;

  // std::cout << "Labels:" << std::endl << std::endl;

  // std::cout << y << std::endl << std::endl;

  CSVETL::DataSplitResult splitResult = csvETL.splitData(y, X);

  std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  unsigned int noOfFeatures = X.cols() <= 2 ? X.cols() : X.cols() - 1;

  DecisionTree decisionTree(maxDepth, minSampleSplit, noOfFeatures);

  decisionTree.fit(splitResult.trainSamples, splitResult.trainLabels);

  Eigen::VectorXd predictions = decisionTree.predict(splitResult.testSamples);

  int correctCount = 0;
  int incorrectCount = 0;

  for (int i = 0; i < predictions.size(); i++) {
    double prediction = predictions(i);
    double testLabel = splitResult.testLabels(i);

    if (prediction == testLabel) {
      correctCount++;
    } else {
      incorrectCount++;
    }
  }

  // std::cout << "Predictions: " << predictionX << std::endl;

  double accuracy = ((double) correctCount) / splitResult.testLabels.size();

  std::cout << "Model accuracy: " << accuracy << std::endl;
  std::cout << "Model accuracy %: " << (accuracy * 100) << std::endl;
  std::cout << "Number of correct predictions: " << correctCount << std::endl;
  std::cout << "Number of incorrect predictions: " << incorrectCount << std::endl;

  return 0;
}

int handleDecisionTree(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits
) {
  return handleDecisionTree(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    100,
    2
  );
}

int handleDecisionTreeWithMaxDepth(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  unsigned int maxDepth
) {
  return handleDecisionTree(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits, 
    maxDepth,
    2
  );
}

int handleDecisionTreeWithMinSampleSplit(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  unsigned int minSampleSplit
) {
  return handleDecisionTree(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits, 
    100,
    minSampleSplit
  );
}

#endif /* CPP_ML_DECISION_TREE_HANDLER_H_ */
