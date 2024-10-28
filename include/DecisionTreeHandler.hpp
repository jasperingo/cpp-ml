#ifndef CPP_ML_DECISION_TREE_HANDLER_H_
#define CPP_ML_DECISION_TREE_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "CSVETL.hpp"
#include "MLUtils.hpp"
#include "DecisionTree.hpp"

struct DecisionTreeConfig {
  CSVETL& etl;
  unsigned int maxDepth = 100;
  unsigned int minSampleSplit = 2;
  unsigned int numberOfFeatures = 0;

  DecisionTreeConfig(CSVETL& etl): etl(etl) {}
};

void handleDecisionTree(DecisionTreeConfig& config) {

  Eigen::MatrixXd X = config.etl.getFeatures();

  Eigen::VectorXd y = config.etl.getLabels();

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;
  std::cout << "Number of labels: " << y.size() << std::endl;

  // std::cout << "Features:" << std::endl << std::endl;

  // std::cout << X << std::endl << std::endl;

  // std::cout << "Labels:" << std::endl << std::endl;

  // std::cout << y << std::endl << std::endl;

  CSVETL::DataSplitResult splitResult = config.etl.splitData(y, X);

  std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  DecisionTree decisionTree(config.maxDepth, config.minSampleSplit, config.numberOfFeatures);

  decisionTree.fit(splitResult.trainSamples, splitResult.trainLabels);

  Eigen::VectorXd predictions = decisionTree.predict(splitResult.testSamples);

  MLUtils::calculateAndPrintAccuracy(predictions, splitResult.testLabels);
}

#endif /* CPP_ML_DECISION_TREE_HANDLER_H_ */
