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
  CSVETL::DataSplitResult splitResult = config.etl.splitData();

  config.etl.printDatasetSplitSize(splitResult);

  DecisionTree decisionTree(config.maxDepth, config.minSampleSplit, config.numberOfFeatures);

  decisionTree.fit(splitResult.trainSamples, splitResult.trainLabels);

  Eigen::VectorXd predictions = decisionTree.predict(splitResult.testSamples);

  MLUtils::calculateAndPrintAccuracy(predictions, splitResult.testLabels);
}

#endif /* CPP_ML_DECISION_TREE_HANDLER_H_ */
