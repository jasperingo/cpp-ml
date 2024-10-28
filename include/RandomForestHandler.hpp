#ifndef CPP_ML_RANDOM_FOREST_HANDLER_H_
#define CPP_ML_RANDOM_FOREST_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "CSVETL.hpp"
#include "MLUtils.hpp"
#include "RandomForest.hpp"

struct RandomForestConfig {
  CSVETL& etl;
  unsigned int numberOfTrees = 5;
  unsigned int maxDepth = 100;
  unsigned int minSampleSplit = 2;
  unsigned int numberOfFeatures = 0;

  RandomForestConfig(CSVETL& etl): etl(etl) {}
};

void handleRandomForest(RandomForestConfig& config) {
  CSVETL::DataSplitResult splitResult = config.etl.splitData();

  config.etl.printDatasetSplitSize(splitResult);

  RandomForest randomForest(config.numberOfTrees, config.maxDepth, config.minSampleSplit, config.numberOfFeatures);

  randomForest.fit(splitResult.trainSamples, splitResult.trainLabels);

  Eigen::VectorXd predictions = randomForest.predict(splitResult.testSamples);

  MLUtils::calculateAndPrintAccuracy(predictions, splitResult.testLabels);
}

#endif /* CPP_ML_RANDOM_FOREST_HANDLER_H_ */
