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

  Eigen::MatrixXd X = config.etl.getFeatures();

  Eigen::VectorXd y = config.etl.getLabels();

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;
  std::cout << "Number of labels: " << y.size() << std::endl;

  CSVETL::DataSplitResult splitResult = config.etl.splitData(y, X);

  std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  RandomForest randomForest(config.numberOfTrees, config.maxDepth, config.minSampleSplit, config.numberOfFeatures);

  randomForest.fit(splitResult.trainSamples, splitResult.trainLabels);

  Eigen::VectorXd predictions = randomForest.predict(splitResult.testSamples);

  MLUtils::calculateAndPrintAccuracy(predictions, splitResult.testLabels);
}

#endif /* CPP_ML_RANDOM_FOREST_HANDLER_H_ */
