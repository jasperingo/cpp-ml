#ifndef CPP_ML_KNN_HANDLER_H_
#define CPP_ML_KNN_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"
#include "MLUtils.hpp"

struct KNNConfig {
  CSVETL& etl;
  unsigned int numberOfK = 3;

  KNNConfig(CSVETL& etl): etl(etl) {}
};

void handleKNN(KNNConfig& config) {
  CSVETL::DataSplitResult splitResult = config.etl.splitData();
  
  config.etl.printDatasetSplitSize(splitResult);

  KNN knn(config.numberOfK);

  knn.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = knn.predict(splitResult.testSamples);

  MLUtils::calculateAndPrintAccuracy(predictions, splitResult.testLabels);
}

#endif /* CPP_ML_KNN_HANDLER_H_ */
