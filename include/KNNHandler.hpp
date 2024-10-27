#ifndef CPP_ML_KNN_HANDLER_H_
#define CPP_ML_KNN_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"

struct KNNConfig {
  CSVETL& etl;
  unsigned int numberOfK = 3;

  KNNConfig(CSVETL& etl): etl(etl) {}
};

void handleKNN(KNNConfig& config) {
  Eigen::MatrixXd X = config.etl.getFeatures();

  Eigen::VectorXd y = config.etl.getLabels();

  // std::cout << "Samples:" << std::endl << std::endl;

  // std::cout << X << std::endl << std::endl;

  // std::cout << "Labels:" << std::endl << std::endl;

  // std::cout << y << std::endl << std::endl;

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;

  CSVETL::DataSplitResult splitResult = config.etl.splitData(y, X);

  std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  KNN knn(config.numberOfK);

  knn.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = knn.predict(splitResult.testSamples);

  int correctCount = 0;
  int incorrectCount = 0;

  for (int i = 0; i < predictions.rows(); i++) {
    double prediction = predictions(i);
    double testLabel = splitResult.testLabels(i);

    if (prediction == testLabel) {
      correctCount++;
    } else {
      incorrectCount++;
    }
  }

  double accuracy = ((double) correctCount) / splitResult.testLabels.rows();

  std::cout << "Model accuracy: " << accuracy << std::endl;
  std::cout << "Model accuracy %: " << (accuracy * 100) << std::endl;
  std::cout << "Number of correct predictions: " << correctCount << std::endl;
  std::cout << "Number of incorrect predictions: " << incorrectCount << std::endl;

}

#endif /* CPP_ML_KNN_HANDLER_H_ */
