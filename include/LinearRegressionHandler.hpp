#ifndef CPP_ML_LINEAR_REGRESSION_HANDLER_H_
#define CPP_ML_LINEAR_REGRESSION_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "CSVETL.hpp"
#include "LinearRegression.hpp"

int handleLinearRegression(
  std::string datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int> featureColumns, 
  std::vector<bool> featureColumnsAreDigits
) {
  CSVETL csvETL(datasetPath);

  if (!csvETL.load(labelColumn, labelColumIsDigit, featureColumns, featureColumnsAreDigits)) {
    std::cout << "Dataset: " << datasetPath << " not loaded " << std::endl;
    return 1;
  }

  std::cout << "Dataset: " << datasetPath << " loaded " << std::endl;

  Eigen::MatrixXd X = csvETL.getFeatures();

  Eigen::VectorXd y = csvETL.getLabels();

  // std::cout << "Samples:" << std::endl << std::endl;

  // std::cout << X << std::endl << std::endl;

  // std::cout << "Labels:" << std::endl << std::endl;

  // std::cout << y << std::endl << std::endl;

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;

  // CSVETL::DataSplitResult splitResult = csvETL.splitData(y, X);

  // std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  // std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  // std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  // std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  
  return 0;
}

#endif /* CPP_ML_LINEAR_REGRESSION_HANDLER_H_ */
