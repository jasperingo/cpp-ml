#ifndef CPP_ML_LINEAR_REGRESSION_HANDLER_H_
#define CPP_ML_LINEAR_REGRESSION_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include "CSVETL.hpp"
#include "LinearRegression.hpp"

int handleLinearRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  double learningRate,
  int maxNumberOfIterations
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

  LinearRegression linearRegression(maxNumberOfIterations, learningRate);
  
  Eigen::VectorXd costs = linearRegression.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = linearRegression.predict(splitResult.testSamples);

  // std::cout << "Predictions: " << predictions << std::endl;
  // std::cout << "Number of predictions: " << predictions.size() << std::endl;
  
  // std::cout << "Costs: " << std::endl << costs << std::endl;

  double mse = linearRegression.meanSquaredError(splitResult.testLabels, predictions);

  std::cout << "Test Mean squared error: " << mse << std::endl;

  std::vector<double> plotX(X.data(), X.data() + X.size());
  std::vector<double> plotY(y.data(), y.data() + y.size());

  Eigen::VectorXd allPredictions = linearRegression.predict(X);

  std::vector<double> plotLine(allPredictions.data(), allPredictions.data() + allPredictions.size());

  try {
    matplot::scatter(plotX, plotY);

    matplot::hold(matplot::on);

    matplot::plot(plotX, plotLine);
    
    matplot::hold(matplot::off);

    matplot::show();
  } catch (std::runtime_error& error) {
    std::cout << "Plotting rror: " << error.what() << std::endl;
  }

  return 0;
}

int handleLinearRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits
) {
  return handleLinearRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    0.001,
    1000
  );
}

int handleLinearRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  int maxNumberOfIterations
) {
  return handleLinearRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    0.001,
    maxNumberOfIterations
  );
}

int handleLinearRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  double learningRate
) {
  return handleLinearRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    learningRate,
    1000
  );
}

#endif /* CPP_ML_LINEAR_REGRESSION_HANDLER_H_ */
