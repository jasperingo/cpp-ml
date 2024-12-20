#ifndef CPP_ML_LINEAR_REGRESSION_HANDLER_H_
#define CPP_ML_LINEAR_REGRESSION_HANDLER_H_

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include "CSVETL.hpp"
#include "LinearRegression.hpp"

struct LinearRegressionConfig {
  CSVETL& etl;
  double learningRate = 0.001;
  unsigned int maxNumberOfIterations = 1000;

  LinearRegressionConfig(CSVETL& etl): etl(etl) {}
};

void handleLinearRegression(LinearRegressionConfig& config) {
  Eigen::MatrixXd X = config.etl.getFeatures();

  Eigen::VectorXd y = config.etl.getLabels();

  CSVETL::DataSplitResult splitResult = config.etl.splitData();
  
  config.etl.printDatasetSplitSize(splitResult);

  LinearRegression linearRegression(config.maxNumberOfIterations, config.learningRate);
  
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
}

#endif /* CPP_ML_LINEAR_REGRESSION_HANDLER_H_ */
