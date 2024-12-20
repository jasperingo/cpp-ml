#ifndef CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_
#define CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_

#include <numeric>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include "CSVETL.hpp"
#include "MLUtils.hpp"
#include "LogisticRegression.hpp"

struct LogisticRegressionConfig {
  CSVETL& etl;
  double learningRate = 0.001;
  unsigned int maxNumberOfIterations = 1000;

  LogisticRegressionConfig(CSVETL& etl): etl(etl) {}
};

void handleLogisticRegression(LogisticRegressionConfig& config) {
  CSVETL::DataSplitResult splitResult = config.etl.splitData();

  config.etl.printDatasetSplitSize(splitResult);

  LogisticRegression logisticRegression(config.maxNumberOfIterations, config.learningRate);
  
  Eigen::VectorXd costs = logisticRegression.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = logisticRegression.predict(splitResult.testSamples);

  Eigen::VectorXd booleanPredictions = predictions.unaryExpr([](double x) { return x > 0.5 ? 1.0 : 0.0; });
  
  MLUtils::calculateAndPrintAccuracy(booleanPredictions, splitResult.testLabels);

  // Eigen::VectorXd X1 = X.col(0);

  // std::vector<double> plotX(X1.data(), X1.data() + X1.size());
  // std::vector<double> plotY(y.data(), y.data() + y.size());

  // Eigen::VectorXd allPredictions = logisticRegression.predict(X).unaryExpr([](double x) { return x > 0.5 ? 1.0 : 0.0; });

  // std::vector<double> plotLine(allPredictions.data(), allPredictions.data() + allPredictions.size());

  try {
    // matplot::scatter(plotX, plotY);

    // matplot::hold(matplot::on);

    std::vector<int> plotX(config.maxNumberOfIterations);
    
    std::iota(plotX.begin(), plotX.end(), 0);

    std::vector<double> plotLine(costs.data(), costs.data() + costs.size());

    matplot::plot(plotX, plotLine);
    matplot::ylabel("Costs");
    matplot::xlabel("Number of iterations");
    
    // matplot::hold(matplot::off);

    matplot::show();
  } catch (std::runtime_error& error) {
    std::cout << "Plotting rror: " << error.what() << std::endl;
  }
}

#endif /* CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_ */
