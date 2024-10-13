#ifndef CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_
#define CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_

#include <numeric>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include "CSVETL.hpp"
#include "LogisticRegression.hpp"

int handleLogisticRegression(
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

  LogisticRegression logisticRegression(maxNumberOfIterations, learningRate);
  
  Eigen::VectorXd costs = logisticRegression.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = logisticRegression.predict(splitResult.testSamples);

  Eigen::VectorXd predictionX(predictions.size());

  int correctCount = 0;
  int incorrectCount = 0;

  for (int i = 0; i < predictions.size(); i++) {
    double prediction = predictions(i) > 0.5 ? 1 : 0;
    double testLabel = splitResult.testLabels(i);
    predictionX(i) = prediction;

    if (prediction == testLabel) {
      correctCount++;
    } else {
      incorrectCount++;
    }
  }

  // std::cout << "Predictions: " << predictionX << std::endl;

  double accuracy = ((double) correctCount) / splitResult.testLabels.size();

  std::cout << "Model accuracy: " << accuracy << std::endl;
  std::cout << "Model accuracy %: " << (accuracy * 100) << std::endl;
  std::cout << "Number of correct predictions: " << correctCount << std::endl;
  std::cout << "Number of incorrect predictions: " << incorrectCount << std::endl;

  // Eigen::VectorXd X1 = X.col(0);

  // std::vector<double> plotX(X1.data(), X1.data() + X1.size());
  // std::vector<double> plotY(y.data(), y.data() + y.size());

  // Eigen::VectorXd allPredictions = logisticRegression.predict(X).unaryExpr([](double x) { return x > 0.5 ? 1.0 : 0.0; });

  // std::vector<double> plotLine(allPredictions.data(), allPredictions.data() + allPredictions.size());

  try {
    // matplot::scatter(plotX, plotY);

    // matplot::hold(matplot::on);

    std::vector<int> plotX(maxNumberOfIterations);
    
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

  return 0;
}

int handleLogisticRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits
) {
  return handleLogisticRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    0.001,
    1000
  );
}

int handleLogisticRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  int maxNumberOfIterations
) {
  return handleLogisticRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    0.001,
    maxNumberOfIterations
  );
}

int handleLogisticRegression(
  std::string& datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int>& featureColumns, 
  std::vector<bool>& featureColumnsAreDigits,
  double learningRate
) {
  return handleLogisticRegression(
    datasetPath, 
    labelColumn, 
    labelColumIsDigit, 
    featureColumns, 
    featureColumnsAreDigits,
    learningRate,
    1000
  );
}

#endif /* CPP_ML_LOGISTIC_REGRESSION_HANDLER_H_ */
