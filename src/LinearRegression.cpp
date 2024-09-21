#include "LinearRegression.hpp"

Eigen::VectorXd LinearRegression::makePredictions(Eigen::MatrixXd &features) {
  Eigen::ArrayXd featuresXWeight = features * weights;

  return featuresXWeight + bias;
}

void LinearRegression::fit(Eigen::VectorXd &labels, Eigen::MatrixXd &features) {
  size_t numberOfSamples = features.rows();
  size_t numberOfFeatures = features.cols();
  bias = 0.0;
  weights = Eigen::VectorXd(numberOfFeatures);

  for (size_t i = 0; i < numberOfFeatures; i++) {
    weights(i) = 0;
  }

  double oneDivideSamples = (1.0 / (double) numberOfSamples);
  
  Eigen::MatrixXd featuresTranspose = features.transpose();

  for (unsigned int i = 0; i < numberOfIterations; i++) {
    Eigen::VectorXd predictions = makePredictions(features);

    Eigen::VectorXd predictionsSubtractLabels = predictions - labels;

    double dw = oneDivideSamples * (2 * (featuresTranspose * predictionsSubtractLabels)).sum();
    double db = oneDivideSamples * (2 * predictionsSubtractLabels).sum();

    weights = weights.array() - (learningRate * dw);
    bias = bias - (learningRate * db);
  }
}

Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd& features) {
  return makePredictions(features);
}

double LinearRegression::meanSquaredError(Eigen::VectorXd &labels, Eigen::VectorXd &predictions) {
  return (labels - predictions).array().pow(2).mean();
}
