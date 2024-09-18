#include "LinearRegression.hpp"

void LinearRegression::fit(Eigen::VectorXd& labels, Eigen::MatrixXd& features) {
  size_t numberOfSamples = features.rows();
  size_t numberOfFeatures = features.cols();
  bias = 0.0;
  weights = Eigen::VectorXd(numberOfFeatures);

  for (size_t i = 0; i < numberOfFeatures; i++) {
    weights(i) = 0;
  }

  Eigen::ArrayXd featuresXWeight = features * weights;

  Eigen::VectorXd predictions = featuresXWeight + bias;

  Eigen::VectorXd predictionsSubtractLabels = predictions - labels;

  double dw = (1 / numberOfSamples) * ((features * predictionsSubtractLabels).array().sum() * 2);
  double db = (1 / numberOfSamples) * (predictionsSubtractLabels.array().sum() * 2);


}

Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd& features) {

  return Eigen::VectorXd();
}
