#include "LogisticRegression.hpp"

Eigen::VectorXd LogisticRegression::sigmoid(const Eigen::ArrayXd &predictions) {
  return 1.0 / (1.0 + (-1.0 * predictions).exp());
}

Eigen::VectorXd LogisticRegression::makePredictions(const Eigen::MatrixXd &features) {
  Eigen::ArrayXd featuresXWeight = features * weights;

  return sigmoid(featuresXWeight + bias);
}

Eigen::VectorXd LogisticRegression::fit(Eigen::VectorXd &labels, Eigen::MatrixXd &features) {
  const size_t numberOfSamples = features.rows();
  const size_t numberOfFeatures = features.cols();
  bias = 0.0;
  weights = Eigen::VectorXd::Zero(numberOfFeatures);
  
  Eigen::MatrixXd featuresTranspose = features.transpose();

  Eigen::VectorXd costs(numberOfIterations);

  for (unsigned int i = 0; i < numberOfIterations; i++) {
    Eigen::VectorXd predictions = makePredictions(features);

    Eigen::VectorXd predictionsSubtractLabels = predictions - labels;

    double dw = (1.0 / ((double) numberOfSamples)) * (featuresTranspose * predictionsSubtractLabels).sum();
    double db = (1.0 / ((double) numberOfSamples)) * predictionsSubtractLabels.sum();

    weights = weights.array() - (learningRate * dw);
    bias = bias - (learningRate * db);

    costs(i) = crossEntropy(labels, predictions);

    if (i % (numberOfIterations / 10) == 0) {
      std::cout << "Cost after " << i << " iterations: " << costs(i) << std::endl;
    }
  }

  std::cout << "Cost after all iterations: " << costs(numberOfIterations - 1) << std::endl;

  return costs;
}

Eigen::VectorXd LogisticRegression::predict(Eigen::MatrixXd& features) {
  return makePredictions(features);
}

double LogisticRegression::crossEntropy(Eigen::VectorXd &labels, Eigen::VectorXd &predictions) {
  Eigen::RowVectorXd predictionsTranspose = predictions.transpose();
  Eigen::RowVectorXd predictionsTransposeLog = predictionsTranspose.array().log();
  Eigen::RowVectorXd predictionsTransposeLogMinusOne = (1.0 - predictionsTranspose.array()).log();
  return -((1.0 / ((double) labels.size())) * (((labels * predictionsTransposeLog)) + ((1.0 - labels.array()).matrix() * predictionsTransposeLogMinusOne)).sum());
}
