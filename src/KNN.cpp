#include "KNN.hpp"

void KNN::fit(Eigen::VectorXd labels, Eigen::MatrixXd samples) {
  trainLabels = labels;
  trainSamples = samples;
}

Eigen::VectorXd KNN::predict(Eigen::MatrixXd samples) {
  Eigen::VectorXd predictions(samples.rows());

  for (int i = 0; i < samples.rows(); i++) {
    predictions(i) = predict(samples.row(i));
  }

  return predictions;
}

double KNN::predict(Eigen::Block<Eigen::MatrixXd, 1> features) {
  return 0.0;
}

double KNN::euclideanDistance(Eigen::Block<Eigen::MatrixXd, 1> test, Eigen::Block<Eigen::MatrixXd, 1> train) {
  // test.pow(2);
  // double d = test - train;
  return 0.0;
}
