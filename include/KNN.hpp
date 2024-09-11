#ifndef CPP_ML_KNN_H_
#define CPP_ML_KNN_H_

#include <vector>
#include <Eigen/Dense>

class KNN {
  size_t K;
  Eigen::VectorXd trainLabels;
  Eigen::MatrixXd trainSamples;

  double predict(Eigen::Block<Eigen::MatrixXd, 1> features);

  double euclideanDistance(Eigen::Block<Eigen::MatrixXd, 1> test, Eigen::Block<Eigen::MatrixXd, 1> train);

public:
  KNN(size_t K) : K(K) {}

  void fit(Eigen::VectorXd labels, Eigen::MatrixXd samples);

  Eigen::VectorXd predict(Eigen::MatrixXd samples);
};

#endif /* CPP_ML_KNN_H_ */
