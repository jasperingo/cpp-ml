#ifndef CPP_ML_LOGISTIC_REGRESSION_H_
#define CPP_ML_LOGISTIC_REGRESSION_H_

#include <iostream>
#include <cmath>
#include <map>
#include <Eigen/Dense>

class LogisticRegression {
  double bias;
  double learningRate;
  unsigned int numberOfIterations;
  Eigen::VectorXd weights;

  Eigen::VectorXd sigmoid(const Eigen::VectorXd& predictions);

  Eigen::VectorXd makePredictions(const Eigen::MatrixXd& features);

public:
  LogisticRegression(unsigned int numberOfIterations, double learningRate) : 
    numberOfIterations(numberOfIterations), 
    learningRate(learningRate) {}

  Eigen::VectorXd fit(Eigen::VectorXd& labels, Eigen::MatrixXd& features);

  Eigen::VectorXd predict(Eigen::MatrixXd& fetures);

  double crossEntropy(Eigen::VectorXd& labels, Eigen::VectorXd& predictions);
};

#endif /* CPP_ML_LOGISTIC_REGRESSION_H_ */
