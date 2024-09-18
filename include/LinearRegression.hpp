#ifndef CPP_ML_LINEAR_REGRESSION_H_
#define CPP_ML_LINEAR_REGRESSION_H_

#include <cmath>
#include <map>
#include <Eigen/Dense>

class LinearRegression {
  double bias;
  double learningRate;
  unsigned int numberOfIterations;
  Eigen::VectorXd weights;

public:
  LinearRegression(unsigned int numberOfIterations, double learningRate) : 
    numberOfIterations(numberOfIterations), 
    learningRate(learningRate) {}

  void fit(Eigen::VectorXd& labels, Eigen::MatrixXd& features);

  Eigen::VectorXd predict(Eigen::MatrixXd& fetures);
};

#endif /* CPP_ML_LINEAR_REGRESSION_H_ */
