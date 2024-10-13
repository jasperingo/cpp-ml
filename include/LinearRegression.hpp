#ifndef CPP_ML_LINEAR_REGRESSION_H_
#define CPP_ML_LINEAR_REGRESSION_H_

#include <Eigen/Dense>

class LinearRegression {
  double bias;
  double learningRate;
  unsigned int numberOfIterations;
  Eigen::VectorXd weights;

  Eigen::VectorXd makePredictions(Eigen::MatrixXd& features);

public:
  LinearRegression(unsigned int numberOfIterations, double learningRate) : 
    numberOfIterations(numberOfIterations), 
    learningRate(learningRate) {}

  Eigen::VectorXd fit(Eigen::VectorXd& labels, Eigen::MatrixXd& features);

  Eigen::VectorXd predict(Eigen::MatrixXd& fetures);

  double meanSquaredError(Eigen::VectorXd& labels, Eigen::VectorXd& predictions);
};

#endif /* CPP_ML_LINEAR_REGRESSION_H_ */
