#ifndef CPP_ML_ML_UTILS_H_
#define CPP_ML_ML_UTILS_H_

#include <map>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

namespace MLUtils {
  std::tuple<double, unsigned int, unsigned int> calculateAccuracy(Eigen::VectorXd predictions, Eigen::VectorXd labels);

  void calculateAndPrintAccuracy(Eigen::VectorXd predictions, Eigen::VectorXd labels);

  std::map<double, unsigned int> labelsFrequency(Eigen::VectorXd &labels);

  double mostCommonLabel(Eigen::VectorXd &labels);
}

#endif /* CPP_ML_ML_UTILS_H_ */
