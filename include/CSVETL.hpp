#ifndef CPP_ML_CSVETL_H_
#define CPP_ML_CSVETL_H_

#include <stdlib.h>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>

class CSVETL {
  char delimiter;
  std::string filePath;
  std::vector<std::vector<std::string>> data;
  Eigen::VectorXd labels;
  Eigen::MatrixXd features;

  bool loadFile();

  bool extractLabels(unsigned int labelColumn, bool labelColumIsDigit);

  bool extractFeatures(std::vector<unsigned int> featureColumns, std::vector<bool> featureColumnsAreDigits);

public:
  struct DataSplitResult {
    Eigen::VectorXd trainLabels;
    Eigen::VectorXd testLabels;
    Eigen::MatrixXd trainSamples;
    Eigen::MatrixXd testSamples;
  };

  CSVETL(std::string& filePath, char delimiter = ',') : filePath(filePath), delimiter(delimiter) {}

  Eigen::VectorXd& getLabels() {
    return labels;
  }

  Eigen::MatrixXd& getFeatures() {
    return features;
  }

  bool load(unsigned int labelColumn, bool labelColumIsDigit, std::vector<unsigned int> featureColumns, std::vector<bool> featureColumnsAreDigits);

  DataSplitResult splitData(Eigen::VectorXd& labels, Eigen::MatrixXd& samples, float testSize = 0.2);
};

#endif /* CPP_ML_CSVETL_H_ */
