#ifndef CPP_ML_CSVETL_H_
#define CPP_ML_CSVETL_H_

#include <stdlib.h>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>

class CSVETL {
  char delimiter;
  std::string filePath;
  std::vector<std::vector<std::string>> data; 

  int findIndexInColumns(size_t column, size_t columns[], size_t columnsLength);

public:
  struct DataSplitResult {
    Eigen::VectorXd trainLabels;
    Eigen::VectorXd testLabels;
    Eigen::MatrixXd trainSamples;
    Eigen::MatrixXd testSamples;
  };

  CSVETL(std::string filePath, char delimiter = ',') : filePath(filePath), delimiter(delimiter) {}

  bool load();

  Eigen::VectorXd extractLabels(size_t labelColumn);

  Eigen::MatrixXd extractSamples(size_t sampleColumns[], size_t sampleColumnsLength);

  DataSplitResult splitData(Eigen::VectorXd labels, Eigen::MatrixXd samples, float testSize = 0.2);
};

#endif /* CPP_ML_CSVETL_H_ */
