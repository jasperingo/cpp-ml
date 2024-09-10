#ifndef CPP_ML_CSVETL_H_
#define CPP_ML_CSVETL_H_

#include <stdlib.h>
#include <string>
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
  CSVETL(std::string filePath, char delimiter = ',') : filePath(filePath), delimiter(delimiter) {}

  bool load();

  Eigen::VectorXd extractLabels(size_t labelColumn);

  Eigen::MatrixXd extractSamples(size_t sampleColumns[], size_t sampleColumnsLength);
};

#endif /* CPP_ML_CSVETL_H_ */
