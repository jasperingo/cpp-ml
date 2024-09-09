#ifndef CPP_ML_CSVETL_H_
#define CPP_ML_CSVETL_H_

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

class CSVETL {
  char delimiter;
  std::string filePath;
  std::vector<std::vector<std::string>> data; 

public:
  CSVETL(std::string filePath, char delimiter = ',') : filePath(filePath), delimiter(delimiter) {}

  bool load();
};

#endif /* CPP_ML_CSVETL_H_ */
